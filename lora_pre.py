import math
import queue
import random
import threading
import traceback
import types
import torch
from torch import nn
from typing import Callable, Iterable, Optional
from torch.optim import Optimizer


# ============================================================
# Projection generator
# ============================================================


class SketchMatrixGenerator:
    def __init__(self, in_dim: int, k: int, base_seed: int, sketch_ttl: int, step_idx: int = 0):
        self.in_dim = in_dim
        self.k = k
        self.k_sqrt = math.sqrt(k)
        self.base_seed = base_seed
        self.sketch_ttl = sketch_ttl
        self.step_idx = step_idx
        self._gen = None

    def copy(self) -> "SketchMatrixGenerator":
        return SketchMatrixGenerator(self.in_dim, self.k, self.base_seed, self.sketch_ttl, self.step_idx)

    def materialize_p(self, device, dtype):
        if self._gen is None:
            self._gen = torch.Generator(device)

        if self.sketch_ttl <= 1:
            self._gen.manual_seed(self.base_seed)
            return torch.randn(
                self.in_dim, self.k, device=device, dtype=dtype, generator=self._gen
            ) / self.k_sqrt

        alpha = float(self.step_idx) / float(self.sketch_ttl - 1)
        w0 = 1.0 - alpha
        w1 = alpha
        norm = math.sqrt(w0 * w0 + w1 * w1)

        self._gen.manual_seed(self.base_seed)
        p0 = torch.randn(
            self.in_dim,
            self.k,
            device=device,
            dtype=dtype,
            generator=self._gen,
        ) / self.k_sqrt
        self._gen.manual_seed((self.base_seed + 1) % (2**64))
        p1 = torch.randn(
            self.in_dim,
            self.k,
            device=device,
            dtype=dtype,
            generator=self._gen,
        ) / self.k_sqrt
        p0.mul_(w0).add_(p1, alpha=w1).div_(norm)
        return p0

    def bump_seed(self):
        self.step_idx = (self.step_idx + 1) % self.sketch_ttl
        if self.step_idx == 0:
            self.base_seed = (self.base_seed + 1) % (2**64)


# ============================================================
# Hook helpers (per-module context)
# ============================================================


def _ensure_lr_context(module: nn.Module):
    """
    Attach a simple context dict to a module for storing per-forward
    quantities like Z (sketched activations).
    """
    if not hasattr(module, "_lr_ctx"):
        module._lr_ctx = {"z_stack": []}
    return module._lr_ctx


# ============================================================
# Hooks for Linear
# ============================================================


class LowRankLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        proj: SketchMatrixGenerator,
    ):
        projection_matrix = proj.materialize_p(
            device=input_tensor.device,
            dtype=weight.cpu_fp16.dtype,
        )
        sketch = input_tensor @ projection_matrix

        ctx.save_for_backward(sketch, weight)
        ctx.has_bias = bias is not None

        weight_fp16 = fp8_to_fp16_with_scale(weight.detach(), weight.fp8_scale)
        output = torch.nn.functional.linear(input_tensor, weight_fp16, bias)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_output, *_ = grad_outputs
        sketch, weight = ctx.saved_tensors
        has_bias = ctx.has_bias

        weight_fp16 = fp8_to_fp16_with_scale(weight.detach(), weight.fp8_scale)
        grad_input = grad_output @ weight_fp16

        reduce_dims = tuple(range(grad_output.ndim - 1))
        grad_bias = grad_output.sum(dim=reduce_dims) if has_bias else None

        if getattr(weight, "requires_grad_lr", False):
            go = grad_output.reshape(-1, grad_output.shape[-1])
            sk = sketch.reshape(-1, sketch.shape[-1])
            sketched_grad = go.mT @ sk

            pre_hooks = getattr(weight, "lr_pre_accumulate_hooks", None)
            if pre_hooks is not None:
                for h in pre_hooks:
                    out = h(weight, sketched_grad)
                    if out is not None:
                        sketched_grad = out

            if getattr(weight, "grad_lr", None) is None:
                weight.grad_lr = sketched_grad
            else:
                weight.grad_lr = weight.grad_lr + sketched_grad

            hooks = getattr(weight, "lr_post_accumulate_hooks", None)
            if hooks is not None:
                for h in hooks:
                    h(weight)

        return grad_input, None, grad_bias, None


class LowRankConv2dFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        proj: SketchMatrixGenerator,
        stride,
        padding,
        dilation,
        groups: int,
    ):
        weight_fp16 = fp8_to_fp16_with_scale(weight.detach(), weight.fp8_scale)
        output = torch.nn.functional.conv2d(
            input_tensor,
            weight_fp16,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

        batch_size, in_channels, _, _ = input_tensor.shape
        _, _, kernel_height, kernel_width = weight.shape

        projection_matrix = proj.materialize_p(
            device=input_tensor.device,
            dtype=weight.cpu_fp16.dtype,
        ).T.reshape(proj.k, in_channels, kernel_height, kernel_width).contiguous()
        sketch = torch.nn.functional.conv2d(
            input_tensor,
            projection_matrix,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=1,
        )

        ctx.save_for_backward(sketch, weight)
        ctx.input_shape = input_tensor.shape
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.has_bias = bias is not None

        return output

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_output, *_ = grad_outputs
        sketch, weight = ctx.saved_tensors

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        has_bias = ctx.has_bias

        if groups != 1:
            raise RuntimeError("LowRankConv2dFn currently supports groups=1 only")

        if sketch.shape[0] != grad_output.shape[0] or sketch.shape[2:] != grad_output.shape[2:]:
            raise RuntimeError("sketch shape mismatch with grad_output")

        in_h, in_w = ctx.input_shape[2], ctx.input_shape[3]
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        s_h, s_w = (stride if isinstance(stride, tuple) else (stride, stride))
        p_h, p_w = (padding if isinstance(padding, tuple) else (padding, padding))
        d_h, d_w = (dilation if isinstance(dilation, tuple) else (dilation, dilation))

        k_h, k_w = weight.shape[2], weight.shape[3]
        eff_kh = d_h * (k_h - 1) + 1
        eff_kw = d_w * (k_w - 1) + 1

        out_pad_h = in_h - ((out_h - 1) * s_h - 2 * p_h + eff_kh)
        out_pad_w = in_w - ((out_w - 1) * s_w - 2 * p_w + eff_kw)

        output_padding = (int(out_pad_h), int(out_pad_w))

        weight_fp16 = fp8_to_fp16_with_scale(weight.detach(), weight.fp8_scale)
        grad_input = torch.nn.functional.conv_transpose2d(
            grad_output,
            weight_fp16,
            bias=None,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
        )

        grad_bias = grad_output.sum(dim=(0, 2, 3)) if has_bias else None

        if getattr(weight, "requires_grad_lr", False):
            sketched_grad = torch.einsum("bohw,bkhw->ok", grad_output, sketch)

            pre_hooks = getattr(weight, "lr_pre_accumulate_hooks", None)
            if pre_hooks is not None:
                for h in pre_hooks:
                    out = h(weight, sketched_grad)
                    if out is not None:
                        sketched_grad = out

            if getattr(weight, "grad_lr", None) is None:
                weight.grad_lr = sketched_grad
            else:
                weight.grad_lr = weight.grad_lr + sketched_grad

            hooks = getattr(weight, "lr_post_accumulate_hooks", None)
            if hooks is not None:
                for h in hooks:
                    h(weight)

        return (
            grad_input,
            None,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
        )


# ============================================================
# Patching helper
# ============================================================


class LrHandle:
    def __init__(self, entries, parameters):
        self.entries = entries
        self.parameters = parameters

    def remove(self):
        for module, orig_forward in self.entries:
            if orig_forward is not None:
                module.forward = orig_forward
        self.entries = ()

        for param in self.parameters:
            for name in ("requires_grad_lr", "grad_lr", "lr_rank", "lr_sketch_rank", "lr_proj"):
                if hasattr(param, name):
                    delattr(param, name)
        self.parameters = ()


def patch_model_lr(
    model: nn.Module,
    rank_ratio: float = 0.05,
    min_dim_threshold: int = 2,
    sketches_ttl: int = 200,
    base_seed: Optional[int] = None,
) -> LrHandle:
    if base_seed is None:
        base_seed = random.randint(0, 2**31 - 1)

    entries = []
    layer_index = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight
            out_dim, in_dim = weight.shape
            min_dim = min(out_dim, in_dim)
            if min_dim <= min_dim_threshold:
                continue

            r = max(int(min_dim * rank_ratio), 1)
            if r >= min_dim:
                continue

            k = r

            weight.requires_grad = False
            weight.requires_grad_lr = True
            weight.lr_rank = r
            weight.lr_sketch_rank = k
            weight.cpu_fp16 = weight.to(torch.float16, copy=True).pin_memory()

            weight.lr_proj = SketchMatrixGenerator(
                in_dim=in_dim,
                k=k,
                base_seed=base_seed + layer_index,
                sketch_ttl=sketches_ttl,
            )

            weight.lr_post_accumulate_hooks = []
            weight.lr_pre_accumulate_hooks = []

            orig_forward = module.forward

            def forward(self, x):
                return LowRankLinearFn.apply(x, self.weight, self.bias, self.weight.lr_proj)

            module.forward = types.MethodType(forward, module)
            entries.append((module, orig_forward))
            layer_index += 1
        elif isinstance(module, nn.Conv2d):
            weight = module.weight
            c_out, c_in, k_h, k_w = weight.shape
            d_kernel = c_in * k_h * k_w
            min_dim = min(c_out, d_kernel)
            if min_dim <= min_dim_threshold:
                continue

            if module.groups != 1:
                continue

            r = max(int(min_dim * rank_ratio), 1)
            if r >= min_dim:
                continue

            k = r

            weight.requires_grad = False
            weight.requires_grad_lr = True
            weight.lr_rank = r
            weight.lr_sketch_rank = k
            weight.cpu_fp16 = weight.to(torch.float16, copy=True).pin_memory()

            weight.lr_proj = SketchMatrixGenerator(
                in_dim=d_kernel,
                k=k,
                base_seed=base_seed + layer_index,
                sketch_ttl=sketches_ttl,
            )

            weight.lr_post_accumulate_hooks = []
            weight.lr_pre_accumulate_hooks = []

            orig_forward = module.forward

            def forward(self, x):
                return LowRankConv2dFn.apply(
                    x,
                    self.weight,
                    self.bias,
                    self.weight.lr_proj,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )

            module.forward = types.MethodType(forward, module)
            entries.append((module, orig_forward))
            layer_index += 1

    return LrHandle(entries, model.parameters())


def register_lr_pre_accumulate_hooks(p: torch.nn.Parameter, hook: Callable):
    if not hasattr(p, "lr_pre_accumulate_hooks"):
        raise RuntimeError("Parameter is not low rank. (you need to call patch_model_lr(model))")
    p.lr_pre_accumulate_hooks.append(hook)
    return hook


def register_lr_post_accumulate_hook(p: torch.nn.Parameter, hook: Callable):
    if not hasattr(p, "lr_post_accumulate_hooks"):
        raise RuntimeError("Parameter is not low rank. (you need to call patch_model_lr(model))")
    p.lr_post_accumulate_hooks.append(hook)
    return hook


class Worker:
    def __init__(self, device, maxsize):
        self.queue = queue.Queue(maxsize=maxsize)
        self.device = device
        self.stream = torch.cuda.Stream(device)

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def send(self, fn, *args, **kwargs):
        self.queue.put((fn, args, kwargs))

    def close(self):
        self.send(self._stop_fn)
        self.thread.join()

    def _loop(self):
        with torch.cuda.device(torch.device(self.device).index), torch.cuda.stream(self.stream):
            while True:
                fn, args, kwargs = self.queue.get()
                try:
                    fn(*args, **kwargs)
                except StopIteration:
                    return
                except Exception:
                    traceback.print_exc()
                finally:
                    self.queue.task_done()

    def _stop_fn(self):
        raise StopIteration


class WorkerPool:
    def __init__(self, device: str, num_workers: int = 8, maxsize: int = 0):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")

        self.device = device
        self.workers = [Worker(device, maxsize) for _ in range(num_workers)]
        self._rr = 0
        self._lock = threading.Lock()

    def _pick(self, key=None) -> int:
        if key is None:
            with self._lock:
                i = self._rr
                self._rr = (self._rr + 1) % len(self.workers)
            return i
        # key might be unhashable (e.g., a Tensor/Parameter); id() is stable enough
        return id(key) % len(self.workers)

    def send(self, fn, *args, key=None, **kwargs):
        i = self._pick(key)
        self.workers[i].send(fn, *args, **kwargs)

    def close(self):
        for w in self.workers:
            w.close()


# ============================================================
# Adam + LoRA-Pre with low-rank gradient sketching
# ============================================================


class AdamLoraPre(Optimizer):
    """
    Adam on normal params.
    For low-rank-sketch params (p.requires_grad_lr=True):
      - g_sk = p.grad_lr  (out_dim x k)
      - metric-correct via (P^T P)^-1 using Cholesky (fp32 only here)
      - Adam in subspace (dtype = param dtype)
      - apply update to weights via delta_w = delta_h @ P.T

    Integrated refresh:
      - maintains pinned CPU fp16 masters (p.cpu_fp16)
      - periodically refreshes GPU fp8 params from CPU masters over a K-step cycle
      - synchronizes refresh with any in-flight GPU->CPU master writes via a per-param Event
      - uses a single global optimizer step counter (self.oft_step)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        rank_ratio: float = 0.05,
        betas=(0.9, 0.999),
        eps: float = 1e-5,
        weight_decay: float = 0.0,
        cholesky_eps: float = 1e-4,
        max_grad_norm: Optional[float] = None,
        match_energy: bool = False,
        cpu_refresh_ttl: int = 200,
        refresh_seed: int = 0,
        refresh_stream: Optional[torch.cuda.Stream] = None,
        worker_device: str = "cuda:0",
        num_workers: int = 8,
        worker_queue_maxsize: int = 0,
    ):
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2")

        self.worker = WorkerPool(worker_device, num_workers=num_workers, maxsize=worker_queue_maxsize)

        defaults = dict(
            lr=lr,
            rank_ratio=rank_ratio,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            cholesky_eps=cholesky_eps,
            max_grad_norm=max_grad_norm,
            match_energy=match_energy,
            refresh_K=cpu_refresh_ttl,
        )
        super().__init__(params, defaults)

        self.oft_step = 0

        self._refresh_k = cpu_refresh_ttl
        self._refresh_stream = refresh_stream if refresh_stream is not None else torch.cuda.Stream()
        self._refresh_done = torch.cuda.Event()
        self._refresh_gen = torch.Generator(device="cpu")
        self._refresh_gen.manual_seed(refresh_seed)
        self._refresh_cycle_step = 0
        self._refresh_perm: list[int] = []
        self._refresh_params: list[torch.nn.Parameter] = []

        for group in self.param_groups:
            group_rank_ratio = group["rank_ratio"]

            for p in group["params"]:
                if p is None:
                    continue

                state = self.state[p]

                lr_flag = getattr(p, "requires_grad_lr", False)
                is_matrix = lr_flag and (p.dim() >= 2)

                if is_matrix:
                    out_dim = p.shape[0]
                    in_dim = p.shape[1:].numel()
                    min_dim = min(out_dim, in_dim)

                    r = getattr(p, "lr_rank", None)
                    if r is None:
                        r = max(int(min_dim * group_rank_ratio), 1)
                        r = min(r, min_dim)

                    k = getattr(p, "lr_sketch_rank", r)

                    state["mode"] = 1
                    state["shape_2d"] = (out_dim, in_dim)
                    state["sketch_rank"] = k

                    state["exp_avg_lr"] = torch.zeros((out_dim, k), device=p.device, dtype=p.dtype)
                    state["exp_avg_sq_lr"] = torch.zeros((out_dim, k), device=p.device, dtype=p.dtype)

                    # event: set => CPU master is safe to read/copy; cleared => worker writing CPU master
                    if hasattr(p, "cpu_fp16"):
                        state["cpu_ready_evt"] = threading.Event()
                        state["cpu_ready_evt"].set()
                        self._refresh_params.append(p)
                else:
                    state["mode"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

        if self._refresh_params:
            seen = set()
            uniq = []
            for p in self._refresh_params:
                i = id(p)
                if i in seen:
                    continue
                seen.add(i)
                uniq.append(p)
            self._refresh_params = uniq

        self._refresh_n = len(self._refresh_params)
        self._refresh_new_cycle()

    def close(self):
        self.worker.close()

    def _refresh_new_cycle(self):
        self._refresh_cycle_step = 0
        if self._refresh_n == 0:
            self._refresh_perm = []
        else:
            self._refresh_perm = torch.randperm(self._refresh_n, generator=self._refresh_gen).tolist()

    def _refresh_pick_indices(self) -> list[int]:
        if self._refresh_k <= 0 or self._refresh_n == 0:
            return []
        if self._refresh_cycle_step >= self._refresh_k:
            self._refresh_new_cycle()

        s = self._refresh_cycle_step
        self._refresh_cycle_step += 1

        start = (s * self._refresh_n) // self._refresh_k
        end = ((s + 1) * self._refresh_n) // self._refresh_k
        if start >= end:
            return []
        return self._refresh_perm[start:end]

    def _refresh_from_cpu(self):
        idxs = self._refresh_pick_indices()
        if not idxs:
            with torch.cuda.stream(self._refresh_stream):
                pass
            self._refresh_done.record(self._refresh_stream)
            torch.cuda.current_stream().wait_event(self._refresh_done)
            return

        # host-side: ensure CPU master is not being written
        for i in idxs:
            p = self._refresh_params[i]
            evt: threading.Event = self.state[p]["cpu_ready_evt"]
            evt.wait()

        with torch.cuda.stream(self._refresh_stream):
            for i in idxs:
                p = self._refresh_params[i]

                # H2D + fp16->fp8 on GPU
                w_fp16 = p.cpu_fp16.to(device=p.device, dtype=torch.float16, non_blocking=True)
                w_fp8, scale = fp16_to_fp8_with_rescaling(
                    w_fp16,
                    out_device=p.device,
                    axis_muliscale=True,
                    scale_dim=0,
                )
                p.data.copy_(w_fp8, non_blocking=True)
                p.fp8_scale = scale

        self._refresh_done.record(self._refresh_stream)
        torch.cuda.current_stream().wait_event(self._refresh_done)

    @staticmethod
    def _finish_in_subspace_update_and_signal(
        producer_event,
        p_cpu,
        delta_w,          # (out_dim, in_dim) on GPU fp16
        out_dim, in_dim,
        lr: float,
        weight_decay: float,
        cpu_ready_evt: threading.Event,
    ):
        cpu_device = p_cpu.device

        consumer_stream = torch.cuda.current_stream()
        consumer_stream.wait_event(producer_event)

        delta_w_cpu = delta_w.to(cpu_device)

        done = torch.cuda.Event()
        done.record(consumer_stream)
        done.synchronize()

        w_mat_cpu = p_cpu.data.view(out_dim, in_dim)
        if weight_decay != 0.0:
            w_mat_cpu.mul_(1.0 - lr * weight_decay)
        w_mat_cpu.add_(delta_w_cpu)

        cpu_ready_evt.set()

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.oft_step += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            cholesky_eps = group["cholesky_eps"]
            max_grad_norm = group["max_grad_norm"]
            match_energy = group["match_energy"]

            for param in group["params"]:
                if param is None:
                    continue

                st = self.state[param]
                lr_flag = getattr(param, "requires_grad_lr", False)

                # -------- normal Adam --------
                if not lr_flag or st["mode"] == 0:
                    grad = param.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("AdamLoraPre does not support sparse gradients")

                    if weight_decay != 0.0:
                        param.data.add_(param.data, alpha=-lr * weight_decay)

                    exp_avg = st["exp_avg"]
                    exp_avg_sq = st["exp_avg_sq"]

                    g = grad
                    if max_grad_norm is not None:
                        g_norm = g.norm()
                        if g_norm > max_grad_norm:
                            g = g * (max_grad_norm / (g_norm + 1e-6))

                    exp_avg.mul_(beta1).add_(g, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1**self.oft_step
                    bias_correction2 = 1.0 - beta2**self.oft_step
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    denom = exp_avg_sq.sqrt().add_(eps)
                    param.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                # -------- low-rank sketch path --------
                g_sk = getattr(param, "grad_lr", None)
                proj: SketchMatrixGenerator = getattr(param, "lr_proj", None)

                if g_sk is None:
                    if proj is not None:
                        proj.bump_seed()
                    continue
                if proj is None:
                    raise RuntimeError("Low-rank parameter is missing lr_proj")

                out_dim, in_dim = st["shape_2d"]
                k = st["sketch_rank"]

                # materialize P once
                P = proj.materialize_p(device=param.device, dtype=torch.float16)  # (in_dim, k)

                # fp32 ONLY here: metric correction
                G = (P.T.float() @ P.float())
                G.diagonal().add_(cholesky_eps)
                L, _ = safe_cholesky(G, cholesky_eps)

                g = torch.cholesky_solve(g_sk.T.float(), L, upper=False).T.to(dtype=torch.float16)

                if max_grad_norm is not None:
                    g_norm = g.norm()
                    if g_norm > max_grad_norm:
                        g = g * (max_grad_norm / (g_norm + 1e-6))

                exp_avg_lr = st["exp_avg_lr"]
                exp_avg_sq_lr = st["exp_avg_sq_lr"]

                exp_avg_lr.mul_(beta1).add_(g, alpha=1.0 - beta1)
                exp_avg_sq_lr.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**self.oft_step
                bias_correction2 = 1.0 - beta2**self.oft_step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if match_energy:
                    step_size *= math.sqrt(in_dim / k)

                denom_lr = exp_avg_sq_lr.sqrt().add_(eps)
                delta_h = exp_avg_lr / denom_lr
                delta_h.mul_(-step_size)  # (out_dim, k) in param.dtype

                delta_w = delta_h.to(dtype=torch.float16) @ P.T  # (out_dim, in_dim) fp16

                # update GPU fp8 param via fp16 round-trip + rescale
                w_mat = param.data.view(out_dim, in_dim)
                w_fp16 = fp8_to_fp16_with_scale(w_mat, param.fp8_scale)

                if weight_decay != 0.0:
                    w_fp16.mul_(1.0 - lr * weight_decay)

                w_fp16.add_(delta_w)
                res, param.fp8_scale = fp16_to_fp8_with_rescaling(w_fp16)
                w_mat.copy_(res)

                # async update CPU master, and gate refresh by cpu_ready_evt
                if hasattr(param, "cpu_fp16"):
                    cpu_ready_evt: threading.Event = st["cpu_ready_evt"]
                    cpu_ready_evt.clear()

                    producer_stream = torch.cuda.current_stream(param.device)
                    producer_event = torch.cuda.Event()
                    producer_event.record(producer_stream)

                    self.worker.send(
                        AdamLoraPre._finish_in_subspace_update_and_signal,
                        producer_event,
                        param.cpu_fp16,
                        delta_w,
                        out_dim,
                        in_dim,
                        lr,
                        weight_decay,
                        cpu_ready_evt,
                        key=param,
                    )

                param.grad_lr = None
                proj.bump_seed()

        if self._refresh_k > 0:
            self._refresh_from_cpu()

        return loss

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if p is None:
                    continue

                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        p.grad.zero_()

                if getattr(p, "requires_grad_lr", False) and hasattr(p, "grad_lr") and p.grad_lr is not None:
                    if set_to_none:
                        p.grad_lr = None
                    else:
                        if p.grad_lr.grad_fn is not None:
                            p.grad_lr.detach_()
                        p.grad_lr.zero_()


def safe_cholesky(a, base_eps, max_tries=6):
    a = (a + a.transpose(-1, -2)) * 0.5
    eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
    eps = base_eps
    for _ in range(max_tries):
        l, info = torch.linalg.cholesky_ex(a + eps * eye)
        if int(info.max()) == 0:
            return l, eps
        eps *= 10.0

    v, vs = torch.linalg.eigh(a)
    evals = v.clamp_min(base_eps)
    l = (vs * evals.sqrt()) @ vs.transpose(-1, -2)
    return l, eps


def fp8_to_fp16_with_scale(
    w_fp8: torch.Tensor,
    scale: torch.Tensor,
    out_device: torch.device | str = None,
    *,
    axis_muliscale: bool = True,
    scale_dim: int = 0,
) -> torch.Tensor:
    w_fp16 = w_fp8.to(device=out_device, dtype=torch.float16)

    if axis_muliscale:
        broadcast_shape = [1] * w_fp16.ndim
        broadcast_shape[scale_dim] = -1
        return w_fp16 * scale.view(broadcast_shape)

    return w_fp16 * scale


def fp16_to_fp8_with_rescaling(
    w_fp16: torch.Tensor,
    out_device: torch.device | str = None,
    *,
    axis_muliscale: bool = True,
    scale_dim: int = 0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fnuz).max

    if axis_muliscale:
        reduce_dims = [d for d in range(w_fp16.ndim) if d != scale_dim % w_fp16.ndim]
        amax = w_fp16.abs().amax(dim=reduce_dims)
    else:
        amax = w_fp16.abs().max()

    scale = (amax / fp8_max).clamp(min=eps)

    if axis_muliscale:
        broadcast_shape = [1] * w_fp16.ndim
        broadcast_shape[scale_dim] = -1
        w_scaled = w_fp16 / scale.view(broadcast_shape)
    else:
        w_scaled = w_fp16 / scale

    w_scaled.clamp_(-fp8_max, fp8_max)
    w_fp8 = w_scaled.to(device=out_device, dtype=torch.float8_e4m3fnuz)
    return w_fp8, scale
