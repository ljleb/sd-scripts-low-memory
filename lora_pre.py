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

        output = torch.nn.functional.linear(input_tensor, weight.detach().to(weight.cpu_fp16.dtype), bias)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):
        grad_output, *_ = grad_outputs
        sketch, weight = ctx.saved_tensors
        has_bias = ctx.has_bias

        grad_input = grad_output @ weight.detach().to(weight.cpu_fp16.dtype)

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
        output = torch.nn.functional.conv2d(
            input_tensor,
            weight.detach().to(weight.cpu_fp16.dtype),
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

        grad_input = torch.nn.functional.conv_transpose2d(
            grad_output,
            weight.detach().to(weight.cpu_fp16.dtype),
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
    ):
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        beta1, beta2 = betas
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta1")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta2")

        gamma1 = math.sqrt(beta1)
        gamma2 = beta2 ** 0.25
        if not 0.0 <= gamma1 < 1.0:
            raise ValueError("Invalid gamma1")
        if not 0.0 <= gamma2 < 1.0:
            raise ValueError("Invalid gamma2")

        self.worker = WorkerPool("cuda:0", num_workers=8, maxsize=0)

        defaults = dict(
            lr=lr,
            rank_ratio=rank_ratio,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma1=gamma1,
            gamma2=gamma2,
            cholesky_eps=cholesky_eps,
            max_grad_norm=max_grad_norm,
            match_energy=match_energy,
        )
        super().__init__(params, defaults)

        # Initialize state
        for group in self.param_groups:
            group_rank_ratio = group["rank_ratio"]

            for p in group["params"]:
                if p is None:
                    continue

                state = self.state[p]
                state["step"] = 0

                lr_flag = getattr(p, "requires_grad_lr", False)

                if lr_flag and p.dim() >= 2:
                    out_dim = p.shape[0]
                    in_dim = p.shape[1:].numel()
                    min_dim = min(out_dim, in_dim)

                    r = getattr(p, "lr_rank", None)
                    if r is None:
                        r = max(int(min_dim * group_rank_ratio), 1)
                        r = min(r, min_dim)

                    state["mode"] = 1
                    state["shape_2d"] = (out_dim, in_dim)
                    state["rank"] = r

                    device = p.device
                    dtype = p.dtype

                    state["m_B"] = torch.zeros(out_dim, r, device=device, dtype=dtype)
                    state["m_A"] = torch.randn(r, in_dim, device=device, dtype=dtype) * 0.02

                    state["v_B"] = torch.zeros(out_dim, r, device=device, dtype=dtype)
                    state["v_A"] = torch.randn(r, in_dim, device=device, dtype=dtype) * 0.02
                else:
                    state["mode"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            gamma1 = group["gamma1"]
            gamma2 = group["gamma2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            cholesky_eps = group["cholesky_eps"]
            max_grad_norm = group["max_grad_norm"]
            match_energy = group["match_energy"]

            for p in group["params"]:
                if p is None:
                    continue

                state = self.state[p]
                is_vector = state["mode"] == 0

                lr_flag = getattr(p, "requires_grad_lr", False)

                if not lr_flag:
                    grad = p.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("AdamLoraPre does not support sparse gradients")
                else:
                    grad = None

                state["step"] += 1
                t = state["step"]

                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                if is_vector:
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    grad_fp = grad

                    if max_grad_norm is not None:
                        g_norm = grad_fp.norm()
                        if g_norm > max_grad_norm:
                            grad_fp = grad_fp * (max_grad_norm / (g_norm + 1e-6))

                    exp_avg.mul_(beta1).add_(grad_fp, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad_fp, grad_fp, value=1.0 - beta2)

                    bias_correction1 = 1.0 - beta1**t
                    bias_correction2 = 1.0 - beta2**t
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                    denom = exp_avg_sq.sqrt().add_(eps)

                    # producer_stream = torch.cuda.current_stream(p.device)
                    # producer_event = torch.cuda.Event()
                    # producer_event.record(producer_stream)
                    #
                    # self.worker.send(
                    #     finish_adam,
                    #     producer_event, p.cpu_fp16,
                    #     exp_avg, denom,
                    #     step_size,
                    # )

                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    continue

                out_dim, in_dim = state["shape_2d"]
                m_b_prev = state["m_B"]
                m_a_prev = state["m_A"]
                v_b_prev = state["v_B"]
                v_a_prev = state["v_A"]

                if lr_flag:
                    g_sk = getattr(p, "grad_lr", None)
                    proj: SketchMatrixGenerator = getattr(p, "lr_proj", None)
                    if g_sk is None:
                        if proj is not None:
                            proj.bump_seed()
                        continue

                    p_mat = proj.materialize_p(device=p.device, dtype=torch.float16)

                    gram = p_mat.T.float() @ p_mat.float()
                    gram.diagonal().add_(cholesky_eps)
                    l_p = safe_cholesky(gram, cholesky_eps)[0]

                    h = torch.cholesky_solve(g_sk.T.float(), l_p, upper=False).T.to(p_mat.dtype)

                    g2d = h @ p_mat.T

                    if "m_B_pinned" not in state:
                        state["m_B_pinned"] = torch.empty_like(m_b_prev, device=p.cpu_fp16.device, pin_memory=True)
                    if "m_A_pinned" not in state:
                        state["m_A_pinned"] = torch.empty_like(m_a_prev, device=p.cpu_fp16.device, pin_memory=True)
                    if "v_B_pinned" not in state:
                        state["v_B_pinned"] = torch.empty_like(v_b_prev, device=p.cpu_fp16.device, pin_memory=True)
                    if "v_A_pinned" not in state:
                        state["v_A_pinned"] = torch.empty_like(v_a_prev, device=p.cpu_fp16.device, pin_memory=True)
                    if "h_pinned" not in state:
                        state["h_pinned"] = torch.empty_like(h, device=p.cpu_fp16.device, pin_memory=True)
                else:
                    g2d = grad.view(out_dim, in_dim)

                if max_grad_norm is not None:
                    g_norm = g2d.norm()
                    if g_norm > max_grad_norm:
                        g2d = g2d * (max_grad_norm / (g_norm + 1e-6))

                m_a_cov = m_a_prev.float() @ m_a_prev.T.float()
                l_a = safe_cholesky(m_a_cov, cholesky_eps)[0]

                rhs_b_t = g2d @ m_a_prev.T
                b_star_t = torch.cholesky_solve(rhs_b_t.T.float(), l_a, upper=False).to(m_a_prev.dtype)
                b_star = b_star_t.T

                m_b_cov = m_b_prev.T.float() @ m_b_prev.float()
                l_b = safe_cholesky(m_b_cov, cholesky_eps)[0]

                rhs_a = m_b_prev.T @ g2d
                a_star = torch.cholesky_solve(rhs_a.float(), l_b, upper=False).to(m_b_prev.dtype)

                m_b = gamma1 * m_b_prev + (1.0 - gamma1) * b_star
                m_a = gamma1 * m_a_prev + (1.0 - gamma1) * a_star

                state["m_B"] = m_b
                state["m_A"] = m_a

                g_abs = g2d.abs()

                v_a_cov = v_a_prev.float() @ v_a_prev.T.float()
                l_av = safe_cholesky(v_a_cov, cholesky_eps)[0]

                rhs_bv_t = g_abs @ v_a_prev.T
                bv_star_t = torch.cholesky_solve(rhs_bv_t.T.float(), l_av, upper=False).to(v_a_prev.dtype)
                bv_star = bv_star_t.T

                v_b_cov = v_b_prev.T.float() @ v_b_prev.float()
                l_bv = safe_cholesky(v_b_cov, cholesky_eps)[0]

                rhs_av = v_b_prev.T @ g_abs
                av_star = torch.cholesky_solve(rhs_av.float(), l_bv, upper=False).to(v_b_prev.dtype)

                v_b = gamma2 * v_b_prev + (1.0 - gamma2) * bv_star
                v_a = gamma2 * v_a_prev + (1.0 - gamma2) * av_star

                state["v_B"] = v_b
                state["v_A"] = v_a

                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                if lr_flag and match_energy:
                    # energy correction
                    k = p.lr_sketch_rank
                    step_size *= math.sqrt(in_dim / k)

                # if lr_flag and hasattr(p, "cpu_fp16"):
                #     producer_stream = torch.cuda.current_stream(p.device)
                #     producer_event = torch.cuda.Event()
                #     producer_event.record(producer_stream)
                #
                #     self.worker.send(
                #         finish_lora_pre,
                #         producer_event, p.cpu_fp16,
                #         m_a_prev, m_b_prev,
                #         v_a_prev, v_b_prev,
                #         h, p.lr_proj.copy(),
                #         state["m_A_pinned"], state["m_B_pinned"],
                #         state["v_A_pinned"], state["v_B_pinned"],
                #         state["h_pinned"],
                #         out_dim, in_dim,
                #         beta1, beta2,
                #         eps, step_size,
                #         key=p,
                #     )

                w_mat = p.data.view(out_dim, in_dim)

                m_prev_full = m_b_prev @ m_a_prev
                v_prev_full = v_b_prev @ v_a_prev

                m_t = beta1 * m_prev_full + (1.0 - beta1) * g2d
                v_t = beta2 * v_prev_full.square() + (1.0 - beta2) * g2d.square()

                denom = v_t.sqrt_().add_(eps)

                if lr_flag and hasattr(p, "cpu_fp16"):
                    producer_stream = torch.cuda.current_stream(p.device)
                    producer_event = torch.cuda.Event()
                    producer_event.record(producer_stream)

                    self.worker.send(
                        finish_lora_pre_v2,
                        producer_event, p.cpu_fp16,
                        m_t, denom,
                        out_dim, in_dim,
                        step_size,
                    )

                w_mat.copy_(w_mat.float().addcdiv(m_t, denom, value=-step_size).to(w_mat))
                if lr_flag and hasattr(p, "lr_proj"):
                    p.lr_proj.bump_seed()

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


def finish_lora_pre_v2(
    producer_event, p_cpu,
    m_t, denom,
    out_dim, in_dim,
    step_size,
):
    cpu_device = p_cpu.device

    consumer_stream = torch.cuda.current_stream()
    consumer_stream.wait_event(producer_event)

    m_t = m_t.to(cpu_device)
    denom = denom.to(cpu_device)

    done = torch.cuda.Event()
    done.record(consumer_stream)
    done.synchronize()

    w_mat = p_cpu.data.view(out_dim, in_dim)

    w_mat.addcdiv_(m_t, denom, value=-step_size)


def finish_lora_pre(
    producer_event, p_cpu,
    m_a_prev, m_b_prev,
    v_a_prev, v_b_prev,
    h, lr_proj,
    m_a_pinned, m_b_pinned,
    v_a_pinned, v_b_pinned,
    h_pinned,
    out_dim, in_dim,
    beta1, beta2,
    eps, step_size,
):
    cpu_device = p_cpu.device

    consumer_stream = torch.cuda.current_stream()
    consumer_stream.wait_event(producer_event)

    m_a_prev = m_a_pinned.copy_(m_a_prev, non_blocking=True)
    m_b_prev = m_b_pinned.copy_(m_b_prev, non_blocking=True)
    v_a_prev = v_a_pinned.copy_(v_a_prev, non_blocking=True)
    v_b_prev = v_b_pinned.copy_(v_b_prev, non_blocking=True)
    h = h_pinned.copy_(h, non_blocking=True)

    done = torch.cuda.Event()
    done.record(consumer_stream)
    done.synchronize()

    g2d = h @ lr_proj.materialize_p(cpu_device, torch.float16).T
    w_mat = p_cpu.data.view(out_dim, in_dim)

    m_prev_full = m_b_prev @ m_a_prev
    v_prev_full = v_b_prev @ v_a_prev

    m_t = beta1 * m_prev_full + (1.0 - beta1) * g2d
    v_t = beta2 * v_prev_full.square() + (1.0 - beta2) * g2d.square()

    denom = v_t.sqrt_().add_(eps)
    w_mat.addcdiv_(m_t, denom, value=-step_size)


def finish_adam(
    producer_event, p_cpu,
    exp_avg, denom,
    step_size,
):
    consumer_stream = torch.cuda.current_stream()
    consumer_stream.wait_event(producer_event)

    done = torch.cuda.Event()
    done.record(consumer_stream)
    done.synchronize()

    w_mat = p_cpu.data

    w_mat.addcdiv_(exp_avg.to(w_mat), denom.to(w_mat), value=-step_size)


class Fp8RefreshScheduler:
    """
    Refresh ~N/K parameters per optimizer step from CPU fp16 masters onto GPU fp8 params,
    without replacement across each K-step cycle.

    Assumes: p.cpu_fp16 is already pinned CPU memory (so non_blocking H2D is meaningful).
    """

    def __init__(self, params, K: int, seed: int = 0, stream=None):
        assert K >= 1
        self.params = self._dedup([
            p for p in params
            if hasattr(p, "cpu_fp16")
        ])
        self.N = len(self.params)
        self.K = K

        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)

        self._cycle_step = 0
        self._perm = []

        self.stream = stream if stream is not None else torch.cuda.Stream()
        self._done = torch.cuda.Event()

        self._new_cycle()

    def _dedup(self, ps):
        seen = set()
        out = []
        for p in ps:
            i = id(p)
            if i in seen:
                continue
            seen.add(i)
            out.append(p)
        return out

    def _new_cycle(self):
        self._cycle_step = 0
        if self.N == 0:
            self._perm = []
        else:
            self._perm = torch.randperm(self.N, generator=self._gen).tolist()

    def step(self):
        if self.N == 0:
            return

        if self._cycle_step >= self.K:
            self._new_cycle()

        s = self._cycle_step
        self._cycle_step += 1

        start = (s * self.N) // self.K
        end = ((s + 1) * self.N) // self.K

        if start >= end:
            # empty slice this step (common when K > N)
            with torch.cuda.stream(self.stream):
                pass
            self._done.record(self.stream)
            return

        idxs = self._perm[start:end]

        with torch.cuda.stream(self.stream):
            for i in idxs:
                p = self.params[i]      # fp8 param on GPU
                src = p.cpu_fp16        # pinned fp16 master on CPU
                p.data.copy_(
                    src.to(device=p.device, dtype=p.dtype, non_blocking=True),
                    non_blocking=True,
                )

        self._done.record(self.stream)
        torch.cuda.current_stream().wait_event(self._done)
