"""Train a SequenceOfDenseBlocks on CIFAR10 with either plain SGD or Newton
steps that apply the Hessian inverse to the gradient.

Run plain SGD:
    python src/train_newton.py --mode sgd --num-steps 200

Run Newton:
    python src/train_newton.py --mode newton --num-steps 200

Each run writes a tensorboard event file to --logdir/<mode>-<timestamp>/.
Launch tensorboard against the parent directory to overlay runs:
    tensorboard --logdir runs/
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import dataclasses
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import block_partitioned_matrices as bpm
import hessian
from hessian import SequenceOfDenseBlocks


CIFAR10_NUM_CLASSES = 10
CIFAR10_CHANNELS = 3

ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
}


@dataclasses.dataclass
class StepScalars:
    """Per-step metrics written to tensorboard.

    The training loop fills in `loss`, `batch_accuracy`, `grad_norm`,
    `step_norm`. `StepLogger.step()` stamps `step_seconds` and `wall_clock_s`.
    """

    loss: float = 0.0
    train_loss_avg10: float = 0.0
    probe_loss: float = 0.0
    probe_accuracy: float = 0.0
    batch_accuracy: float = 0.0
    grad_norm: float = 0.0
    step_norm: float = 0.0
    epsilon: float = 0.0
    lr: float = 0.0
    accepted: float = 1.0
    step_seconds: float = 0.0
    wall_clock_s: float = 0.0
    cos_step_neg_grad: float = 0.0
    pred_loss_change: float = 0.0
    actual_loss_change: float = 0.0
    h_eig_min: float = 0.0
    h_eig_max: float = 0.0
    # Trust-region fields
    trust_radius: float = 0.0
    rho: float = 0.0
    lambda_star: float = 0.0
    hard_case: float = 0.0
    step_type: float = 0.0  # 0 = interior, 1 = boundary
    tr_solves: float = 0.0  # oracle (H+lambda I) solves used by the efficient solver


class StepLogger:
    """Per-step tensorboard logger.

    Times each step end-to-end. The training loop writes fields onto the
    `StepScalars` yielded by `step()`; on exit, the logger stamps
    `step_seconds` and `wall_clock_s` and forwards every field to a
    `SummaryWriter`. Scalars are written under their plain names (e.g.
    "loss", "grad_norm") so multiple runs in the same parent logdir overlay
    automatically.
    """

    def __init__(self, logdir: Path, device: torch.device, tb_log_every: int = 1):
        self.writer = SummaryWriter(log_dir=str(logdir))
        self.device = device
        self.t0: float | None = None
        # Write scalars to tensorboard only every `tb_log_every` steps. Keeps
        # event files small on very long runs (e.g. hundreds of thousands of
        # SGD steps) where per-step resolution is unnecessary for plotting.
        self.tb_log_every = tb_log_every

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def start(self) -> None:
        self._sync()
        self.t0 = time.perf_counter()

    @contextlib.contextmanager
    def step(self, step_idx: int):
        if self.t0 is None:
            raise RuntimeError("StepLogger.start() must be called before step()")
        scalars = StepScalars()
        self._sync()
        t_step_start = time.perf_counter()
        yield scalars
        self._sync()
        t_now = time.perf_counter()
        scalars.step_seconds = t_now - t_step_start
        scalars.wall_clock_s = t_now - self.t0
        if step_idx % self.tb_log_every == 0:
            for name, value in dataclasses.asdict(scalars).items():
                self.writer.add_scalar(name, value, global_step=step_idx)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


# The canonical URL https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz now
# serves a redirect chain that bounces https -> http -> https across hosts.
# torchvision's urllib-based downloader refuses to follow the https->http
# downgrade, so download_and_extract_archive() silently stalls and no run ever
# reaches a training step. Point CIFAR10 straight at the final URL, which
# returns a clean 200, so urllib never sees the broken redirect.
torchvision.datasets.CIFAR10.url = (
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz"
)


def load_cifar10_loader(
    data_dir: Path, batch_size: int, seed: int, image_size: int
) -> torch.utils.data.DataLoader:
    """CIFAR10 train loader. Images are resized to (image_size, image_size) and flattened."""

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: t.flatten()),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=True,
    )


def assemble_gradient_vector(model: SequenceOfDenseBlocks) -> bpm.Vertical:
    """Pack each layer's parameter gradients into a per-layer column block.

    Iteration order matches `model.__iter__` (hidden layers then loss_layer),
    which is the same ordering used by `hessian_inverse_product`.
    """
    blocks = []
    for layer in model:
        flat = torch.cat([p.grad.detach().flatten() for p in layer.parameters()])
        blocks.append(flat.unsqueeze(1))
    return bpm.Vertical(blocks)


def dense_newton_step(
    model: SequenceOfDenseBlocks,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_vec: bpm.Vertical,
    epsilon: float,
    method: str,
) -> tuple[bpm.Vertical, dict[str, float]]:
    """Compute the Newton step by materializing the full Hessian and solving / pinv-ing it."""
    def loss_fn(params):
        return torch.func.functional_call(model, params, (x, y))

    hessian_dict = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
    H = hessian.flatten_2d_pytree(hessian_dict)
    P = H.shape[0]
    g_flat = grad_vec.to_tensor().flatten()
    A = H + epsilon * torch.eye(P, dtype=H.dtype, device=H.device)
    if method == "dense-solve":
        x_flat = torch.linalg.solve(A, g_flat)
    elif method == "dense-pinv":
        x_flat = torch.linalg.pinv(A) @ g_flat
    else:
        raise ValueError(f"unknown dense newton method: {method}")

    with torch.no_grad():
        eigvals = torch.linalg.eigvalsh(H)
        diagnostics = {
            "cos_step_neg_grad": float(
                (g_flat @ x_flat) / (g_flat.norm() * x_flat.norm() + 1e-30)
            ),
            "pred_linear": float(g_flat @ x_flat),
            "pred_quadratic": float(x_flat @ H @ x_flat),
            "h_eig_min": float(eigvals.min()),
            "h_eig_max": float(eigvals.max()),
        }

    blocks = []
    offset = 0
    for block in grad_vec.flat:
        n = block.shape[0]
        blocks.append(x_flat[offset : offset + n].unsqueeze(1))
        offset += n
    return bpm.Vertical(blocks), diagnostics


def apply_update(model: SequenceOfDenseBlocks, update: bpm.Vertical, lr: float) -> None:
    """In-place: p <- p - lr * update_block, distributing each block across the layer's params."""
    for layer, block in zip(model, update.flat):
        offset = 0
        flat = block.flatten()
        for p in layer.parameters():
            n = p.numel()
            p.data.add_(flat[offset : offset + n].view_as(p), alpha=-lr)
            offset += n
        assert offset == flat.numel()


def vertical_norm(v: bpm.Vertical) -> float:
    return float(torch.linalg.vector_norm(v.to_tensor()).item())


def solve_trs(
    g: torch.Tensor, H: torch.Tensor, delta: float
) -> tuple[torch.Tensor, float, str, bool, torch.Tensor]:
    """Solve the trust-region subproblem exactly via eigendecomposition.

    Returns (p, lambda_star, step_type, hard_case, eigvals) where step_type is
    "interior" or "boundary" and hard_case is True when g is nearly
    orthogonal to the minimum eigenvector (logged but not specially handled).
    eigvals is the ascending spectrum of H, returned so the caller can read
    h_eig_min/max without a second (expensive) eigendecomposition.
    """
    eigvals, Q = torch.linalg.eigh(H)
    lambda_min = float(eigvals[0].item())

    # Project gradient into eigenbasis: g_hat = Qᵀ g
    g_hat = Q.T @ g

    # Try unconstrained Newton step (lambda=0), valid only if H is PD
    if lambda_min > 0:
        p_hat_newton = -g_hat / eigvals
        if float(p_hat_newton.norm().item()) <= delta:
            p = Q @ p_hat_newton
            return p, 0.0, "interior", False, eigvals

    # Need lambda > 0. Secular equation: ‖(H + λI)⁻¹g‖ = delta
    # In eigenbasis this is sum(g_hat_i² / (λ_i + λ)²) = delta²
    # Bisect over lambda in [lambda_lb, lambda_ub]
    lambda_lb = max(0.0, -lambda_min)

    hard_case = lambda_min <= 0 and float(g_hat[0].abs().item()) < 1e-10 * float(g_hat.norm().item())

    def secular(lam: float) -> float:
        denom = eigvals + lam
        return float((g_hat / denom).norm().item())

    # Find upper bound where secular(lambda) < delta
    lambda_ub = lambda_lb + 1.0
    for _ in range(60):
        if secular(lambda_ub) <= delta:
            break
        lambda_ub *= 2.0

    # Bisect
    for _ in range(50):
        lam_mid = (lambda_lb + lambda_ub) / 2.0
        if secular(lam_mid) > delta:
            lambda_lb = lam_mid
        else:
            lambda_ub = lam_mid
        if lambda_ub - lambda_lb < 1e-10 * (1.0 + lambda_ub):
            break

    lambda_star = (lambda_lb + lambda_ub) / 2.0
    p_hat = -g_hat / (eigvals + lambda_star)
    p = Q @ p_hat
    return p, lambda_star, "boundary", hard_case, eigvals


def vertical_like(flat: torch.Tensor, template: bpm.Vertical) -> bpm.Vertical:
    """Repack a flat vector into the per-layer column-block structure of `template`."""
    blocks = []
    offset = 0
    for block in template.flat:
        n = block.shape[0]
        blocks.append(flat[offset : offset + n].unsqueeze(1))
        offset += n
    return bpm.Vertical(blocks)


def solve_trs_oracle(
    g: torch.Tensor,
    apply_inverse: Callable[[float, torch.Tensor], torch.Tensor],
    delta: float,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> tuple[torch.Tensor, float, str, bool, int]:
    """Solve the trust-region subproblem using only damped-inverse solves.

    `apply_inverse(lam, rhs)` must return `(H + lam·I)⁻¹ rhs` for flat vectors.
    This is the linear-time oracle the paper provides; this routine never forms
    or eigendecomposes H. It mirrors the math of `solve_trs` but reaches the
    same minimizer `p = -(H + λ*I)⁻¹ g` through a 1-D search on the damping λ.

    Returns (p, lambda_star, step_type, hard_case, n_solves). `p` is the step
    (params should move to params + p), matching `solve_trs`'s convention.

    Caveat — the hard case: without λ_min we use the curvature gᵀH⁻¹g as a
    positive-definiteness proxy and search λ from 0. This is exact when H ⪰ 0;
    when H is indefinite it is flagged via `hard_case` (logged, not specially
    handled), per the project's "just log the hard case for now" decision.
    """
    n_solves = 0

    def inv(lam: float, rhs: torch.Tensor) -> torch.Tensor:
        nonlocal n_solves
        n_solves += 1
        return apply_inverse(lam, rhs)

    # Interior step (λ = 0): p = -H⁻¹g, valid only if it lies inside the region
    # AND H is positive definite. gᵀH⁻¹g is the PD proxy — positive when the
    # Newton step descends into a bowl, non-positive when g has negative-
    # curvature content (indefinite H), in which case we go to the boundary.
    s = inv(0.0, g)
    phi = float(torch.linalg.vector_norm(s))
    curvature = float(g @ s)
    if phi <= delta and curvature > 0:
        return -s, 0.0, "interior", False, n_solves

    hard_case = curvature <= 0

    # Boundary: find λ > 0 with ‖(H + λI)⁻¹g‖ = delta. φ(λ) decreases to 0 as λ
    # grows, so doubling brackets a feasible hi.
    lo, hi = 0.0, 1.0
    s = inv(hi, g)
    while float(torch.linalg.vector_norm(s)) > delta and hi < 1e16:
        hi *= 2.0
        s = inv(hi, g)
    lam = hi

    # Safeguarded Newton on the secular equation 1/φ(λ) = 1/delta. The
    # reciprocal 1/φ is nearly affine in λ, so this converges in a few steps.
    # φ'(λ) = -(sᵀq)/φ with q = (H+λI)⁻¹ s costs one extra solve per iteration.
    for _ in range(max_iter):
        phi = float(torch.linalg.vector_norm(s))
        if abs(phi - delta) <= tol * delta:
            break
        if phi > delta:
            lo = max(lo, lam)
        else:
            hi = min(hi, lam)
        q = inv(lam, s)
        phi_prime = -float(s @ q) / phi
        h_prime = -phi_prime / (phi * phi)
        if h_prime > 0:
            lam_next = lam + (1.0 / delta - 1.0 / phi) / h_prime
        else:
            lam_next = 0.5 * (lo + hi)
        if not (lo < lam_next < hi):
            lam_next = 0.5 * (lo + hi)
        lam = lam_next
        s = inv(lam, g)

    return -s, lam, "boundary", hard_case, n_solves


def efficient_solve_trs(
    model: SequenceOfDenseBlocks,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_vec: bpm.Vertical,
    delta: float,
    tol: float = 1e-6,
    max_iter: int = 20,
) -> tuple[torch.Tensor, float, str, bool, int]:
    """Trust-region subproblem via the paper's linear-time damped solver.

    Computes the network derivatives once (`hessian_inverse_setup`) and reuses
    them for every trial damping λ, so the per-λ cost is just the block-
    tridiagonal factorization rather than a fresh functorch pass.
    """
    setup = model.hessian_inverse_setup(x, y)
    g = grad_vec.to_tensor().flatten()

    def apply_inverse(lam: float, rhs_flat: torch.Tensor) -> torch.Tensor:
        rhs_vert = vertical_like(rhs_flat, grad_vec)
        out = model.hessian_inverse_solve(setup, rhs_vert, lam)
        # The solve is a fixed linear algebra step; we never backprop through it.
        return out.to_tensor().flatten().detach()

    return solve_trs_oracle(g, apply_inverse, delta, tol=tol, max_iter=max_iter)


def sgd_warmup(
    model: "SequenceOfDenseBlocks",
    loader: torch.utils.data.DataLoader,
    lr: float,
    num_steps: int,
    device: torch.device,
) -> float:
    """Run num_steps of SGD and return the mean step norm, for Δ initialization."""
    step_norms = []
    data_iter = iter(loader)
    for _ in range(num_steps):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = model(x, y)
        loss.backward()
        grad_vec = assemble_gradient_vector(model)
        step_norms.append(lr * vertical_norm(grad_vec))
        apply_update(model, grad_vec, lr)
    # Undo the warmup steps so model state is unaffected
    # (We just need the scale, not the actual optimization.)
    # Actually it's fine to leave it — a few SGD steps are a reasonable init.
    return float(sum(step_norms) / len(step_norms))


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device: {device}", file=sys.stderr)

    loader = load_cifar10_loader(
        Path(args.data_dir), args.batch_size, args.seed, args.image_size
    )
    # Fixed probe batch for a clean per-step loss reading that doesn't move
    # with the data sampler. Seed is hard-coded so the same probe is used
    # across every run, making the `probe_loss` scalar directly comparable.
    probe_loader = load_cifar10_loader(
        Path(args.data_dir), args.probe_batch_size, seed=999_999, image_size=args.image_size
    )
    probe_x, probe_y = next(iter(probe_loader))
    probe_x = probe_x.to(device)
    probe_y = probe_y.to(device)
    input_dim = CIFAR10_CHANNELS * args.image_size * args.image_size

    model = SequenceOfDenseBlocks(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=CIFAR10_NUM_CLASSES,
        num_layers=args.num_layers,
        activation=ACTIVATIONS[args.activation],
    ).to(device)

    # Default nn.Linear init under-scales weights for deep nets. Re-initialize
    # so gradients propagate through all layers at depth.
    nonlinearity = "relu" if args.activation in {"relu", "gelu"} else "tanh"
    for layer in model:
        torch.nn.init.kaiming_normal_(layer.linear.weight, nonlinearity=nonlinearity)

    run_name = (
        args.run_name or f"{args.mode}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    logdir = Path(args.logdir) / run_name
    print(f"tensorboard logdir: {logdir}", file=sys.stderr)

    logger = StepLogger(logdir, device=device, tb_log_every=args.tb_log_every)
    logger.writer.add_text(
        "config", "  \n".join(f"{k}: {v}" for k, v in vars(args).items())
    )
    logger.start()

    step_idx = 0
    data_iter = iter(loader)
    epsilon = args.epsilon
    lr = args.lr
    loss_window = collections.deque(maxlen=2 * args.lr_decay_window)
    x, y = None, None

    # Trust-region state
    if args.mode == "trust-region":
        if args.delta_init is not None:
            trust_radius = args.delta_init
            print(f"trust radius init: {trust_radius:.4e} (manual)", file=sys.stderr)
        else:
            print(f"running {args.delta_init_steps}-step SGD warmup to calibrate Δ ...", file=sys.stderr)
            trust_radius = sgd_warmup(model, loader, lr, args.delta_init_steps, device)
            print(f"trust radius init: {trust_radius:.4e} (from SGD warmup)", file=sys.stderr)
    else:
        trust_radius = 0.0
    while step_idx < args.num_steps:
        if step_idx % args.reuse_batch == 0:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x = x.to(device)
            y = y.to(device)

        with logger.step(step_idx) as scalars:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            loss = model(x, y)
            scalars.loss = float(loss.item())
            loss_window.append(scalars.loss)
            window = args.lr_decay_window
            scalars.train_loss_avg10 = sum(loss_window) / len(loss_window)
            scalars.epsilon = epsilon
            scalars.lr = lr
            with torch.no_grad():
                scalars.probe_loss = float(model(probe_x, probe_y).item())
                probe_features = model.layers(probe_x)
                probe_logits = model.loss_layer.linear(probe_features)
                scalars.probe_accuracy = float(
                    (probe_logits.argmax(dim=1) == probe_y).float().mean().item()
                )

            # Batch accuracy uses the loss layer's pre-loss logits. Re-run the
            # cheap forward up to the loss layer to read them.
            with torch.no_grad():
                features = model.layers(x)
                logits = model.loss_layer.linear(features)
                scalars.batch_accuracy = float(
                    (logits.argmax(dim=1) == y).float().mean().item()
                )

            loss.backward()
            grad_vec = assemble_gradient_vector(model)
            scalars.grad_norm = vertical_norm(grad_vec)

            if args.mode == "sgd":
                update = grad_vec
                scalars.step_norm = lr * vertical_norm(update)
                apply_update(model, update, lr)
            elif args.mode == "newton":
                # Levenberg-Marquardt: try the step at the current damping. If
                # loss decreased on the same batch, accept and decrease ε. Else
                # undo the Newton step, fall back to a small SGD step so the
                # iteration still makes progress along a descent direction,
                # and increase ε so the next Newton attempt is better damped.
                if args.newton_step_method == "custom":
                    update = model.hessian_inverse_product(x, y, grad_vec, epsilon)
                else:
                    update, diagnostics = dense_newton_step(
                        model, x, y, grad_vec, epsilon, args.newton_step_method
                    )
                    scalars.cos_step_neg_grad = diagnostics["cos_step_neg_grad"]
                    scalars.h_eig_min = diagnostics["h_eig_min"]
                    scalars.h_eig_max = diagnostics["h_eig_max"]
                raw_step_norm = lr * vertical_norm(update)
                if args.max_step_norm is not None and raw_step_norm > args.max_step_norm:
                    effective_lr = lr * args.max_step_norm / raw_step_norm
                else:
                    effective_lr = lr
                scalars.step_norm = effective_lr * vertical_norm(update)
                apply_update(model, update, effective_lr)
                if args.lm_check_batch == "fresh":
                    try:
                        check_x, check_y = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        check_x, check_y = next(data_iter)
                    check_x = check_x.to(device)
                    check_y = check_y.to(device)
                    with torch.no_grad():
                        trial_loss = float(model(check_x, check_y).item())
                        apply_update(model, update, -effective_lr)
                        baseline_check_loss = float(model(check_x, check_y).item())
                        apply_update(model, update, effective_lr)
                    accept = trial_loss < baseline_check_loss
                else:
                    with torch.no_grad():
                        trial_loss = float(model(x, y).item())
                    accept = trial_loss < scalars.loss
                if args.newton_step_method != "custom" and args.lm_check_batch == "same":
                    scalars.actual_loss_change = trial_loss - scalars.loss
                    scalars.pred_loss_change = (
                        -effective_lr * diagnostics["pred_linear"]
                        + 0.5 * effective_lr ** 2 * diagnostics["pred_quadratic"]
                    )
                if accept:
                    epsilon = max(epsilon * args.lm_down, args.lm_eps_min)
                    lr = max(lr * args.lr_lm_on_accept, args.lr_min)
                    scalars.accepted = 1.0
                else:
                    apply_update(model, update, -effective_lr)
                    apply_update(model, grad_vec, args.sgd_fallback_lr)
                    scalars.step_norm = args.sgd_fallback_lr * vertical_norm(grad_vec)
                    epsilon = min(epsilon * args.lm_up, args.lm_eps_max)
                    lr = max(lr * args.lr_lm_on_reject, args.lr_min)
                    scalars.accepted = 0.0
            elif args.mode == "trust-region":
                g_flat = grad_vec.to_tensor().flatten()

                if args.tr_solver == "dense":
                    # Build the full dense Hessian (same path as dense-solve Newton)
                    def loss_fn(params):
                        return torch.func.functional_call(model, params, (x, y))

                    hessian_dict = torch.func.hessian(loss_fn)(
                        dict(model.named_parameters())
                    )
                    H = hessian.flatten_2d_pytree(hessian_dict)

                    p_flat, lambda_star, step_type, hard_case, eigvals = solve_trs(
                        g_flat, H, trust_radius
                    )
                    scalars.h_eig_min = float(eigvals[0].item())
                    scalars.h_eig_max = float(eigvals[-1].item())

                    # Predicted reduction from the quadratic model: -gᵀp - ½pᵀHp
                    with torch.no_grad():
                        pred_reduction = float(
                            -(g_flat @ p_flat) - 0.5 * (p_flat @ (H @ p_flat))
                        )
                else:
                    p_flat, lambda_star, step_type, hard_case, n_solves = (
                        efficient_solve_trs(model, x, y, grad_vec, trust_radius)
                    )
                    scalars.tr_solves = float(n_solves)
                    # h_eig_min/max need an eigendecomposition we deliberately
                    # avoid. At the solution (H + λI)p = -g, so pᵀHp = -gᵀp - λ‖p‖²
                    # and the predicted reduction -gᵀp - ½pᵀHp needs no Hessian:
                    with torch.no_grad():
                        pred_reduction = float(
                            -0.5 * (g_flat @ p_flat)
                            + 0.5 * lambda_star * (p_flat @ p_flat)
                        )

                scalars.lambda_star = lambda_star
                scalars.step_type = 1.0 if step_type == "boundary" else 0.0
                scalars.hard_case = 1.0 if hard_case else 0.0
                scalars.trust_radius = trust_radius

                # Pack the step into bpm.Vertical. solve_trs returns the true
                # step p (params should move to params + p), but apply_update
                # *subtracts* its argument (params -= update), matching the SGD
                # `params -= grad` convention. Negate so apply_update(., +1)
                # yields params + p, and the rejection undo apply_update(., -1)
                # cleanly backs it out.
                neg_p = -p_flat
                blocks = []
                offset = 0
                for block in grad_vec.flat:
                    n = block.shape[0]
                    blocks.append(neg_p[offset: offset + n].unsqueeze(1))
                    offset += n
                update = bpm.Vertical(blocks)

                scalars.step_norm = float(p_flat.norm().item())

                # Apply the step and measure actual reduction on the same batch
                apply_update(model, update, 1.0)
                with torch.no_grad():
                    trial_loss = float(model(x, y).item())
                actual_reduction = scalars.loss - trial_loss

                scalars.actual_loss_change = -actual_reduction
                scalars.pred_loss_change = -pred_reduction

                rho = actual_reduction / (pred_reduction + 1e-30)
                scalars.rho = rho

                # Standard trust-region radius update
                if rho < 0.25:
                    trust_radius = trust_radius * 0.25
                elif rho >= 0.75 and step_type == "boundary":
                    trust_radius = min(trust_radius * 2.0, args.delta_max)

                if rho >= args.tr_eta:
                    scalars.accepted = 1.0
                else:
                    apply_update(model, update, -1.0)
                    scalars.accepted = 0.0
            else:
                raise ValueError(f"unknown mode: {args.mode}")

        if (
            args.adapt_lr_on_plateau
            and len(loss_window) == 2 * window
            and (step_idx + 1) % window == 0
            and lr > args.lr_min
        ):
            recent = sum(list(loss_window)[-window:]) / window
            prev = sum(list(loss_window)[:window]) / window
            if recent >= prev - args.lr_decay_tol:
                lr = max(lr * args.lr_decay_factor, args.lr_min)

        if step_idx % args.log_every == 0:
            extra = ""
            if args.mode == "newton":
                extra = (
                    f" ε={scalars.epsilon:.2e} {'ok' if scalars.accepted else 'REJ'}"
                )
                if args.newton_step_method != "custom":
                    extra += (
                        f" cos(Δ,-g)={scalars.cos_step_neg_grad:+.3f}"
                        f" pred={scalars.pred_loss_change:+.4f} actual={scalars.actual_loss_change:+.4f}"
                        f" eig=[{scalars.h_eig_min:+.2e},{scalars.h_eig_max:+.2e}]"
                    )
            elif args.mode == "trust-region":
                extra = (
                    f" Δ={scalars.trust_radius:.3e} ρ={scalars.rho:+.3f} λ*={scalars.lambda_star:.3e}"
                    f" {'ok' if scalars.accepted else 'REJ'}"
                    f" {'boundary' if scalars.step_type else 'interior'}"
                    f"{' HARD' if scalars.hard_case else ''}"
                )
                if args.tr_solver == "efficient":
                    extra += f" solves={int(scalars.tr_solves)}"
                else:
                    extra += f" eig=[{scalars.h_eig_min:+.2e},{scalars.h_eig_max:+.2e}]"
            print(
                f"[{args.mode}] step={step_idx:4d} "
                f"loss={scalars.loss:.4f} avg10={scalars.train_loss_avg10:.4f} probe={scalars.probe_loss:.4f} probe_acc={scalars.probe_accuracy:.3f} bacc={scalars.batch_accuracy:.3f} "
                f"lr={scalars.lr:.3e} |g|={scalars.grad_norm:.3e} |Δ|={scalars.step_norm:.3e}{extra} "
                f"t={scalars.wall_clock_s:.2f}s (step {scalars.step_seconds * 1000:.0f}ms)",
                file=sys.stderr,
            )
        step_idx += 1

    logger.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sgd", "newton", "trust-region"], required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument(
        "--logdir",
        default="runs",
        help="Parent directory for tensorboard run subdirectories.",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory name under --logdir. Defaults to <mode>-<timestamp>.",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--probe-batch-size", type=int, default=256,
                   help="Held-out fixed batch used to log probe_loss every step.")
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument(
        "--lr", type=float, default=None, help="Default 0.5 for Newton, 0.1 for SGD."
    )
    p.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Initial Hessian damping for Newton mode. Adapted by Levenberg-Marquardt.",
    )
    p.add_argument(
        "--lm-up",
        type=float,
        default=1.1,
        help="Multiplicative increase in epsilon when a Newton step is rejected.",
    )
    p.add_argument(
        "--lm-down",
        type=float,
        default=0.9,
        help="Multiplicative factor (<1) applied to epsilon when a Newton step is accepted.",
    )
    p.add_argument("--lm-eps-min", type=float, default=1e-6)
    p.add_argument("--lm-eps-max", type=float, default=1e6)
    p.add_argument(
        "--sgd-fallback-lr",
        type=float,
        default=0.05,
        help="SGD step size used when a Newton step is rejected by LM.",
    )
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument(
        "--image-size",
        type=int,
        default=16,
        help="Resize CIFAR images to this height and width before flattening. "
        "Newton mode's cost scales as (input_dim * hidden_dim)^2 per first-layer Hessian block, "
        "so keep input_dim * hidden_dim small (the default config gives a ~150 MB first-layer block).",
    )
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--activation", choices=sorted(ACTIVATIONS.keys()), default="tanh")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Stderr-print every N steps. Default 1 for newton, 10 for sgd.",
    )
    p.add_argument(
        "--reuse-batch",
        type=int,
        default=1,
        help="Reuse the same (x, y) minibatch for this many consecutive steps before drawing a new one.",
    )
    p.add_argument(
        "--max-step-norm",
        type=float,
        default=None,
        help="If set, cap |lr * update| at this value. Newton mode only.",
    )
    p.add_argument(
        "--newton-step-method",
        choices=["custom", "dense-solve", "dense-pinv"],
        default="custom",
        help="How to compute the Newton step. 'custom' uses model.hessian_inverse_product "
        "(linear-time matrix-free, may be numerically unstable for ill-conditioned H). "
        "'dense-solve' materializes the full Hessian and solves (H+εI)x=g. "
        "'dense-pinv' uses torch.linalg.pinv(H+εI) @ g.",
    )
    p.add_argument(
        "--lm-check-batch",
        choices=["same", "fresh"],
        default="same",
        help="LM accept check: 'same' uses the training batch the step came from; "
        "'fresh' draws a separate batch from the loader and checks loss there.",
    )
    p.add_argument(
        "--lr-decay-window",
        type=int,
        default=10,
        help="Width of the trailing-loss window for lr plateau detection.",
    )
    p.add_argument(
        "--adapt-lr-on-plateau",
        action="store_true",
        help="If set, halve lr (down to --lr-min) whenever the trailing-loss mean over the last "
        "--lr-decay-window steps does not improve over the previous window of the same length.",
    )
    p.add_argument("--lr-min", type=float, default=1e-3)
    p.add_argument(
        "--lr-decay-factor",
        type=float,
        default=0.5,
        help="Multiplicative factor applied to lr on a plateau decay event.",
    )
    p.add_argument(
        "--lr-lm-on-accept",
        type=float,
        default=1.0,
        help="Per-step lr multiplier applied when LM accepts (Newton only). 1.0 disables.",
    )
    p.add_argument(
        "--lr-lm-on-reject",
        type=float,
        default=1.0,
        help="Per-step lr multiplier applied when LM rejects (Newton only). 1.0 disables.",
    )
    p.add_argument(
        "--lr-decay-tol",
        type=float,
        default=0.02,
        help="Plateau tolerance: decay lr when (prev_window_mean - recent_window_mean) "
        "is at most this value. Set above the mini-batch standard error to avoid being "
        "tricked by noise-floor descent.",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--tb-log-every",
        type=int,
        default=1,
        help="Write tensorboard scalars every N steps. Use >1 on very long "
        "runs to keep event files small.",
    )
    # Trust-region flags
    p.add_argument(
        "--tr-solver",
        choices=["dense", "efficient"],
        default="dense",
        help=(
            "Trust-region subproblem solver. 'dense' eigendecomposes the full "
            "Hessian (O(P^3)); 'efficient' uses the paper's linear-time damped "
            "solver as an oracle and searches the damping lambda."
        ),
    )
    p.add_argument(
        "--delta-init",
        type=float,
        default=None,
        help="Initial trust radius. Default: auto-calibrate via SGD warmup.",
    )
    p.add_argument(
        "--delta-max",
        type=float,
        default=10.0,
        help="Maximum trust radius.",
    )
    p.add_argument(
        "--tr-eta",
        type=float,
        default=0.1,
        help="Acceptance threshold: accept step if rho >= eta.",
    )
    p.add_argument(
        "--delta-init-steps",
        type=int,
        default=10,
        help="Number of SGD warmup steps used to auto-calibrate the initial trust radius.",
    )
    args = p.parse_args()
    if args.lr is None:
        args.lr = 0.5 if args.mode == "newton" else 0.1
    if args.log_every is None:
        args.log_every = 1 if args.mode in ("newton", "trust-region") else 10
    return args


if __name__ == "__main__":
    train(parse_args())
