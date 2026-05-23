"""Train a SequenceOfDenseBlocks with low-rank Saddle-Free Newton.

This driver mirrors `src/train_newton.py` for SGD and Newton modes, then adds
an `sfn` mode that uses the Krylov / Lanczos SFN step implemented in `sfn.py`.
It shares as much logic with the in-tree trainer as the in-tree trainer
exposes; the rest (CIFAR loader, metric logging, parameter init, LM-style
accept / reject) is duplicated rather than monkey-patched, because the
in-tree CLI did not anticipate a third mode and we want this driver to be
self-contained for the Phase-D experiments.

The SFN mode supports the same LM accept/reject machinery as Newton: it
tries a step, checks loss on the same batch, accepts if loss dropped, and
falls back to an SGD step on rejection while increasing the damping floor
`epsilon`. The orthogonal-complement damping defaults to `|lambda_k| +
epsilon` so the orthogonal step shrinks with the smallest captured
curvature; pass `--eps-perp` to override.

Run:
    python train_sfn.py --mode sfn --num-steps 500 --batch-size 64 \\
        --image-size 16 --hidden-dim 24 --num-layers 8 --activation relu \\
        --k 20 --epsilon 1.0 --lr 0.5 --logdir runs/D1
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

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import block_partitioned_matrices as bpm  # noqa: E402
from hessian import SequenceOfDenseBlocks  # noqa: E402
from sfn import (  # noqa: E402
    CachedHessianOperator,
    sfn_update,
    vertical_block_sizes,
)


CIFAR10_NUM_CLASSES = 10
CIFAR10_CHANNELS = 3
ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
}


@dataclasses.dataclass
class StepScalars:
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
    # SFN diagnostics. Zero in non-sfn modes.
    lam_max_abs: float = 0.0
    lam_min_abs: float = 0.0
    frac_neg: float = 0.0
    n_matvecs: float = 0.0


class StepLogger:
    def __init__(self, logdir: Path, device: torch.device):
        self.writer = SummaryWriter(log_dir=str(logdir))
        self.device = device
        self.t0: float | None = None

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
        for name, value in dataclasses.asdict(scalars).items():
            self.writer.add_scalar(name, value, global_step=step_idx)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


def load_cifar10_loader(
    data_dir: Path, batch_size: int, seed: int, image_size: int
) -> torch.utils.data.DataLoader:
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


def assemble_gradient_vector(model) -> bpm.Vertical:
    blocks = []
    for layer in model:
        flat = torch.cat([p.grad.detach().flatten() for p in layer.parameters()])
        blocks.append(flat.unsqueeze(1))
    return bpm.Vertical(blocks)


def apply_update(model, update: bpm.Vertical, lr: float) -> None:
    for layer, block in zip(model, update.flat):
        offset = 0
        flat = block.flatten()
        for p in layer.parameters():
            n = p.numel()
            p.data.add_(flat[offset : offset + n].view_as(p), alpha=-lr)
            offset += n


def vertical_norm(v: bpm.Vertical) -> float:
    return float(torch.linalg.vector_norm(v.to_tensor()).item())


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"device: {device}", file=sys.stderr)

    loader = load_cifar10_loader(
        Path(args.data_dir), args.batch_size, args.seed, args.image_size
    )
    probe_loader = load_cifar10_loader(
        Path(args.data_dir),
        args.probe_batch_size,
        seed=999_999,
        image_size=args.image_size,
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
    nonlinearity = "relu" if args.activation in {"relu", "gelu"} else "tanh"
    for layer in model:
        torch.nn.init.kaiming_normal_(layer.linear.weight, nonlinearity=nonlinearity)

    run_name = (
        args.run_name or f"{args.mode}-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    logdir = Path(args.logdir) / run_name
    print(f"tensorboard logdir: {logdir}", file=sys.stderr)

    logger = StepLogger(logdir, device=device)
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
            elif args.mode == "sfn":
                block_sizes = vertical_block_sizes(grad_vec)
                op = CachedHessianOperator(
                    model, x, y, block_sizes=block_sizes, input_numel=x.numel()
                )
                update, info = sfn_update(
                    op,
                    grad_vec,
                    k=args.k,
                    epsilon=epsilon,
                    eps_perp=args.eps_perp,
                    lanczos_tol=args.lanczos_tol,
                )
                scalars.lam_max_abs = info["lam_max_abs"]
                scalars.lam_min_abs = info["lam_min_abs"]
                scalars.frac_neg = info["frac_neg"]
                scalars.n_matvecs = info["n_matvecs"]

                raw_step_norm = lr * vertical_norm(update)
                if args.max_step_norm is not None and raw_step_norm > args.max_step_norm:
                    effective_lr = lr * args.max_step_norm / raw_step_norm
                else:
                    effective_lr = lr
                scalars.step_norm = effective_lr * vertical_norm(update)
                apply_update(model, update, effective_lr)
                with torch.no_grad():
                    trial_loss = float(model(x, y).item())
                accept = trial_loss < scalars.loss
                if accept:
                    epsilon = max(epsilon * args.lm_down, args.lm_eps_min)
                    scalars.accepted = 1.0
                else:
                    apply_update(model, update, -effective_lr)
                    apply_update(model, grad_vec, args.sgd_fallback_lr)
                    scalars.step_norm = args.sgd_fallback_lr * vertical_norm(grad_vec)
                    epsilon = min(epsilon * args.lm_up, args.lm_eps_max)
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
            if args.mode == "sfn":
                extra = (
                    f" ε={scalars.epsilon:.2e} {'ok' if scalars.accepted else 'REJ'}"
                    f" λmax={scalars.lam_max_abs:.2e} λmin={scalars.lam_min_abs:.2e}"
                    f" neg={scalars.frac_neg:.2f} mv={int(scalars.n_matvecs)}"
                )
            print(
                f"[{args.mode}] step={step_idx:4d} "
                f"loss={scalars.loss:.4f} avg10={scalars.train_loss_avg10:.4f} "
                f"probe={scalars.probe_loss:.4f} probe_acc={scalars.probe_accuracy:.3f} "
                f"bacc={scalars.batch_accuracy:.3f} "
                f"lr={scalars.lr:.3e} |g|={scalars.grad_norm:.3e} |Δ|={scalars.step_norm:.3e}"
                f"{extra} t={scalars.wall_clock_s:.2f}s (step {scalars.step_seconds * 1000:.0f}ms)",
                file=sys.stderr,
                flush=True,
            )
        step_idx += 1

    logger.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sgd", "sfn"], required=True)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=None,
                   help="Default 0.5 for sfn, 0.1 for sgd.")
    p.add_argument("--epsilon", type=float, default=1.0,
                   help="Initial damping floor; for SFN this is the eps in (|H|+eps I)^-1.")
    p.add_argument("--eps-perp", type=float, default=None,
                   help="Orthogonal-complement damping. Defaults to |lambda_k|+epsilon.")
    p.add_argument("--k", type=int, default=20,
                   help="Number of top-LM eigenpairs Lanczos extracts per step.")
    p.add_argument("--lanczos-tol", type=float, default=1e-3,
                   help="SciPy eigsh convergence tol; loosening cuts matvec count.")
    p.add_argument("--lm-up", type=float, default=1.1)
    p.add_argument("--lm-down", type=float, default=0.9)
    p.add_argument("--lm-eps-min", type=float, default=1e-6)
    p.add_argument("--lm-eps-max", type=float, default=1e6)
    p.add_argument("--sgd-fallback-lr", type=float, default=0.05)
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--activation", choices=sorted(ACTIVATIONS.keys()), default="tanh")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--reuse-batch", type=int, default=1)
    p.add_argument("--max-step-norm", type=float, default=None)
    p.add_argument("--lr-decay-window", type=int, default=10)
    p.add_argument("--adapt-lr-on-plateau", action="store_true")
    p.add_argument("--lr-min", type=float, default=1e-3)
    p.add_argument("--lr-decay-factor", type=float, default=0.5)
    p.add_argument("--lr-decay-tol", type=float, default=0.02)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.lr is None:
        args.lr = 0.5 if args.mode == "sfn" else 0.1
    if args.log_every is None:
        args.log_every = 1 if args.mode == "sfn" else 10
    return args


if __name__ == "__main__":
    train(parse_args())
