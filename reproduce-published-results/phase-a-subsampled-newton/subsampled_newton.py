"""Sub-sampled Newton driver for the Phase 5 anchor.

Per step, this driver materializes the full per-batch Hessian of the
cross-entropy loss of [SequenceOfDenseBlocks](../../src/hessian.py) via
torch.func.hessian, adds eps * I, solves (H + eps I) delta = g via dense
linear algebra, and applies params -= lr * delta. Metrics match
[train_newton.py](../../src/train_newton.py) so trajectories are
apples-to-apples with the existing Newton and SGD runs.

Run on the Phase 5 anchor (full CIFAR-10):
    python reproduce-published-results/phase-a-subsampled-newton/subsampled_newton.py \
        --num-steps 200 --batch-size 64 --logdir runs --run-name ssn-a1

Run the fixed-batch memorization diagnostic (single batch held for many
steps):
    python ... --num-steps 200 --batch-size 64 --reuse-batch 200 --lr 0.5 --epsilon 1.0
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import dataclasses
import datetime as dt
import sys
import time
from pathlib import Path

import torch
import torchvision

# Make `import hessian` work when this script runs from any cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import hessian as hessian_module  # noqa: E402
from hessian import SequenceOfDenseBlocks  # noqa: E402

CIFAR10_NUM_CLASSES = 10
CIFAR10_CHANNELS = 3

ACTIVATIONS = {
    "tanh": torch.tanh,
    "relu": torch.relu,
    "gelu": torch.nn.functional.gelu,
}


@dataclasses.dataclass
class StepRow:
    step: int = 0
    loss: float = 0.0
    train_loss_avg10: float = 0.0
    probe_loss: float = 0.0
    probe_accuracy: float = 0.0
    batch_accuracy: float = 0.0
    grad_norm: float = 0.0
    step_norm: float = 0.0
    epsilon: float = 0.0
    lr: float = 0.0
    step_seconds: float = 0.0
    wall_clock_s: float = 0.0


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


def build_model(args: argparse.Namespace) -> SequenceOfDenseBlocks:
    input_dim = CIFAR10_CHANNELS * args.image_size * args.image_size
    model = SequenceOfDenseBlocks(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=CIFAR10_NUM_CLASSES,
        num_layers=args.num_layers,
        activation=ACTIVATIONS[args.activation],
    )
    nonlinearity = "relu" if args.activation in {"relu", "gelu"} else "tanh"
    for layer in model:
        torch.nn.init.kaiming_normal_(layer.linear.weight, nonlinearity=nonlinearity)
    return model


def dense_hessian(
    model: SequenceOfDenseBlocks, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """Materialize the full P x P cross-entropy Hessian for the given batch.

    Uses torch.func.hessian on a functional_call closure so we do not need to
    detach and re-attach autograd state on the live parameters.
    """

    def loss_fn(params_dict):
        return torch.func.functional_call(model, params_dict, (x, y))

    hd = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
    H = hessian_module.flatten_2d_pytree(hd)
    # The Hessian is symmetric in theory; symmetrize away the floating-point
    # asymmetry so cholesky_solve does not fail spuriously when eps is large.
    return 0.5 * (H + H.T)


def flat_params(model: SequenceOfDenseBlocks) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def flat_grad(model: SequenceOfDenseBlocks) -> torch.Tensor:
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()])


def apply_flat_update(
    model: SequenceOfDenseBlocks, delta: torch.Tensor, lr: float
) -> None:
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.add_(delta[offset : offset + n].view_as(p), alpha=-lr)
        offset += n
    assert offset == delta.numel()


def solve_damped(H: torch.Tensor, eps: float, g: torch.Tensor) -> torch.Tensor:
    """Solve (H + eps I) delta = g, preferring cholesky_solve when possible.

    If H + eps I is not PD (which happens when H has eigenvalues more negative
    than -eps), fall back to torch.linalg.solve. The fallback is the same
    quantity the spec asks for; only the factorization changes.
    """
    P = H.shape[0]
    A = H + eps * torch.eye(P, dtype=H.dtype, device=H.device)
    try:
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(g.unsqueeze(1), L).squeeze(1)
    except RuntimeError:
        return torch.linalg.solve(A, g)


class StepLogger:
    """CSV logger that mirrors train_newton.py's StepLogger fields."""

    FIELDS = [
        "step",
        "loss",
        "train_loss_avg10",
        "probe_loss",
        "probe_accuracy",
        "batch_accuracy",
        "grad_norm",
        "step_norm",
        "epsilon",
        "lr",
        "step_seconds",
        "wall_clock_s",
    ]

    def __init__(self, csv_path: Path):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fp = csv_path.open("w", newline="")
        self.writer = csv.DictWriter(self.fp, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.t0 = time.perf_counter()

    @contextlib.contextmanager
    def step(self, step_idx: int):
        row = StepRow(step=step_idx)
        t_step = time.perf_counter()
        yield row
        t_now = time.perf_counter()
        row.step_seconds = t_now - t_step
        row.wall_clock_s = t_now - self.t0
        self.writer.writerow({k: getattr(row, k) for k in self.FIELDS})
        self.fp.flush()

    def close(self):
        self.fp.close()


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if args.float64:
        torch.set_default_dtype(torch.float64)
    device = torch.device("cpu")  # MPS solve is slower than CPU on this size.
    print(f"device: {device}, dtype: {torch.get_default_dtype()}", file=sys.stderr)

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
    probe_x = probe_x.to(device).to(torch.get_default_dtype())
    probe_y = probe_y.to(device)

    model = build_model(args).to(device)
    P = sum(p.numel() for p in model.parameters())
    H_mb = (P * P * (8 if args.float64 else 4)) / (1024 ** 2)
    print(
        f"P={P}, dense H is {P}x{P} ~ {H_mb:.1f} MB in current dtype",
        file=sys.stderr,
    )

    run_name = (
        args.run_name
        or f"ssn-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    logdir = Path(args.logdir) / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    csv_path = logdir / "metrics.csv"
    print(f"logdir: {logdir}", file=sys.stderr)
    logger = StepLogger(csv_path)

    eps = args.epsilon
    lr = args.lr
    loss_window = collections.deque(maxlen=2 * args.lr_decay_window)

    data_iter = iter(loader)
    x = y = None
    step_idx = 0
    while step_idx < args.num_steps:
        if step_idx % args.reuse_batch == 0:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x = x.to(device).to(torch.get_default_dtype())
            y = y.to(device)

        with logger.step(step_idx) as row:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            loss = model(x, y)
            row.loss = float(loss.item())
            loss_window.append(row.loss)
            row.train_loss_avg10 = sum(loss_window) / len(loss_window)
            row.epsilon = eps
            row.lr = lr

            with torch.no_grad():
                row.probe_loss = float(model(probe_x, probe_y).item())
                pf = model.layers(probe_x)
                pl = model.loss_layer.linear(pf)
                row.probe_accuracy = float(
                    (pl.argmax(dim=1) == probe_y).float().mean().item()
                )
                bf = model.layers(x)
                bl = model.loss_layer.linear(bf)
                row.batch_accuracy = float(
                    (bl.argmax(dim=1) == y).float().mean().item()
                )

            loss.backward()
            g = flat_grad(model)
            row.grad_norm = float(g.norm().item())

            H = dense_hessian(model, x, y)
            delta = solve_damped(H, eps, g)
            row.step_norm = lr * float(delta.norm().item())
            apply_flat_update(model, delta, lr)

        if step_idx % args.log_every == 0:
            print(
                f"[ssn] step={step_idx:4d} loss={row.loss:.4f} "
                f"avg10={row.train_loss_avg10:.4f} probe={row.probe_loss:.4f} "
                f"probe_acc={row.probe_accuracy:.3f} bacc={row.batch_accuracy:.3f} "
                f"lr={row.lr:.3e} eps={row.epsilon:.3e} "
                f"|g|={row.grad_norm:.3e} |Δ|={row.step_norm:.3e} "
                f"t={row.wall_clock_s:.1f}s (step {row.step_seconds:.2f}s)",
                file=sys.stderr,
            )
        step_idx += 1

    logger.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--logdir", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--activation", choices=sorted(ACTIVATIONS.keys()), default="relu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument(
        "--reuse-batch",
        type=int,
        default=1,
        help="Reuse the same (x, y) for this many consecutive steps before drawing a new one.",
    )
    p.add_argument("--lr-decay-window", type=int, default=10)
    p.add_argument(
        "--float64",
        action="store_true",
        help="Run the entire pipeline in float64. Doubles memory but matches the A3 sanity check.",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
