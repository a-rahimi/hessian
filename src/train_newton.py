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

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

import block_partitioned_matrices as bpm
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


class StepLogger:
    """Per-step tensorboard logger.

    Times each step end-to-end. The training loop writes fields onto the
    `StepScalars` yielded by `step()`; on exit, the logger stamps
    `step_seconds` and `wall_clock_s` and forwards every field to a
    `SummaryWriter`. Scalars are written under their plain names (e.g.
    "loss", "grad_norm") so multiple runs in the same parent logdir overlay
    automatically.
    """

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
                update = model.hessian_inverse_product(x, y, grad_vec, epsilon)
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
    p.add_argument("--mode", choices=["sgd", "newton"], required=True)
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
    args = p.parse_args()
    if args.lr is None:
        args.lr = 0.5 if args.mode == "newton" else 0.1
    if args.log_every is None:
        args.log_every = 1 if args.mode == "newton" else 10
    return args


if __name__ == "__main__":
    train(parse_args())
