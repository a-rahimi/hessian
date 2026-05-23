"""Train the Hinton & Salakhutdinov 2006 deep MNIST autoencoder.

Two optimizers are supported: K-FAC (via the vendored
``third_party/kfac_pytorch`` implementation) and SGD with momentum. Both run
on the same data and step budget so their reconstruction-error curves are
directly comparable to Figure 1 of Martens & Grosse 2015.

Logs are written as JSONL lines under ``logs/`` so the plotting script can
recover them without parsing free text.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "third_party"))
from kfac_pytorch.kfac import KFAC  # noqa: E402  (after sys.path insert)

from data import make_loaders  # noqa: E402
from model import DeepAutoencoder, reconstruction_loss, reconstruction_mse  # noqa: E402


@dataclass
class StepLog:
    step: int
    train_loss: float
    test_loss: float | None
    test_mse: float | None
    wall_seconds: float


def _pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _evaluate(model: nn.Module, test_x: torch.Tensor, batch_size: int) -> tuple[float, float]:
    """Return ``(test_bce, test_mse)`` averaged over the test set.

    The two metrics let us track the training objective (BCE) and the figure-1
    metric from Martens & Grosse 2015 (sum-of-squared-errors per example)
    side by side. The published "K-FAC ~0.96, SGD ~2.0-2.5" numbers are MSE,
    so the MSE column is the one we compare against the gate.
    """
    model.eval()
    total_bce = 0.0
    total_mse = 0.0
    n_batches = 0
    with torch.no_grad():
        for i in range(0, test_x.shape[0] - batch_size + 1, batch_size):
            batch = test_x[i : i + batch_size]
            recon = model(batch)
            total_bce += reconstruction_loss(recon, batch).item()
            total_mse += reconstruction_mse(recon, batch).item()
            n_batches += 1
    model.train()
    return total_bce / max(n_batches, 1), total_mse / max(n_batches, 1)


def _build_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    momentum: float,
    kfac_eps: float,
    kfac_update_freq: int,
    kfac_alpha: float,
    kfac_constraint_norm: bool,
):
    """Return ``(optimizer, preconditioner_or_None)`` for the requested method."""
    if name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        return opt, None
    if name == "kfac":
        # Use SGD as the underlying parameter updater; the KFAC preconditioner
        # rewrites the gradients in-place between ``loss.backward()`` and
        # ``optimizer.step()``. ``constraint_norm`` is the Ba/Grosse/Martens
        # 2017 norm constraint, which keeps the K-FAC step from blowing up
        # without us having to run a per-step LM rule from scratch.
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        preconditioner = KFAC(
            model,
            eps=kfac_eps,
            sua=False,
            pi=True,
            update_freq=kfac_update_freq,
            alpha=kfac_alpha,
            constraint_norm=kfac_constraint_norm,
        )
        return opt, preconditioner
    raise ValueError(f"unknown optimizer {name!r}")


def train(args: argparse.Namespace) -> None:
    device = _pick_device()
    torch.manual_seed(args.seed)

    logs_dir = HERE / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_path = logs_dir / f"{args.tag}.jsonl"
    if log_path.exists():
        log_path.unlink()

    train_loader, test_x = make_loaders(HERE / "data", args.batch_size, device)
    model = DeepAutoencoder().to(device)
    optimizer, preconditioner = _build_optimizer(
        name=args.optimizer,
        model=model,
        lr=args.lr,
        momentum=args.momentum,
        kfac_eps=args.kfac_eps,
        kfac_update_freq=args.kfac_update_freq,
        kfac_alpha=args.kfac_alpha,
        kfac_constraint_norm=args.kfac_constraint_norm,
    )

    print(
        f"[{args.tag}] device={device} optimizer={args.optimizer} lr={args.lr} "
        f"momentum={args.momentum} batch_size={args.batch_size} steps={args.num_steps} "
        f"kfac_eps={args.kfac_eps} kfac_update_freq={args.kfac_update_freq} "
        f"kfac_alpha={args.kfac_alpha}",
        flush=True,
    )

    step = 0
    start = time.perf_counter()
    train_iter = iter(train_loader)
    last_train_loss = float("nan")

    with log_path.open("w") as log_file:
        while step < args.num_steps:
            try:
                (batch,) = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                (batch,) = next(train_iter)

            optimizer.zero_grad(set_to_none=True)
            recon = model(batch)
            loss = reconstruction_loss(recon, batch)
            loss.backward()
            if preconditioner is not None:
                # Optional geometric damping decay. The K-FAC paper recommends
                # adapting damping by the Levenberg-Marquardt rule; this simple
                # decay schedule is a coarse stand-in that lets damping drop as
                # the model improves, which is the empirical effect of LM.
                if args.kfac_damping_decay < 1.0 and step > 0 and step % args.kfac_damping_decay_every == 0:
                    preconditioner.eps = max(preconditioner.eps * args.kfac_damping_decay, args.kfac_damping_floor)
                preconditioner.step()
            optimizer.step()

            last_train_loss = loss.item()

            if step % args.eval_every == 0 or step == args.num_steps - 1:
                test_bce, test_mse = _evaluate(model, test_x, args.batch_size)
                entry = StepLog(
                    step=step,
                    train_loss=last_train_loss,
                    test_loss=test_bce,
                    test_mse=test_mse,
                    wall_seconds=time.perf_counter() - start,
                )
                log_file.write(json.dumps(asdict(entry)) + "\n")
                log_file.flush()
                print(
                    f"[{args.tag}] step={step:>6d}  train_bce={last_train_loss:7.4f}  "
                    f"test_bce={test_bce:7.4f}  test_mse={test_mse:7.4f}  "
                    f"t={entry.wall_seconds:7.1f}s",
                    flush=True,
                )
            step += 1

    final_bce, final_mse = _evaluate(model, test_x, args.batch_size)
    print(
        f"[{args.tag}] FINAL test_bce={final_bce:.4f} test_mse={final_mse:.4f}",
        flush=True,
    )


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", choices=["kfac", "sgd"], required=True)
    p.add_argument("--tag", required=True, help="log file basename under logs/")
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    # SGD-side.
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    # K-FAC-side. ``eps`` is the Tikhonov damping on the Kronecker factors;
    # ``update_freq`` reuses cached inverses for several steps to amortize the
    # O(d^3) factor inverses; ``alpha`` is the EMA on the factor estimates.
    p.add_argument("--kfac-eps", type=float, default=0.1)
    p.add_argument("--kfac-update-freq", type=int, default=100)
    p.add_argument("--kfac-alpha", type=float, default=0.1)
    p.add_argument(
        "--kfac-constraint-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the Ba/Grosse/Martens 2017 norm constraint to the K-FAC step.",
    )
    # Coarse stand-in for the K-FAC paper's Levenberg-Marquardt damping rule.
    p.add_argument(
        "--kfac-damping-decay",
        type=float,
        default=1.0,
        help="Multiplicative damping decay factor applied every "
        "--kfac-damping-decay-every steps (default 1.0 disables the schedule).",
    )
    p.add_argument("--kfac-damping-decay-every", type=int, default=500)
    p.add_argument(
        "--kfac-damping-floor",
        type=float,
        default=1e-4,
        help="Lower bound on damping after decay.",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
