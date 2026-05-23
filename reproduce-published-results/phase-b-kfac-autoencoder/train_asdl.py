"""Train the Hinton & Salakhutdinov 2006 deep MNIST autoencoder using asdfghjkl.

The Thrandis K-FAC implementation in ``third_party/kfac_pytorch`` reaches the
mean-prediction plateau and gets stuck because its empirical-Fisher
preconditioner vanishes at that local minimum. The asdfghjkl ``KfacGradientMaker``
supports a Monte Carlo Fisher estimate, which samples gradients from model
outputs rather than from the empirical data labels, and does not collapse at
the mean plateau. This script uses asdfghjkl's K-FAC on the same autoencoder
with MSE reconstruction loss (asdl supports ``cross_entropy`` and ``mse``
loss types out of the box).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# asdl warns about the non-full backward hook every step; suppress.
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# asdl unconditionally calls ``torch.cuda.nvtx.range`` in its natural-gradient
# code, which raises on a non-CUDA build. We don't need NVTX profiling, so
# monkey-patch it to a no-op context manager before importing asdl.
import contextlib as _ctx  # noqa: E402

import torch.cuda.nvtx as _nvtx  # noqa: E402


@_ctx.contextmanager
def _nvtx_range_noop(*args, **kwargs):
    yield


_nvtx.range = _nvtx_range_noop  # type: ignore[assignment]
_nvtx.range_push = lambda *_a, **_k: None  # type: ignore[assignment]
_nvtx.range_pop = lambda *_a, **_k: None  # type: ignore[assignment]

import asdl  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
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

    # MSE training objective. asdl drives the ``reduction`` argument itself, so
    # the function must accept and respect it. We use the raw ``F.mse_loss``
    # rather than wrapping with a divide, because asdl's gradient maker assumes
    # ``reduction='sum'`` for unscaled summing.
    loss_fn = F.mse_loss

    config = asdl.PreconditioningConfig(
        data_size=60000,
        damping=args.kfac_eps,
        preconditioner_upd_interval=args.kfac_update_freq,
        curvature_upd_interval=1,
        ema_decay=args.kfac_alpha,
    )
    fisher_type = (
        "fisher_mc" if args.fisher_type == "mc"
        else "fisher_emp" if args.fisher_type == "emp"
        else "fisher_exact"
    )
    grad_maker = asdl.KfacGradientMaker(
        model,
        config,
        fisher_type=fisher_type,
        loss_type="mse",
        n_mc_samples=args.n_mc_samples,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    print(
        f"[{args.tag}] device={device} asdl-KFAC fisher_type={fisher_type} "
        f"lr={args.lr} momentum={args.momentum} batch_size={args.batch_size} "
        f"steps={args.num_steps} eps={args.kfac_eps} update_freq={args.kfac_update_freq}",
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
            dummy_recon = grad_maker.setup_model_call(model, batch)
            grad_maker.setup_loss_call(loss_fn, dummy_recon, batch, reduction="mean")
            _recon, loss = grad_maker.forward_and_backward()
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
                    f"[{args.tag}] step={step:>6d}  train_mse={last_train_loss:7.4f}  "
                    f"test_bce={test_bce:7.4f}  test_mse={test_mse:7.4f}  "
                    f"t={entry.wall_seconds:7.1f}s",
                    flush=True,
                )
            step += 1

    final_bce, final_mse = _evaluate(model, test_x, args.batch_size)
    print(f"[{args.tag}] FINAL test_bce={final_bce:.4f} test_mse={final_mse:.4f}", flush=True)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--kfac-eps", type=float, default=0.1)
    p.add_argument("--kfac-update-freq", type=int, default=20)
    p.add_argument("--kfac-alpha", type=float, default=0.1)
    p.add_argument("--fisher-type", choices=["mc", "emp", "exact"], default="mc")
    p.add_argument("--n-mc-samples", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
