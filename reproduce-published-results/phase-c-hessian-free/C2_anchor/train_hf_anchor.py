"""Phase C2: train the Phase 5 anchor model under Hessian-Free in both
Gauss-Newton (default) and raw-Hessian modes, and dump per-step metrics that
match the schema produced by `src/train_newton.py`.

The anchor matches `train_newton.py` defaults: `num_layers=8`, `hidden_dim=24`,
`image_size=16`, `relu` activation, batch size 64. We mirror the architecture
as a plain `nn.Sequential` of `nn.Linear` + `nn.ReLU` modules (with the same
no-bias linear layers as the project's `DenseBlock`) so BackPACK can hook into
it. The final linear layer feeds `nn.CrossEntropyLoss`, which BackPACK
supports.

Usage:
    python train_hf_anchor.py --curvature-opt ggn     --num-steps 1000
    python train_hf_anchor.py --curvature-opt hessian --num-steps 1000
"""

from __future__ import annotations

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from hessianfree.optimizer import HessianFree


CIFAR10_NUM_CLASSES = 10
CIFAR10_CHANNELS = 3


def build_flat_anchor(input_dim: int, hidden_dim: int, num_classes: int,
                      num_layers: int) -> nn.Sequential:
    """Plain `nn.Sequential` mirror of the project's `SequenceOfDenseBlocks`.

    The project model has `num_layers - 1` DenseBlocks producing
    `hidden_dim` features, each `linear` no-bias followed by ReLU, then one
    `LossLayer` no-bias linear producing class logits that feeds the fused
    cross-entropy. Here we keep the seven Linear+ReLU stages plus the final
    logit Linear, and let the caller apply `F.cross_entropy` outside the model
    so BackPACK can attach to its `CrossEntropyLoss` extension.
    """

    layers: list[nn.Module] = []
    layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
    layers.append(nn.ReLU())
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, num_classes, bias=False))
    return nn.Sequential(*layers)


def kaiming_init(model: nn.Sequential) -> None:
    """Mirror `train_newton.py`'s init: kaiming-normal on every Linear."""

    for m in model:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")


def load_cifar10_loader(data_dir: Path, batch_size: int, seed: int,
                        image_size: int) -> torch.utils.data.DataLoader:
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda t: t.flatten()),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=g, drop_last=True
    )


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}", file=sys.stderr)

    loader = load_cifar10_loader(
        Path(args.data_dir), args.batch_size, args.seed, args.image_size
    )
    probe_loader = load_cifar10_loader(
        Path(args.data_dir), args.probe_batch_size, seed=999_999,
        image_size=args.image_size
    )
    probe_x, probe_y = next(iter(probe_loader))
    probe_x = probe_x.to(device)
    probe_y = probe_y.to(device)

    input_dim = CIFAR10_CHANNELS * args.image_size * args.image_size
    model = build_flat_anchor(input_dim, args.hidden_dim, CIFAR10_NUM_CLASSES,
                              args.num_layers).to(device)
    kaiming_init(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params={num_params}", file=sys.stderr)

    opt = HessianFree(
        model.parameters(),
        curvature_opt=args.curvature_opt,
        damping=args.damping,
        adapt_damping=True,
        cg_max_iter=args.cg_max_iter,
        lr=1.0,
        use_linesearch=True,
        # The library's `cg_efficient_backtracking` crashes with an
        # `UnboundLocalError` when CG terminates with no candidate iterations
        # (which happens early in our 22k-param anchor when the first quadratic
        # model gets trivially solved). Disable backtracking so the final CG
        # iterate is used directly.
        use_cg_backtracking=False,
        verbose=False,
    )

    logdir = Path(args.logdir) / args.run_name
    logdir.mkdir(parents=True, exist_ok=True)
    log_path = logdir / "step_log.csv"
    f_log = open(log_path, "w", newline="")
    log_writer = csv.writer(f_log)
    log_writer.writerow([
        "step", "loss", "train_loss_avg10", "probe_loss", "probe_accuracy",
        "batch_accuracy", "step_seconds", "wall_clock_s",
    ])

    loss_window: collections.deque[float] = collections.deque(maxlen=10)
    t0 = time.perf_counter()
    step_idx = 0
    data_iter = iter(loader)

    # Fixed-batch mode: read the first batch and keep using it.
    fixed_x = fixed_y = None
    if args.fixed_batch:
        fixed_x, fixed_y = next(data_iter)
        fixed_x = fixed_x.to(device); fixed_y = fixed_y.to(device)

    while step_idx < args.num_steps:
        if args.fixed_batch:
            x, y = fixed_x, fixed_y
        else:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x = x.to(device); y = y.to(device)

        t_step = time.perf_counter()

        def forward():
            outputs = model(x)
            loss = F.cross_entropy(outputs, y)
            return loss, outputs

        with torch.no_grad():
            pre_loss = float(forward()[0].item())
        loss_window.append(pre_loss)

        opt.step(forward=forward, test_deterministic=(step_idx == 0))

        with torch.no_grad():
            post_loss = float(forward()[0].item())
            if not torch.isfinite(torch.tensor(post_loss)):
                # Once HF blows past the basin its quadratic-model damping can
                # no longer recover. Stop early so the log file ends with the
                # last finite measurement rather than a tail of NaNs.
                print(f"[hf-{args.curvature_opt}] NaN at step {step_idx}, "
                      f"stopping early.", file=sys.stderr, flush=True)
                break
            probe_out = model(probe_x)
            probe_loss = float(F.cross_entropy(probe_out, probe_y).item())
            probe_acc = float((probe_out.argmax(dim=1) == probe_y).float().mean().item())
            batch_out = model(x)
            batch_acc = float((batch_out.argmax(dim=1) == y).float().mean().item())

        avg10 = sum(loss_window) / len(loss_window)
        step_seconds = time.perf_counter() - t_step
        wall = time.perf_counter() - t0
        log_writer.writerow([
            step_idx, post_loss, avg10, probe_loss, probe_acc, batch_acc,
            step_seconds, wall,
        ])
        if step_idx % args.log_every == 0:
            print(f"[hf-{args.curvature_opt}] step={step_idx:4d} "
                  f"pre={pre_loss:.4f} post={post_loss:.4f} avg10={avg10:.4f} "
                  f"probe={probe_loss:.4f} probe_acc={probe_acc:.3f} "
                  f"step={step_seconds*1000:.0f}ms t={wall:.1f}s",
                  file=sys.stderr, flush=True)
        step_idx += 1

    f_log.close()
    print(f"[hf-{args.curvature_opt}] DONE log={log_path}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--curvature-opt", choices=["ggn", "hessian"], default="ggn")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--logdir", default="./logs")
    p.add_argument("--run-name", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--num-steps", type=int, default=1000)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--cg-max-iter", type=int, default=50)
    p.add_argument("--fixed-batch", action="store_true",
                   help="Reuse a single batch for the entire run "
                   "(the exp-048b-style memorization diagnostic).")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
