"""Phase C1: train the Hinton & Salakhutdinov 2006 deep autoencoder on MNIST
under Hessian-Free and SGD-momentum, then compare reconstruction errors.

Architecture is the canonical 784 -> 1000 -> 500 -> 250 -> 30 -> 250 -> 500 ->
1000 -> 784 sigmoid network with binary cross-entropy reconstruction loss on
the 28x28 pixel values normalized to [0, 1].

Each run logs the per-step training loss and the per-epoch held-out test
reconstruction error to a CSV under --logdir.

Usage:
    python train_autoencoder.py --optimizer hf  --num-steps 400
    python train_autoencoder.py --optimizer sgd --num-steps 400
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Subset

from hessianfree.optimizer import HessianFree


def build_autoencoder(init_std: float | None = None) -> nn.Sequential:
    """Build the canonical Hinton & Salakhutdinov 2006 autoencoder.

    The encoder maps 784 -> 1000 -> 500 -> 250 -> 30 and the decoder mirrors
    it, so the full network has eight Linear+Sigmoid blocks. We end on a
    sigmoid so the output sits in [0, 1] and matches the BCE-against-pixels
    loss BackPACK supports through `BCEWithLogitsLoss`.

    BackPACK's GGN-vector product requires a flat sequential model with
    leaf-module activations rather than `torch.sigmoid` calls, so we use
    `nn.Sigmoid` instances throughout.

    When `init_std` is set, every linear layer's weight is re-initialized
    from `N(0, init_std^2)`. This reproduces the Martens 2010 small-Gaussian
    regime under which SGD without RBM pre-training stalls and HF wins.
    """

    sizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.Sigmoid())
    net = nn.Sequential(*layers)
    if init_std is not None:
        for m in net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=init_std)
                nn.init.zeros_(m.bias)
    return net


def load_mnist(
    data_dir: Path, batch_size: int, num_train: int, num_test: int, seed: int
) -> tuple[DataLoader, DataLoader]:
    """MNIST loaders that flatten to 784-d float vectors in [0, 1].

    `num_train` and `num_test` cap the dataset size so the experiment fits in
    a CPU budget. Pass `0` to use the full split.
    """

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda t: t.flatten()),
        ]
    )
    train_full = torchvision.datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    test_full = torchvision.datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=transform
    )
    if num_train and num_train < len(train_full):
        train_set = Subset(train_full, list(range(num_train)))
    else:
        train_set = train_full
    if num_test and num_test < len(test_full):
        test_set = Subset(test_full, list(range(num_test)))
    else:
        test_set = test_full

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, generator=g, drop_last=True
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean per-pixel BCE on a loader. Lower is better; the H&S paper reports
    the mean-squared per-pixel error, but BCE is the loss our optimizer
    minimizes so we report it directly to keep training-time and eval-time
    metrics on the same scale.
    """

    model.eval()
    total = 0.0
    count = 0
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        loss = F.binary_cross_entropy(out, x, reduction="sum").item()
        total += loss
        count += x.numel()
    return total / count


def train_hf(args: argparse.Namespace, model: nn.Module, device: torch.device,
             train_loader: DataLoader, test_loader: DataLoader,
             writer: csv.writer, epoch_writer: csv.writer) -> None:
    """HF training loop. Uses BCE-with-logits-style loss but our model already
    ends in a sigmoid, so we call plain `binary_cross_entropy` per batch.
    """

    opt = HessianFree(
        model.parameters(),
        curvature_opt=args.curvature_opt,
        damping=args.damping,
        adapt_damping=True,
        cg_max_iter=args.cg_max_iter,
        lr=1.0,
        use_linesearch=True,
        use_cg_backtracking=True,
        verbose=False,
    )

    step_idx = 0
    t0 = time.perf_counter()
    epoch = 0
    while step_idx < args.num_steps:
        epoch += 1
        for x, _ in train_loader:
            if step_idx >= args.num_steps:
                break
            x = x.to(device)

            def forward():
                out = model(x)
                loss = F.binary_cross_entropy(out, x, reduction="mean")
                return loss, out

            opt.step(forward=forward, test_deterministic=(step_idx == 0))
            with torch.no_grad():
                tr_loss = forward()[0].item()
            elapsed = time.perf_counter() - t0
            writer.writerow([step_idx, tr_loss, elapsed])
            print(f"[hf] step={step_idx} train_bce={tr_loss:.4f} t={elapsed:.1f}s",
                  flush=True)
            step_idx += 1

        # End of epoch eval.
        test_bce_per_pixel = evaluate(model, test_loader, device)
        epoch_writer.writerow([epoch, step_idx, test_bce_per_pixel,
                                time.perf_counter() - t0])
        print(f"[hf] epoch={epoch} step={step_idx} "
              f"test_bce_per_pixel={test_bce_per_pixel:.5f}", flush=True)


def train_sgd(args: argparse.Namespace, model: nn.Module, device: torch.device,
              train_loader: DataLoader, test_loader: DataLoader,
              writer: csv.writer, epoch_writer: csv.writer) -> None:
    """SGD-with-momentum baseline. We use the per-batch BCE so its loss is
    directly comparable to HF's per-batch BCE.
    """

    opt = torch.optim.SGD(model.parameters(), lr=args.sgd_lr,
                           momentum=args.sgd_momentum)

    step_idx = 0
    t0 = time.perf_counter()
    epoch = 0
    while step_idx < args.num_steps:
        epoch += 1
        for x, _ in train_loader:
            if step_idx >= args.num_steps:
                break
            x = x.to(device)
            opt.zero_grad()
            out = model(x)
            loss = F.binary_cross_entropy(out, x, reduction="mean")
            loss.backward()
            opt.step()
            elapsed = time.perf_counter() - t0
            writer.writerow([step_idx, loss.item(), elapsed])
            if step_idx % 25 == 0:
                print(f"[sgd] step={step_idx} train_bce={loss.item():.4f} "
                      f"t={elapsed:.1f}s", flush=True)
            step_idx += 1

        test_bce_per_pixel = evaluate(model, test_loader, device)
        epoch_writer.writerow([epoch, step_idx, test_bce_per_pixel,
                                time.perf_counter() - t0])
        print(f"[sgd] epoch={epoch} step={step_idx} "
              f"test_bce_per_pixel={test_bce_per_pixel:.5f}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--optimizer", choices=["hf", "sgd"], required=True)
    p.add_argument("--num-steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=200)
    p.add_argument("--num-train", type=int, default=10000,
                   help="Subsample train set to keep CPU runtime reasonable. "
                   "Set to 0 to use the full 60k.")
    p.add_argument("--num-test", type=int, default=2000)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--logdir", default="./logs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--seed", type=int, default=0)
    # HF-specific
    p.add_argument("--curvature-opt", choices=["ggn", "hessian"], default="ggn")
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--cg-max-iter", type=int, default=50)
    # SGD-specific
    p.add_argument("--sgd-lr", type=float, default=0.1)
    p.add_argument("--sgd-momentum", type=float, default=0.9)
    p.add_argument("--init-std", type=float, default=None,
                   help="If set, every Linear weight is sampled from "
                   "N(0, init_std**2). Use a small value (e.g. 0.01) to "
                   "reproduce the Martens 2010 vanishing-gradient regime "
                   "where SGD without RBM pretraining stalls and HF wins.")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"device={device}", flush=True)

    train_loader, test_loader = load_mnist(
        Path(args.data_dir), args.batch_size, args.num_train, args.num_test, args.seed
    )
    model = build_autoencoder(init_std=args.init_std).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"num_params={num_params:_}", flush=True)

    run_name = args.run_name or f"{args.optimizer}-seed{args.seed}"
    logdir = Path(args.logdir) / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    with open(logdir / "step_log.csv", "w", newline="") as f_step, \
         open(logdir / "epoch_log.csv", "w", newline="") as f_epoch:
        step_writer = csv.writer(f_step)
        step_writer.writerow(["step", "train_bce_mean", "wall_s"])
        epoch_writer = csv.writer(f_epoch)
        epoch_writer.writerow(["epoch", "step", "test_bce_per_pixel", "wall_s"])

        # Initial eval at step 0.
        test_bce0 = evaluate(model, test_loader, device)
        epoch_writer.writerow([0, 0, test_bce0, 0.0])
        print(f"[{args.optimizer}] init test_bce_per_pixel={test_bce0:.5f}",
              flush=True)

        if args.optimizer == "hf":
            train_hf(args, model, device, train_loader, test_loader,
                     step_writer, epoch_writer)
        else:
            train_sgd(args, model, device, train_loader, test_loader,
                      step_writer, epoch_writer)

    # Final eval.
    final_test_bce = evaluate(model, test_loader, device)
    print(f"[{args.optimizer}] FINAL test_bce_per_pixel={final_test_bce:.5f}",
          flush=True)


if __name__ == "__main__":
    main()
