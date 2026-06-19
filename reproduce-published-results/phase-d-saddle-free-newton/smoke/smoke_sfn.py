"""Smoke test: verify the SFN update assembles, descends, and is cheap enough.

Builds the Phase-5 anchor model, draws one batch, computes a gradient and an
SFN step, then steps 5 times and prints the per-step loss along with timing.
The goal is to confirm: (1) the Lanczos / LinearOperator plumbing works on
the actual model and one HVP returns the same `Vertical` shape we feed in,
(2) per-step cost is in the predicted range (~10 s derivatives + a few s
Lanczos), and (3) loss decreases on the very first step.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_PHASE_DIR = _THIS_DIR.parent
if str(_PHASE_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE_DIR))
_SRC_DIR = _PHASE_DIR.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import block_partitioned_matrices as bpm  # noqa: E402
from hessian import SequenceOfDenseBlocks  # noqa: E402
from sfn import (  # noqa: E402
    CachedHessianOperator,
    sfn_update,
    vertical_block_sizes,
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


def main():
    torch.manual_seed(0)
    image_size = 16
    input_dim = 3 * image_size * image_size
    model = SequenceOfDenseBlocks(
        input_dim=input_dim,
        hidden_dim=24,
        num_classes=10,
        num_layers=8,
        activation=torch.relu,
    )
    for layer in model:
        torch.nn.init.kaiming_normal_(layer.linear.weight, nonlinearity="relu")

    P = sum(p.numel() for p in model.parameters())
    print(f"P = {P} params")

    x = torch.randn(64, input_dim)
    y = torch.randint(0, 10, (64,))

    k = 20
    epsilon = 1.0
    lr = 0.5

    for step in range(5):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        loss = model(x, y)
        loss.backward()
        grad_vec = assemble_gradient_vector(model)
        block_sizes = vertical_block_sizes(grad_vec)

        t0 = time.perf_counter()
        op = CachedHessianOperator(
            model, x, y, block_sizes=block_sizes, input_numel=x.numel()
        )
        t_deriv = time.perf_counter() - t0

        t0 = time.perf_counter()
        delta, info = sfn_update(op, grad_vec, k=k, epsilon=epsilon)
        t_lanczos = time.perf_counter() - t0

        apply_update(model, delta, lr)
        with torch.no_grad():
            new_loss = float(model(x, y).item())

        print(
            f"step {step}: loss={loss.item():.4f} -> {new_loss:.4f} "
            f"|deriv {t_deriv:.2f}s |lanczos {t_lanczos:.2f}s "
            f"|matvecs={info['n_matvecs']} "
            f"|lam_max_abs={info['lam_max_abs']:.2e} lam_min_abs={info['lam_min_abs']:.2e} "
            f"frac_neg={info['frac_neg']:.2f} g_K/|g|={info['g_K_norm']:.2e}"
        )


if __name__ == "__main__":
    main()
