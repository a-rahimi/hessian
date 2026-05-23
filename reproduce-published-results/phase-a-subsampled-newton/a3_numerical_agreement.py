"""A3 — numerical agreement of dense and linear-inverse Newton steps.

For the first batch produced by the data loader of the Phase 5 anchor, compute
both delta_dense = (H + eps I)^{-1} g via torch.linalg.solve on the dense
Hessian materialized by torch.func.hessian, and delta_ours via
[hessian_inverse_product](../../src/hessian.py). Report
||delta_dense - delta_ours|| / ||delta_dense|| for eps in {1.0, 0.1}.

The Section 3 sanity check at [SECTION3_RESULT.md](./SECTION3_RESULT.md)
established the same equality on a 28-parameter model; A3 lifts the check to
the real anchor with P ~ 22k.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import block_partitioned_matrices as bpm  # noqa: E402
import hessian as hessian_module  # noqa: E402

# Reuse the driver's data loading and model construction so the configuration
# matches A1 / A2 exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from subsampled_newton import (  # noqa: E402
    build_model,
    load_cifar10_loader,
)


def assemble_gradient_vertical(model) -> bpm.Vertical:
    """Pack each layer's parameter gradients into a per-layer column block.

    Matches the layout that hessian_inverse_product expects, copied from
    [assemble_gradient_vector](../../src/train_newton.py).
    """
    blocks = []
    for layer in model:
        flat = torch.cat([p.grad.detach().flatten() for p in layer.parameters()])
        blocks.append(flat.unsqueeze(1))
    return bpm.Vertical(blocks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--probe-batch-size", type=int, default=256)
    p.add_argument("--hidden-dim", type=int, default=24)
    p.add_argument("--image-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--activation", default="relu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--float64", action="store_true",
                   help="Use float64 throughout so float32 round-off does not dominate.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.float64:
        torch.set_default_dtype(torch.float64)

    loader = load_cifar10_loader(
        Path(args.data_dir), args.batch_size, args.seed, args.image_size
    )
    x, y = next(iter(loader))
    x = x.to(torch.get_default_dtype())

    model = build_model(args)
    P = sum(p.numel() for p in model.parameters())
    print(f"P = {P}, dtype = {torch.get_default_dtype()}")
    print(f"batch x: shape={tuple(x.shape)}, y: shape={tuple(y.shape)}")

    # Run a forward + backward so each parameter has a .grad we can pack into
    # the per-layer Vertical that hessian_inverse_product consumes.
    for p_ in model.parameters():
        if p_.grad is not None:
            p_.grad.zero_()
    loss = model(x, y)
    print(f"initial loss = {loss.item():.4f}")
    loss.backward()
    g_vert = assemble_gradient_vertical(model)
    g_flat = g_vert.to_tensor().flatten()

    # Dense Hessian via torch.func.hessian, then symmetrize. Same code path as
    # the driver, so any discrepancy is genuine, not a bookkeeping difference.
    def loss_fn(params_dict):
        return torch.func.functional_call(model, params_dict, (x, y))

    print("computing dense Hessian via torch.func.hessian ...")
    hd = torch.func.hessian(loss_fn)(dict(model.named_parameters()))
    H = hessian_module.flatten_2d_pytree(hd)
    H = 0.5 * (H + H.T)

    eye = torch.eye(P, dtype=H.dtype)
    for eps in [1.0, 0.1]:
        delta_dense = torch.linalg.solve(H + eps * eye, g_flat)
        delta_ours = (
            model.hessian_inverse_product(x, y, g_vert, eps).to_tensor().flatten()
        )
        num = torch.linalg.vector_norm(delta_dense - delta_ours).item()
        den = torch.linalg.vector_norm(delta_dense).item()
        print(
            f"eps={eps:<6g}  |delta_dense|={den:.3e}  |delta_dense - delta_ours|={num:.3e}  "
            f"rel_err={num / den:.3e}"
        )


if __name__ == "__main__":
    main()
