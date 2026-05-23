"""Sanity check: compare HF's GGN-vector product (via BackPACK's
`ggn_vector_product_from_plist`) and the raw Hessian-vector product against
explicit autograd-materialized matrices on a tiny problem. This is the C1
debugging gate per Section 3 of the plan.

We build a small 4-layer feed-forward net producing logits, attach a
`CrossEntropyLoss`, materialize J, H_loss, and form `J^T H_loss J v` plus
`H_full v` densely, then compare against the HF backend's products.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from backpack.hessianfree.ggnvp import ggn_vector_product_from_plist
from backpack.hessianfree.hvp import hessian_vector_product


def build_tiny_net(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, hidden, bias=False),
        nn.ReLU(),
        nn.Linear(hidden, out_dim, bias=False),
    )


def flat_params(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def dense_jacobian(outputs: torch.Tensor, plist: list[nn.Parameter]) -> torch.Tensor:
    """Materialize d outputs / d params explicitly. Outputs of shape (N, C).

    Returns a (N*C, P) tensor.
    """

    out_flat = outputs.flatten()
    rows = []
    for i in range(out_flat.numel()):
        grads = torch.autograd.grad(out_flat[i], plist, retain_graph=True,
                                     create_graph=False, allow_unused=True)
        rows.append(torch.cat([
            (g if g is not None else torch.zeros_like(p)).flatten()
            for g, p in zip(grads, plist)
        ]))
    return torch.stack(rows, dim=0)


def dense_loss_hessian_wrt_outputs(loss: torch.Tensor, outputs: torch.Tensor
                                    ) -> torch.Tensor:
    """Materialize d^2 loss / d outputs^2 explicitly. Outputs shape (N, C).

    Returns a (N*C, N*C) tensor.
    """

    grad_out = torch.autograd.grad(loss, outputs, create_graph=True)[0].flatten()
    rows = []
    for i in range(grad_out.numel()):
        grads = torch.autograd.grad(grad_out[i], outputs, retain_graph=True)[0]
        rows.append(grads.flatten())
    return torch.stack(rows, dim=0)


def dense_full_hessian(loss: torch.Tensor, plist: list[nn.Parameter]
                       ) -> torch.Tensor:
    grads = torch.autograd.grad(loss, plist, create_graph=True)
    g_flat = torch.cat([g.flatten() for g in grads])
    P = g_flat.numel()
    H = torch.zeros(P, P)
    for i in range(P):
        row = torch.autograd.grad(g_flat[i], plist, retain_graph=True)
        H[i] = torch.cat([r.flatten() for r in row])
    return H


def main() -> None:
    torch.manual_seed(0)
    N, in_dim, hidden, num_classes = 4, 6, 5, 3

    model = build_tiny_net(in_dim, hidden, num_classes)
    x = torch.randn(N, in_dim)
    y = torch.randint(0, num_classes, (N,))
    plist = flat_params(model)
    P = sum(p.numel() for p in plist)

    outputs = model(x)
    loss = F.cross_entropy(outputs, y)

    v_flat = torch.randn(P)

    # GGN-vector product via HF's backend.
    # The HF library expects vec as a single flat tensor and splits it.
    from hessianfree.utils import vector_to_parameter_list
    from torch.nn.utils import parameters_to_vector
    v_list = vector_to_parameter_list(v_flat, plist)
    Gv_ref = parameters_to_vector(
        ggn_vector_product_from_plist(loss, outputs, plist, v_list)
    ).detach()

    # Dense GGN: J^T H_loss J v.
    # Need a fresh graph since `dense_jacobian` consumes one.
    outputs_jac = model(x)
    J = dense_jacobian(outputs_jac, plist)        # (NC, P)
    loss_jac = F.cross_entropy(outputs_jac, y)
    H_loss = dense_loss_hessian_wrt_outputs(loss_jac, outputs_jac)  # (NC, NC)
    Gv_dense = (J.T @ (H_loss @ (J @ v_flat))).detach()

    rel_err_ggn = (Gv_ref - Gv_dense).norm() / Gv_dense.norm()
    print(f"GGN vp relative error: {rel_err_ggn.item():.3e}")

    # Hessian-vector product via HF's backend.
    outputs_hvp = model(x)
    loss_hvp = F.cross_entropy(outputs_hvp, y)
    v_list_h = vector_to_parameter_list(v_flat, plist)
    Hv_ref = torch.nn.utils.parameters_to_vector(
        hessian_vector_product(loss_hvp, plist, v_list_h)
    ).detach()

    # Dense Hessian.
    outputs_full = model(x)
    loss_full = F.cross_entropy(outputs_full, y)
    H_full = dense_full_hessian(loss_full, plist)
    Hv_dense = (H_full @ v_flat).detach()

    rel_err_hvp = (Hv_ref - Hv_dense).norm() / Hv_dense.norm()
    print(f"Hessian vp relative error: {rel_err_hvp.item():.3e}")

    print()
    print(f"Both products agree to single-precision tolerance "
          f"({rel_err_ggn.item() < 1e-4 and rel_err_hvp.item() < 1e-4}).")


if __name__ == "__main__":
    main()
