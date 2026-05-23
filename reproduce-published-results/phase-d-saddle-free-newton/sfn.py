"""Low-rank Saddle-Free Newton step.

Implements the Krylov / Lanczos variant of Dauphin et al. 2014
(arXiv 1406.2572) and the HVP-only scalable form of arXiv 2002.02881.
The idea is to replace `(H + eps I)^{-1} g` with `(|H| + eps I)^{-1} g`,
where the absolute value is applied eigenvalue-wise. We approximate `|H|`
on the top-k eigenspace via Lanczos and fall back to a damped diagonal on
the orthogonal complement, so the step is

    delta = sum_i  (v_i^T g) / (|lambda_i| + eps_K) * v_i
          + (g - g_K) / (|lambda_k| + eps_perp),

where `(lambda_i, v_i)` are the top-k largest-magnitude eigenpairs of the
per-batch Hessian extracted by SciPy's `eigsh`, `g_K` is the projection of
`g` onto the Krylov subspace, and `|lambda_k|` is the smallest of the
extracted absolute eigenvalues, used as the orthogonal-complement floor.

We assume the caller wants `delta` in the same block-partitioned `Vertical`
layout that `model.hessian_inverse_product` returns, so we round-trip
between a flat float32 numpy array (for SciPy) and a `bpm.Vertical` (for
`apply_update`) at the LinearOperator boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh

# Allow `import block_partitioned_matrices as bpm` and `from hessian import ...`
# from the project's `src/` layout, regardless of where the driver is launched.
_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import block_partitioned_matrices as bpm  # noqa: E402
from hessian import SequenceOfBlocks  # noqa: E402


def vertical_block_sizes(v: bpm.Vertical) -> List[int]:
    """Per-layer block heights, in `Vertical.flat` order."""
    return [b.shape[0] for b in v.flat]


def vertical_to_flat(v: bpm.Vertical) -> torch.Tensor:
    """Flatten a Vertical of (n_i, 1) blocks into a 1-D torch tensor."""
    return torch.cat([b.flatten() for b in v.flat])


def flat_to_vertical(flat: torch.Tensor, block_sizes: List[int]) -> bpm.Vertical:
    """Unpack a 1-D torch tensor back into a Vertical with the given block sizes."""
    blocks = []
    offset = 0
    for n in block_sizes:
        blocks.append(flat[offset : offset + n].reshape(n, 1).contiguous())
        offset += n
    assert offset == flat.numel()
    return bpm.Vertical(blocks)


class CachedHessianOperator:
    """LinearOperator wrapping `model.hessian_vector_product` on a fixed (x, y).

    We compute the layer-wise derivatives (the expensive part of an HVP) once
    per construction, then each `matvec` call only assembles the four terms
    of the Pearlmutter expansion. On the Phase-5 anchor this drops a Lanczos
    sweep from `k * ~10s` to `~10s + k * ~0.1s`, because `model.derivatives`
    is what dominates a single HVP call.
    """

    def __init__(
        self,
        model: SequenceOfBlocks,
        x: torch.Tensor,
        y: torch.Tensor,
        block_sizes: List[int],
        input_numel: int,
    ):
        self.block_sizes = block_sizes
        self.P = sum(block_sizes)
        self.shape = (self.P, self.P)
        self.dtype = np.float32

        Dx, Dz, DD_Dxx, DD_Dzx, DM_Dzz = model.derivatives(x, y)
        self.Dx = Dx
        self.DD_Dxx = DD_Dxx
        self.DD_Dzx = DD_Dzx
        self.DM_Dzz = DM_Dzz
        self.M = bpm.IdentityWithLowerDiagonal((-Dz).flat[1:])
        self.P_perm = bpm.downshifting_matrix(
            input_numel, [b.shape[0] for b in Dx.flatten()]
        )

        self.n_matvecs = 0

    def _hvp(self, v: bpm.Vertical) -> bpm.Vertical:
        """Apply the per-batch Hessian to `v` using the cached derivatives.

        Mirrors `SequenceOfBlocks.hessian_vector_product` after the
        `self.derivatives(...)` call, so the two stay in sync by construction.
        """
        Dx, DD_Dxx, DD_Dzx, DM_Dzz = self.Dx, self.DD_Dxx, self.DD_Dzx, self.DM_Dzz
        M = self.M
        P_perm = self.P_perm
        t1 = P_perm @ M.solve(Dx @ v)
        return (
            DD_Dxx @ v
            + DD_Dzx @ t1
            + Dx.T @ M.T.solve((P_perm.T @ (DD_Dzx.T @ v)))
            + Dx.T @ M.T.solve((P_perm.T @ (DM_Dzz @ t1)))
        )

    def matvec(self, np_v: np.ndarray) -> np.ndarray:
        """SciPy LinearOperator hook. Bridges numpy <-> Vertical."""
        self.n_matvecs += 1
        flat = torch.from_numpy(np.ascontiguousarray(np_v, dtype=np.float32))
        v = flat_to_vertical(flat, self.block_sizes)
        hv = self._hvp(v)
        return vertical_to_flat(hv).detach().cpu().numpy().astype(np.float32)


def lanczos_top_k_eigenpairs(
    op: CachedHessianOperator, k: int, tol: float = 1e-4, maxiter: int | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Run SciPy's implicitly-restarted Lanczos for the top-k LM eigenpairs.

    Returns `(eigvals, eigvecs)` with eigvals shape `(k,)` and eigvecs shape
    `(P, k)`. `which="LM"` requests largest-magnitude, which is what we want
    for the saddle-free transform: we need the eigenvalues whose absolute
    value dominates the rest, regardless of sign.
    """
    linop = LinearOperator(
        shape=op.shape, matvec=op.matvec, dtype=op.dtype
    )
    # Cap `k` at P-1 since eigsh requires k < n.
    k_eff = min(k, op.P - 1)
    eigvals, eigvecs = eigsh(
        linop, k=k_eff, which="LM", tol=tol, maxiter=maxiter
    )
    return eigvals, eigvecs


def sfn_update(
    op: CachedHessianOperator,
    grad_vec: bpm.Vertical,
    k: int,
    epsilon: float,
    eps_perp: float | None = None,
    lanczos_tol: float = 1e-4,
    lanczos_maxiter: int | None = None,
) -> Tuple[bpm.Vertical, dict]:
    """Compute the SFN step `(|H| + eps I)^{-1} g` via low-rank Lanczos.

    `eps_perp` controls damping on the Krylov-orthogonal complement; if
    `None`, we default to `|lambda_k| + epsilon`, the smallest extracted
    absolute eigenvalue plus the same damping floor. This is the simpler
    of the two choices listed in the plan; treating the complement as
    "diagonally damped with the smallest captured curvature" matches the
    intuition that the orthogonal directions are at most as curved as the
    smallest captured direction (LM-style fallback).

    Returns `(delta, info)`. `delta` is a `Vertical` in the same layout as
    `grad_vec`; `info` is a small diagnostics dict.
    """
    block_sizes = op.block_sizes
    g_flat = vertical_to_flat(grad_vec).detach().cpu().numpy().astype(np.float32)

    eigvals, eigvecs = lanczos_top_k_eigenpairs(
        op, k=k, tol=lanczos_tol, maxiter=lanczos_maxiter
    )
    abs_eigvals = np.abs(eigvals)
    lam_min_abs = float(abs_eigvals.min())

    # Subspace projection coefficients: c_i = v_i^T g, in float32.
    coeffs = eigvecs.T @ g_flat  # shape (k,)
    # Subspace gradient component reconstructed in the full space, for the
    # orthogonal-complement subtraction below.
    g_K = eigvecs @ coeffs  # shape (P,)
    g_perp = g_flat - g_K

    # In-subspace scaled coefficients: divide by |lambda_i| + epsilon.
    scaled = coeffs / (abs_eigvals + epsilon)
    delta_K = eigvecs @ scaled  # shape (P,)

    # Orthogonal complement: SGD-like step with a curvature floor.
    if eps_perp is None:
        eps_perp_used = lam_min_abs + epsilon
    else:
        eps_perp_used = eps_perp
    delta_perp = g_perp / eps_perp_used

    delta_flat = torch.from_numpy(delta_K + delta_perp)
    delta = flat_to_vertical(delta_flat, block_sizes)

    info = {
        "eigvals": eigvals,  # signed
        "lam_max_abs": float(abs_eigvals.max()),
        "lam_min_abs": lam_min_abs,
        "frac_neg": float((eigvals < 0).mean()),
        "g_K_norm": float(np.linalg.norm(g_K)),
        "g_perp_norm": float(np.linalg.norm(g_perp)),
        "eps_perp_used": eps_perp_used,
        "n_matvecs": op.n_matvecs,
    }
    return delta, info
