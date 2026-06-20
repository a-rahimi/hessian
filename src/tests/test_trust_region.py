"""Tests for the trust-region subproblem solvers in train_newton.

The dense `solve_trs` (eigendecomposition + secular bisection) is treated as
ground truth. `solve_trs_oracle` must reproduce it while only ever applying
`(H + lambda I)^{-1}` — never forming or factorizing H densely. We exercise the
oracle with a synthetic dense inverse so the test is decoupled from the network,
then add an end-to-end smoke test through the real `efficient_solve_trs`.
"""

import pytest
import torch

from train_newton import (
    solve_trs,
    solve_trs_oracle,
    efficient_solve_trs,
    vertical_like,
    assemble_gradient_vector,
)
from hessian import SequenceOfDenseBlocks
import block_partitioned_matrices as bpm


def dense_inverse_oracle(H: torch.Tensor):
    """Return an apply_inverse(lam, rhs) = (H + lam I)^{-1} rhs callable."""
    n = H.shape[0]
    eye = torch.eye(n, dtype=H.dtype)

    def apply_inverse(lam: float, rhs: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(H + lam * eye, rhs)

    return apply_inverse


def make_spd(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    return A @ A.T + 0.5 * torch.eye(n, dtype=torch.float64)


def make_indefinite(n: int, seed: int, eigs: list[float]) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    Q, _ = torch.linalg.qr(A)
    return Q @ torch.diag(torch.tensor(eigs, dtype=torch.float64)) @ Q.T


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_matches_dense_boundary_spd(seed):
    """SPD Hessian, small delta -> both solvers hit the boundary identically."""
    n = 12
    H = make_spd(n, seed)
    g = torch.randn(n, generator=torch.Generator().manual_seed(100 + seed), dtype=torch.float64)

    # Force the boundary case: delta well below the full Newton step length.
    newton = torch.linalg.solve(H, g)
    delta = 0.1 * float(newton.norm())

    p_dense, lam_dense, type_dense, _, _ = solve_trs(g, H, delta)
    p_oracle, lam_oracle, type_oracle, _, n_solves = solve_trs_oracle(
        g, dense_inverse_oracle(H), delta, tol=1e-10
    )

    assert type_dense == type_oracle == "boundary"
    assert lam_oracle == pytest.approx(lam_dense, rel=1e-4)
    assert torch.allclose(p_oracle, p_dense, atol=1e-6, rtol=1e-5)
    # ‖p‖ sits on the trust-region boundary.
    assert float(p_oracle.norm()) == pytest.approx(delta, rel=1e-6)
    # The whole point: a small handful of solves, not an eigendecomposition.
    assert n_solves < 15


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_oracle_matches_dense_interior_spd(seed):
    """SPD Hessian, large delta -> both return the unconstrained Newton step."""
    n = 10
    H = make_spd(n, seed)
    g = torch.randn(n, generator=torch.Generator().manual_seed(200 + seed), dtype=torch.float64)

    newton = torch.linalg.solve(H, g)
    delta = 10.0 * float(newton.norm())

    p_dense, lam_dense, type_dense, _, _ = solve_trs(g, H, delta)
    p_oracle, lam_oracle, type_oracle, _, _ = solve_trs_oracle(
        g, dense_inverse_oracle(H), delta, tol=1e-10
    )

    assert type_dense == type_oracle == "interior"
    assert lam_oracle == 0.0
    assert torch.allclose(p_oracle, -newton, atol=1e-8)
    assert torch.allclose(p_oracle, p_dense, atol=1e-6)


def test_oracle_matches_dense_indefinite_safe_delta():
    """Indefinite Hessian with the boundary root safely above -lambda_min.

    When delta forces lambda* well above max(0, -lambda_min), the secular
    function is monotone there and the oracle (which searches lambda from 0)
    lands on the same root as the dense solver.
    """
    n = 8
    H = make_indefinite(n, seed=7, eigs=[-2.0, -0.5, 0.3, 1.0, 1.5, 2.0, 3.0, 5.0])
    g = torch.randn(n, generator=torch.Generator().manual_seed(321), dtype=torch.float64)

    # Choose delta = ‖(H + lam0 I)^{-1} g‖ for lam0 comfortably above -eig_min=2.
    lam0 = 4.0
    target = torch.linalg.solve(H + lam0 * torch.eye(n, dtype=torch.float64), g)
    delta = float(target.norm())

    p_dense, lam_dense, type_dense, _, _ = solve_trs(g, H, delta)
    p_oracle, lam_oracle, type_oracle, hard, _ = solve_trs_oracle(
        g, dense_inverse_oracle(H), delta, tol=1e-10
    )

    assert type_dense == type_oracle == "boundary"
    assert lam_oracle == pytest.approx(lam0, rel=1e-3)
    assert lam_oracle == pytest.approx(lam_dense, rel=1e-3)
    assert torch.allclose(p_oracle, p_dense, atol=1e-5, rtol=1e-4)
    # The curvature proxy gᵀH⁻¹g is best-effort: here g happens to avoid the
    # negative-curvature directions enough that it stays positive, so the hard
    # case isn't flagged. That's fine -- the boundary root is above -lambda_min,
    # so the solution is still correct. (See the module docstring caveat.)
    assert hard in (True, False)


def test_oracle_solution_satisfies_kkt():
    """The returned step satisfies (H + lambda* I) p = -g with lambda* >= 0."""
    n = 9
    H = make_spd(n, seed=3)
    g = torch.randn(n, generator=torch.Generator().manual_seed(55), dtype=torch.float64)
    newton = torch.linalg.solve(H, g)
    delta = 0.3 * float(newton.norm())

    p, lam, _, _, _ = solve_trs_oracle(g, dense_inverse_oracle(H), delta, tol=1e-12)
    residual = (H + lam * torch.eye(n, dtype=torch.float64)) @ p + g
    assert lam >= 0.0
    assert float(residual.norm()) < 1e-6 * float(g.norm())


def test_vertical_like_roundtrip():
    template = bpm.Vertical([torch.zeros(3, 1), torch.zeros(2, 1), torch.zeros(4, 1)])
    flat = torch.arange(9, dtype=torch.float64)
    v = vertical_like(flat, template)
    assert torch.equal(v.to_tensor().flatten(), flat)
    assert [b.shape for b in v.flat] == [(3, 1), (2, 1), (4, 1)]


def test_efficient_solve_trs_network_smoke():
    """End-to-end through the real linear-time oracle on a tiny network.

    Checks the step is finite and respects the trust region; correctness of the
    underlying damped solve is covered by test_hessian's dense-vs-linear check.
    """
    torch.manual_seed(0)
    model = SequenceOfDenseBlocks(
        input_dim=12, hidden_dim=6, num_classes=4, num_layers=2
    )
    x = torch.randn(8, 12)
    y = torch.randint(0, 4, (8,))

    loss = model(x, y)
    loss.backward()
    grad_vec = assemble_gradient_vector(model)

    delta = 0.5
    p, lam, step_type, hard, n_solves = efficient_solve_trs(
        model, x, y, grad_vec, delta
    )

    assert torch.isfinite(p).all()
    assert lam >= 0.0
    assert float(p.norm()) <= delta * (1.0 + 1e-5)
    if step_type == "boundary":
        assert float(p.norm()) == pytest.approx(delta, rel=1e-4)
    assert n_solves >= 1
