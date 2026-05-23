# Section 3 sanity check — result

The Section 3 procedure in [plan.md](../plan.md) is implemented as the pytest [test_section3_sanity_check_dense_vs_linear_inverse](../../src/tests/test_hessian.py) inside [src/tests/test_hessian.py](../../src/tests/test_hessian.py). The test passes.

## Setup

The test builds a tiny `SequenceOfDenseBlocks` with `image_size=4, hidden_dim=4, num_layers=2, num_classes=3`, batch size 8, total parameter count `P = 28`. It materializes the full dense cross-entropy Hessian `H ∈ R^{28×28}` via `torch.func.hessian` on a flat-parameter loss, draws a random per-layer-block gradient `g`, and for each ε computes both the dense solve `torch.linalg.solve(H + ε I, g)` and the linear-time block solve `model.hessian_inverse_product(x, y, g, ε)`, then reports `||x_dense - x_ours|| / ||x_dense||`.

The test runs in float64 because `ε = 1e-3` puts `H + ε I` close to singular relative to float32's `~1e-7` precision. At float32 the same comparison gives `rel_err ≈ 4e-4` for `ε = 1e-3` (still small in absolute terms, but above the `< 1e-4` bar), while at float64 the same configuration drops to `5e-13`. The discrepancy lives in `torch.linalg.solve` and in the block solver rounding, not in the algorithm. Other ε pass at float32 by orders of magnitude.

## Per-ε relative errors (float64)

| ε     | `||x_dense - x_ours|| / ||x_dense||` |
| ----- | ------------------------------------ |
| 1e-3  | 5.246e-13                            |
| 1e-1  | 3.471e-15                            |
| 1.0   | 1.953e-16                            |
| 10.0  | 1.875e-16                            |

All values are at or near `float64` machine epsilon (`~2e-16`) scaled by the condition number of `H + ε I`. The 1e-3 row is the worst because the matrix is most ill-conditioned there, and the result is still 9 orders of magnitude below the `1e-4` bar.

## Decision rule

Per Section 3 of [plan.md](../plan.md), the decision rule is "if the relative error is `< 1e-4` for all ε tested, the algorithm is correct on tiny problems and **Bug** is unlikely on small models, so we proceed to Phase A." The check passes. We proceed to Phase A.

## Reproducing

```
cd /Users/arahimi/hessian && source .venv/bin/activate
cd src && pytest tests/test_hessian.py::test_section3_sanity_check_dense_vs_linear_inverse -xvs
```
