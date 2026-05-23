# A3 — Numerical agreement of dense and linear-inverse Newton steps

## What this measures

A3 lifts the Section 3 sanity check from the 28-parameter toy model up to the Phase 5 anchor (`P = 22128`). It computes `delta_dense = (H + eps I)^{-1} g` on the first CIFAR-10 batch by materializing the full dense Hessian via `torch.func.hessian` and calling `torch.linalg.solve`, then compares it to `delta_ours = model.hessian_inverse_product(x, y, g, eps)` from [hessian.py](../../src/hessian.py#L287). If our linear-inverse code has a bug, it should show up here as a non-trivial relative error on a real-sized batch.

The check runs in float64 because the anchor's first-layer Hessian block is `18432 × 18432` and `eps = 0.1` is small enough that float32 round-off contaminates the relative error.

## Configuration

- `--num-layers 8 --hidden-dim 24 --image-size 16 --activation relu` (Phase 5 anchor).
- `--batch-size 64`, first batch of the CIFAR-10 loader, seed 0.
- Total parameter count `P = 22128`.
- Dense Hessian via `torch.func.hessian` on cross-entropy loss; symmetrized.

## Numerical agreement

| ε   | `|delta_dense|` | `|delta_dense - delta_ours|` | rel_err   |
| --- | --------------- | ----------------------------- | --------- |
| 1.0 | 4.955e+00       | 5.033e-12                     | 1.016e-12 |
| 0.1 | 1.498e+02       | 6.689e-10                     | 4.466e-12 |

Both relative errors are at float64 machine precision scaled by the conditioning of `H + eps I`. The 1.498e+02 step at `eps = 0.1` is large because the un-damped Hessian is genuinely nearly singular for this initialization, which is exactly the regime where `H + eps I` is hard to invert; the linear-inverse algorithm still nails it to 11 significant figures.

## Decision rule

Per Section 5 of [plan.md](../plan.md), A3 with `rel_err < 1e-4` "strongly down-weights **Bug**" on the actual model used for the failing experiments. We saw `rel_err ~ 1e-12`, which is 8 orders of magnitude below the bar. The linear-inverse algorithm at [hessian_inverse_product](../../src/hessian.py#L287) computes the same quantity as the dense solver on the anchor. If A1 or A2 still shows a gap between linear-inverse Newton and sub-sampled Newton, the discrepancy must come from somewhere other than the per-step inverse-Hessian-vector quantity.

## Reproducing

```
python reproduce-published-results/phase-a-subsampled-newton/a3_numerical_agreement.py --float64
```

Float64 is recommended; at float32 the smaller-eps relative error rises to ~1e-3 simply because the dense `solve` itself loses precision when `H + eps I` is near-singular, not because the linear-inverse algorithm disagrees.
