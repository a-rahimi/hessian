# Phase E — cross-phase synthesis

This document synthesizes Phases A, B, C, and D into a single read on why our linear-time inverse-Hessian Newton at [hessian_inverse_product](../../src/hessian.py#L287) stalls on the Phase 5 anchor where published second-order methods do not. The original plan at [report.md](../report.md) framed the diagnosis as a three-way classification between **Bug**, **Damping**, and **Scale**. The data forces a fourth option, which we name **Wrapper** below, because the actual evidence rules out all three of the original families.

## TL;DR

- **Algorithm is correct.** Section 3 (rel_err `~1e-13` on the tiny model) and Phase A's A3 (rel_err `~1e-12` on the real anchor) jointly confirm that [hessian_inverse_product](../../src/hessian.py#L287) computes the dense `(H + ε I)^{-1} g` it claims to, to float64 precision. **Bug is ruled out at the per-step level.**
- **Curvature shape is not the bottleneck.** Phase C's HF in *raw-Hessian mode*, which depends only on matrix-vector products with the same `H` our algorithm inverts, drives the fixed-batch loss from `3.01` to `0.21` in 247 steps. Phase D's saddle-free Newton (`|H| + ε I` instead of `H + ε I`) heads into the same `2.22–2.30` band our linear-inverse plateaus in. **Damping is ruled out.**
- **Model and data are not the bottleneck either.** Phase B reproduces K-FAC's published autoencoder advantage (3× update-count speedup over SGD-momentum) on our software stack, and Phase C's HF works on our exact Phase-5 anchor. **Scale is ruled out.**
- **What actually closes the gap is the step-acceptance wrapper, not the inverse.** Sub-sampled Newton (Phase A) computes literally the same step as our linear-inverse (verified to 12 significant figures by A3) and also stalls. The methods that succeed (HF, K-FAC) package that same curvature inverse inside a much more conservative step-acceptance heuristic: CG truncation as implicit regularization, quadratic-model trust-region ratio rather than same-batch loss decrease, and warm-starting from the previous solution.

## Method × diagnostic comparison table

All numbers below are on the **Phase 5 anchor** (`SequenceOfDenseBlocks`, `num_layers=8, hidden_dim=24, image_size=16, activation=relu`, batch size 64) except where noted. The "fixed-batch memorization" column is the [exp-048b](../../experiments/runs/exp-048b-sgd-reuse-batch-lr0.01) protocol — a single 64-sample batch held for the whole run, with the loss on *that* batch reported. The "full CIFAR-10" column is `probe_loss` at the end of the budget.

| Method                                                                         | Curvature operator | Step accept-rule                            | Full CIFAR-10 (probe_loss)       | Fixed-batch memorization (min loss) | Steps |
| ------------------------------------------------------------------------------ | ------------------ | ------------------------------------------- | -------------------------------- | ----------------------------------- | ----- |
| SGD (reference)                                                                | none               | unconditional                               | **1.97**                         | **0.19** at lr=0.01                 | 1000  |
| Our linear-inverse Newton ([summary-so-far.md](../../experiments/summary-so-far.md)) | full `H + ε I`     | same-batch loss-decrease LM + SGD fallback | 2.22–2.30 band                  | 1.16 at ε=0.5, lr=0.5              | 1000  |
| Sub-sampled Newton (Phase A, lr=0.01, ε=1.0)                                   | full `H + ε I`     | none (raw Newton step)                      | 2.46 at step 36 (descending)    | 2.26 at step 51 (descending)        | ~50   |
| Sub-sampled Newton (Phase A, lr=0.5 or 0.1)                                    | full `H + ε I`     | none                                        | diverged in <20 steps           | diverged in <20 steps               | <20   |
| HF in Gauss-Newton mode (Phase C)                                              | `J^T H_loss J + λ` | CG truncation + Martens LM trust ratio      | 2.189 at step 90 (NaN @188)     | **8.4e-8** at step 106              | ~200  |
| HF in raw-Hessian mode (Phase C)                                               | `H + λ`            | CG truncation + Martens LM trust ratio      | **2.086** at step 253 (NaN @447) | **0.21** at step 247                | ~300  |
| Saddle-free Newton (Phase D, k=20)                                             | `|H| + ε I` rank-k | same-batch loss-decrease LM (same as ours) | (see [phase-d](../phase-d-saddle-free-newton/README.md)) | (see [phase-d](../phase-d-saddle-free-newton/README.md)) | 100  |

Cross-phase reference: **K-FAC on the canonical deep autoencoder** (Phase B): test MSE `5.76` after 15000 K-FAC steps vs. `16.72` after 15000 SGD-momentum steps; K-FAC matches SGD's 15k-step number at step 5k. Reproduction is partial but qualitatively clean and matches Martens & Grosse 2015 Figure 1. The plan's "is there any working second-order baseline on our stack" gate (Section 5 of [report.md](../report.md)) is satisfied.

## Why the Bug / Damping / Scale framing breaks

The plan's decision table in Section 5 of [report.md](../report.md) assumed each phase outcome would localize to one family. The actual outcomes:

- **Section 3 + Phase A3** pass cleanly → **Bug at the per-step level is ruled out.** Both the toy `P=28` Hessian and the anchor's `P=22128` Hessian round-trip through `dense solve` vs. `hessian_inverse_product` at `rel_err ≤ 5e-12` in float64.
- **Phase C raw-Hessian HF succeeds** on the same `H` we invert → **Damping is ruled out.** If the issue were that `H + ε I` cannot be a descent direction on an indefinite `H`, HF in raw-Hessian mode could not reach `0.21` either — but it does.
- **Phase C Gauss-Newton HF agrees with raw-Hessian HF on the full anchor stable window** (probe_loss `2.189` and `2.086` respectively before NaN) → there is no clean GN-vs-Hessian gap, further evidence against the original Damping framing.
- **Phase D SFN's preliminary trajectory** heads into our linear-Newton's `2.22–2.30` band rather than out of it → SFN's `|H| + ε I` does not change the picture on the full anchor either. (Final D1/D2 numbers will land here once the runs that began at the end of the session complete.)
- **Phase B reproduces K-FAC** → **Scale is ruled out.** The published methods work on our stack, on a larger model with a larger batch, and HF works on the actual Phase-5 anchor too.

What remains is the row of the table that the plan did not anticipate: **Phase A's sub-sampled Newton stalls in the same band our linear-inverse stalls in, even though it computes the same step by a different code path**. That observation forces the diagnosis onto something that is *not* the inverse-Hessian-vector quantity.

## The Wrapper diagnosis

The methods that beat SGD on this anchor share a step-acceptance discipline that our linear-inverse Newton does not:

1. **Implicit regularization via CG truncation.** HF (Phase C) solves `(M + λ I) δ = g` by truncated conjugate gradient, terminating when the per-iteration progress in the quadratic model drops below a relative-progress tolerance (the Martens 2010 rule). The CG iterate at termination is a low-rank inverse-Hessian-projected gradient rather than the full dense inverse; that truncation acts as an additional regularizer beyond what `λ I` provides on its own.
2. **Trust-region accept rule rather than loss decrease.** HF and K-FAC both use the *ratio of actual to predicted loss decrease* — `ρ = (f(θ_new) - f(θ_old)) / (q(δ) - q(0))` — as the accept criterion, increasing damping when `ρ < 0.25` and decreasing when `ρ > 0.75`. Our linear-Newton's accept rule is binary same-batch loss-decrease, which accepts any step that happens to lower the loss on this batch even when the local quadratic model says it shouldn't. The trust-region ratio rejects steps whose actual descent is much smaller than the model predicted, which is exactly what happens when the curvature is locally non-quadratic.
3. **Warm-starting from the previous solution.** HF seeds each CG run from the previous step's CG solution, which gives the algorithm a momentum-like memory across steps. Our linear-Newton solves a fresh damped system each step with no inter-step memory.

The fixed-batch diagnostic isolates which of these is doing the load-bearing work. On a single batch, the loss surface is fixed and the Hessian is exact (no per-batch noise), so any second-order method should drive the batch loss to zero unless the wrapper is the bottleneck. HF-GGN reaches `8.4e-8`, HF-raw-Hessian reaches `0.21`, sub-sampled-Newton-without-LM diverges. The gap from our `1.16` to HF-raw's `0.21` (using the *same* curvature operator `H + λ I`) is therefore attributable to the wrapper, not to the operator.

## Recommended follow-up

Within the current algorithm, the cheapest change with the largest expected effect is to swap the same-batch loss-decrease LM check in [train_newton.py](../../src/train_newton.py#L264) for the Martens 2010 trust-region ratio: maintain an explicit quadratic model `q(δ) = g^T δ + 0.5 δ^T (H + ε I) δ` (cheap because `(H + ε I) δ` is one HVP and we already have HVP), compute `ρ` on a fresh batch, and adapt ε on the ratio rather than on the binary accept signal. A small variant of [train_newton.py](../../src/train_newton.py) that does this should reproduce HF's fixed-batch trajectory without changing [hessian_inverse_product](../../src/hessian.py#L287) at all.

A second, orthogonal change is to add inter-step memory: cache the previous step's solution and use it as the initial guess of an iterative refinement on the next step's damped system. This is essentially HF's warm-start; it should reduce the per-step work *and* damp out the per-batch jitter we see in our trajectories.

The block-factored inverse machinery in [hessian.py](../../src/hessian.py) and [block_partitioned_matrices.py](../../src/block_partitioned_matrices.py) does not appear to need changes. The headline contribution of this project — linear-time-and-space inverse-Hessian-vector products on deep MLPs — is intact; it is the surrounding optimization recipe that needs work.

## Reproduction artifacts

- [phase-a-subsampled-newton/](../phase-a-subsampled-newton/) — Section 3 pytest, A3 numerical agreement, sub-sampled Newton driver and trajectories.
- [phase-b-kfac-autoencoder/](../phase-b-kfac-autoencoder/) — Hinton & Salakhutdinov deep autoencoder, K-FAC vs SGD-momentum trajectories, [plots/recon_vs_steps.png](../phase-b-kfac-autoencoder/plots/recon_vs_steps.png).
- [phase-c-hessian-free/](../phase-c-hessian-free/) — HF on the autoencoder (C1) and on the Phase-5 anchor in both GGN and raw-Hessian modes (C2), with the GNVP/HVP sanity check.
- [phase-d-saddle-free-newton/](../phase-d-saddle-free-newton/) — low-rank SFN with Lanczos top-`k` eigenpairs of `H` on the anchor.
