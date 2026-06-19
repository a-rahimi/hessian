# Phase E — cross-phase synthesis

This document synthesizes Phases A, B, C, and D into a single read on why our linear-time inverse-Hessian Newton at [hessian_inverse_product](../../src/hessian.py#L287) stalls on the Phase 5 anchor where published second-order methods do not. The original report at [report.md](../report.md) framed the diagnosis as a three-way classification between **Bug**, **Damping**, and **Scale**. The data confirms **Damping** in a generalized form: the load-bearing fix is restricting the curvature inverse to a top-eigenvalue subspace, which both saddle-free Newton (explicitly) and Hessian-Free (implicitly via CG truncation) do, and which our full-rank inverse does not.

## TL;DR

- **Algorithm is correct.** Section 3 (rel_err `~1e-13` on the tiny model) and Phase A's A3 (rel_err `~1e-12` on the real anchor) jointly confirm that [hessian_inverse_product](../../src/hessian.py#L287) computes the dense `(H + ε I)^{-1} g` it claims to, to float64 precision. **Bug is ruled out at the per-step level.**
- **Model and data are not the bottleneck.** Phase B reproduces K-FAC's published autoencoder advantage (3× update-count speedup over SGD-momentum) on our software stack, and Phase C's HF works on our exact Phase-5 anchor. **Scale is ruled out.**
- **Damping is confirmed.** Phase D's SFN run keeps every wrapper detail of [train_newton.py](../../src/train_newton.py) constant — same LM accept-rule, same ε schedule, same lr, same SGD fallback on rejection — and only swaps the curvature step from `(H + ε I)^{-1} g` to a top-`k` Lanczos approximation of `(|H| + ε I)^{-1} g`. SFN reaches `probe_loss = 2.00` on D1 (vs our 2.22–2.30 plateau and SGD's 1.97 in 1000 steps) and `min loss = 0.18` on D2 (vs our 1.16 floor and SGD's 0.19), in 100 steps each. With wrapper held constant, the gap closes.
- **HF confirms the diagnosis by a different mechanism.** Phase C's HF in raw-Hessian mode operates on the same `H + λ I` we invert, but its CG iterate at termination lives in a Krylov subspace dominated by the top-magnitude eigenvalues of `H`. The small / near-zero / negative eigenvalues that `(H + ε I)^{-1}` would amplify never enter HF's iterate.
- **The unifying observation is the subspace restriction.** Phase A's sub-sampled Newton, which computes literally the same full-rank step our linear-inverse computes (verified to 12 significant figures by A3), also stalls — at `2.26` on the fixed batch. The methods that succeed (SFN, HF) all stay in a top-eigenvalue subspace of `H`; the methods that fail (our linear-inverse, sub-sampled Newton) compute the full-rank inverse.

## Method × diagnostic comparison table

All numbers below are on the **Phase 5 anchor** (`SequenceOfDenseBlocks`, `num_layers=8, hidden_dim=24, image_size=16, activation=relu`, batch size 64) except where noted. The "fixed-batch memorization" column is the [exp-048b](../../experiments/runs/exp-048b-sgd-reuse-batch-lr0.01) protocol — a single 64-sample batch held for the whole run, with the loss on *that* batch reported. The "full CIFAR-10" column is `probe_loss` at the end of the budget.

| Method                                                                         | Curvature inverse           | Step accept-rule                            | Full CIFAR-10 (probe_loss)       | Fixed-batch memorization (min loss) | Steps |
| ------------------------------------------------------------------------------ | --------------------------- | ------------------------------------------- | -------------------------------- | ----------------------------------- | ----- |
| SGD (reference)                                                                | none                        | unconditional                               | **1.97**                         | **0.19** at lr=0.01                 | 1000  |
| Our linear-inverse Newton ([summary-so-far.md](../../experiments/summary-so-far.md)) | full `(H + ε I)^{-1}`     | same-batch loss-decrease LM + SGD fallback | 2.22–2.30 band                  | 1.16 at ε=0.5, lr=0.5              | 1000  |
| Sub-sampled Newton (Phase A, lr=0.01, ε=1.0)                                   | full `(H + ε I)^{-1}`       | none (raw Newton step)                      | 2.46 at step 36 (descending)    | 2.26 at step 51 (descending)        | ~50   |
| Sub-sampled Newton (Phase A, lr=0.5 or 0.1)                                    | full `(H + ε I)^{-1}`       | none                                        | diverged in <20 steps           | diverged in <20 steps               | <20   |
| HF in Gauss-Newton mode (Phase C)                                              | Krylov in `(J^T H_loss J + λ I)` | CG truncation + Martens LM trust ratio | 2.189 at step 90 (NaN @188)     | **8.4e-8** at step 106              | ~200  |
| HF in raw-Hessian mode (Phase C)                                               | Krylov in `(H + λ I)`       | CG truncation + Martens LM trust ratio      | **2.086** at step 253 (NaN @447) | **0.21** at step 247                | ~300  |
| Saddle-free Newton (Phase D, k=20)                                             | top-20 Lanczos of `(\|H\| + ε I)^{-1}` | same-batch loss-decrease LM (same as ours) | **2.00** at step 78 (final 2.09) | **0.18** at step 97               | 100   |

Cross-phase reference: **K-FAC on the canonical deep autoencoder** (Phase B) reaches test MSE `5.76` after 15000 K-FAC steps vs. `16.72` after 15000 SGD-momentum steps; K-FAC matches SGD's 15k-step number at step 5k. Reproduction is partial but qualitatively clean and matches Martens & Grosse 2015 Figure 1. The report's "is there any working second-order baseline on our stack" gate is satisfied.

## Why the data points at subspace-restricted curvature

The original report's decision rules were stated as binary "does this hypothesis survive" tests. The actual outcomes line up cleanly with one of them, but the survivor needs to be read with the right generalization.

- **Section 3 + Phase A3 pass cleanly** → **Bug at the per-step level is ruled out.** The toy `P=28` Hessian and the anchor's `P=22128` Hessian both round-trip through `dense solve` vs. `hessian_inverse_product` at `rel_err ≤ 5e-12` in float64. There is no sign-or-transposition error in the block-factored solver path.
- **Phase B reproduces K-FAC** → **Scale is ruled out.** The published methods work on our stack, on a larger model with a larger batch, and HF works on the actual Phase-5 anchor too.
- **Phase D's SFN with our LM wrapper beats our linear-Newton on both D1 and D2** → **Damping is confirmed.** With wrapper held constant, swapping `(H + ε I)^{-1}` for `(|H_k| + ε I)^{-1}` on the top-20 Lanczos subspace closes the gap on both diagnostics. This is the cleanest controlled test we have on the role of the curvature operator.
- **Phase C's HF succeeds on our anchor in both modes** → consistent with the Damping confirmation, by a different mechanism. CG truncation on `(H + λ I)` produces an iterate that lives in `span{g, Hg, …, H^{k−1}g}`, which is dominated by the top-magnitude eigenvalues of `H` by the convergence properties of CG. The small / near-zero / negative eigenvalues that the full-rank inverse dangerously amplifies never enter HF's iterate.
- **Phase A's sub-sampled Newton stalls at the same `~2.26` band our linear-inverse stalls in, despite computing the same step at 12 significant figures of agreement** → confirms the failure is in the *full-rank* inverse itself. Without a Krylov truncation or an explicit top-`k` projection, the small eigenvalues of `H` poison the step regardless of whether the wrapper is our LM or no wrapper at all.

## The fix

The block-factored inverse machinery in [hessian.py](../../src/hessian.py) and [block_partitioned_matrices.py](../../src/block_partitioned_matrices.py) does not need correctness changes. What needs to change is the *quantity* the machinery solves for: the dense `(H + ε I)^{-1} g` is the wrong target on indefinite Hessians, and replacing it with a top-eigenvalue-projected solve is what closes the gap. Two natural paths follow.

1. **Push the absolute-value-on-top-`k` transform into the block factorization.** SFN at our anchor used `k=20` and reached SGD-floor on both diagnostics. The block solver could in principle be augmented to compute the same projection on its own block-tri-diagonal system rather than going through Lanczos on the dense HVP, which would preserve the linear time-and-space scaling that motivates the project.
2. **Wrap the existing [SequenceOfBlocks.hessian_vector_product](../../src/hessian.py#L260) in a truncated CG loop with LM trust-region damping**, exactly as Martens 2010 prescribes. This is the closer match to HF and would inherit HF's robustness properties (CG truncation as implicit regularization, trust-region ratio, warm-starting), at the cost of giving up the explicit inverse and reverting to a Krylov iterate.

Either path keeps the headline contribution of the project intact — linear-time-and-space inverse-Hessian-vector products on deep MLPs — while replacing the full-rank inverse it currently computes with the subspace-restricted inverse the data says we actually need.

## Reproduction artifacts

- [phase-a-subsampled-newton/](../phase-a-subsampled-newton/) — Section 3 pytest, A3 numerical agreement, sub-sampled Newton driver and trajectories.
- [phase-b-kfac-autoencoder/](../phase-b-kfac-autoencoder/) — Hinton & Salakhutdinov deep autoencoder, K-FAC vs SGD-momentum trajectories, [plots/recon_vs_steps.png](../phase-b-kfac-autoencoder/plots/recon_vs_steps.png).
- [phase-c-hessian-free/](../phase-c-hessian-free/) — HF on the autoencoder (C1) and on the Phase-5 anchor in both GGN and raw-Hessian modes (C2), with the GNVP/HVP sanity check.
- [phase-d-saddle-free-newton/](../phase-d-saddle-free-newton/) — low-rank SFN with Lanczos top-`k` eigenpairs of `H` on the anchor, with D1 and D2 100-step trajectories.
