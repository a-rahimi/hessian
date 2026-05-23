# Plan: reproduce published second-order optimization results

## 1. The three hypotheses

Our Newton recipe, built on the linear-time inverse-Hessian algorithm at [hessian_inverse_product](../src/hessian.py#L287), fails to match SGD on the Phase 5 anchor (`SequenceOfDenseBlocks`, `num_layers=8, hidden_dim=24, image_size=16, activation=relu`). On full CIFAR-10, trailing-average training loss plateaus in the 2.22–2.30 band while SGD reaches `probe_loss=1.97`. On the fixed-batch memorization diagnostic, Newton bottoms at `min loss=1.16` while SGD reaches `0.19`. [summary-so-far.md](../experiments/summary-so-far.md) records the sweeps over ε, lr, batch size, LM accept-check, and step-norm clipping that left the plateau unchanged.

Three hypotheses remain on the table, and hyperparameter sweeps cannot tell them apart. The rest of this document refers to them by the short names defined here.

- **Bug.** The augmented block-tri-diagonal system in [hessian_inverse_product](../src/hessian.py#L287) and the block-partitioned-matrix machinery in [block_partitioned_matrices.py](../src/block_partitioned_matrices.py) are intricate. A sign error, a transposition, or a wrong damping placement could silently produce a slow descent direction instead of the intended Newton step.
- **Damping.** The algorithm computes what it claims to, but `(H + ε I)^{-1} g` is the wrong quantity. When `H` is indefinite, adding `ε I` keeps the small negative eigenvalues near zero rather than pushing them positive, so the result is not even a descent direction. Published recipes avoid this by preconditioning with the Gauss-Newton matrix `J^T H_loss J` (PSD by construction) or by applying the absolute-value transform `|H| + ε I` (saddle-free Newton).
- **Scale.** The algorithm and the recipe are both fine, but the anchor (~5–10k parameters, 64-sample batches) is too small for any per-batch curvature estimate to be useful, regardless of which second-order method consumes it.

The plan below runs published second-order methods on their own benchmarks first and on our anchor second, so each combination of outcomes points at one of **Bug**, **Damping**, or **Scale**.

## 2. Candidate baselines from the literature

The user-imposed constraint on this plan is to consider only methods that use the full Hessian, or curvature approximations *richer than the diagonal*. Diagonal-Hessian estimators (AdaHessian, Sophia, Hutchinson-diagonal preconditioners) are out of scope because they do not actually test whether off-diagonal curvature is what our algorithm fails to exploit; they only test whether a different cheap preconditioner happens to outperform SGD. We therefore drop AdaHessian and Sophia, and choose baselines that span the spectrum of "full-Hessian information" methods.

We pick baselines that satisfy three constraints. First, each must use full-matrix or block-structured curvature richer than diagonal. Second, each must have a working open-source PyTorch implementation or be straightforward to build on top of our existing [hessian_vector_product](../src/hessian.py#L260) (a clean HVP is already in the repo, which gives every Krylov-style method for free). Third, the baselines must collectively discriminate between **Bug**, **Damping**, and **Scale**.

### 2.1 Hessian-Free (Martens, 2010) — closest mathematically to our algorithm

Hessian-Free is conjugate gradient on the Gauss-Newton matrix `G = J^T H_loss J` (or optionally the raw Hessian) with Levenberg-Marquardt damping, an explicit CG truncation rule, and warm-starting of CG from the previous solution. The original paper ([Deep_HessianFree.pdf](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)) reports beating Hinton & Salakhutdinov 2006 on the deep MNIST and CURVES autoencoder benchmarks, and the method depends only on a Hessian-vector or Gauss-Newton-vector product, which we already have via [SequenceOfBlocks.hessian_vector_product](../src/hessian.py#L260). A maintained PyTorch reference implementation exists at [ltatzel/PyTorchHessianFree](https://github.com/ltatzel/PyTorchHessianFree), so we do not have to write CG and LM damping from scratch.

Why include it: HF is the published method most directly comparable to ours. Both methods invert a damped curvature matrix in a way that avoids materializing it; we do so by block factorization, while HF does so by Krylov iteration. If HF works on our model and our linear-inverse algorithm does not, then comparing the two iterates on the *same* gradient and damping isolates whether the discrepancy is **Bug** (the inversion path) or **Damping** (Gauss-Newton versus raw Hessian). If neither works, **Scale** takes over as the leading explanation.

### 2.2 K-FAC (Martens & Grosse, 2015) — Kronecker-factored full block per layer

K-FAC approximates each layer's Fisher block as a Kronecker product `A ⊗ G`, where `A` is the input-activation covariance and `G` is the pre-activation gradient covariance, so the per-layer preconditioner is full within the block (not diagonal) but inverts in time linear in width. The canonical benchmark is the symmetric deep autoencoder on MNIST with encoder sizes `[1000, 500, 250, 30]` (Hinton & Salakhutdinov 2006, replicated by Martens 2010 and Martens & Grosse 2015). The reference implementation script is [autoencoder_mnist.py](https://github.com/tensorflow/kfac/blob/master/kfac/examples/autoencoder_mnist.py) in the official TensorFlow K-FAC repository, and PyTorch ports exist at [Thrandis/kfac-pytorch](https://github.com/Thrandis/kfac-pytorch) and inside [asdfghjkl](https://github.com/kazukiosawa/asdl).

Why include it: K-FAC was designed for exactly the model class we are using (deep feed-forward chains with element-wise nonlinearities), so reproducing its published autoencoder numbers tells us whether *any* second-order method works on a deep MLP under our software stack. If a fresh checkout of a reference implementation does not reach the published autoencoder reconstruction error, we have no working second-order baseline at all and the rest of the plan is on hold.

### 2.3 Saddle-Free Newton (Dauphin et al., 2014) — direct test of family 2

Saddle-Free Newton (SFN) replaces the Newton step `(H + ε I)^{-1} g` with `(|H| + ε I)^{-1} g`, where `|H|` is the matrix obtained by replacing each eigenvalue of `H` with its absolute value. This makes saddle points repulsive rather than attractive, and is the canonical fix for the family-2 failure mode we hypothesize. The original paper ([arXiv 1406.2572](https://arxiv.org/abs/1406.2572)) uses Krylov iteration to compute a low-rank eigendecomposition of `H` and applies the absolute-value transform on that subspace. The more recent low-rank SFN paper ([arXiv 2002.02881](https://arxiv.org/abs/2002.02881)) gives a scalable implementation that only requires Hessian-vector products and explicitly targets the deep-learning regime; an additional HVP-only variant lives in the openreview manuscript [Series of Hessian-Vector Products for Tractable Saddle-Free Newton](https://openreview.net/pdf/d1b7bb20a106d5e9a94ace29f3baaf1c8ee43c90.pdf).

Why include it: SFN is the only baseline that directly tests **Damping**, that is, whether our Newton plateau is caused by `H + ε I` failing to produce a descent direction when `H` has negative eigenvalues. If SFN works on our anchor and HF does not, the diagnosis is unambiguously **Damping**. If neither works, **Bug** and **Scale** are still in play.

### 2.4 Sub-sampled Newton (Roosta-Khorasani & Mahoney, 2016) — explicit full-Hessian baseline

Sub-sampled Newton ([arXiv 1601.04737](https://arxiv.org/abs/1601.04737); follow-up GPU-accelerated version [arXiv 1802.09113](https://arxiv.org/abs/1802.09113)) explicitly *forms and inverts* a full Hessian estimated on a random subset of the training data, rather than using HVPs or block approximations. For our small anchor model (~5–10k parameters), the full per-batch Hessian is dense `P × P` floats and fits in memory, so a direct sub-sampled Newton step is actually a realistic baseline at this scale, even though it does not scale to large models. The CSIE/NTU group's earlier work [Subsampled Hessian Newton for Supervised Learning](https://www.csie.ntu.edu.tw/~cjlin/papers/sub_hessian/sample_hessian.pdf) gives a concrete recipe with reference code.

Why include it: this is the "ground truth" baseline for our own method. Our algorithm claims to compute `(H + ε I)^{-1} g` in linear time on the network's depth; sub-sampled Newton on the *same* batch computes the same quantity via dense linear algebra. The two should agree to numerical precision when both are run on the same `(H, g, ε)`. This subsumes the Section-3 sanity check at a larger scale and gives us a second, independent way to detect a bug in [hessian_inverse_product](../src/hessian.py#L287).

### 2.5 Shampoo (Gupta, Koren & Singer, 2018) — Kronecker per-tensor reference point

Shampoo ([arXiv 1802.09568](https://arxiv.org/abs/1802.09568)) maintains a per-mode preconditioner for each tensor-shaped parameter, structured as a Kronecker product of full-matrix factors per tensor dimension. Like K-FAC the preconditioner is full within each Kronecker factor (richer than diagonal); unlike K-FAC the factor structure is derived from the tensor shape rather than the layer's activation and gradient covariances. PyTorch implementations exist in [google-research/google-research](https://github.com/google-research/google-research/tree/master/scalable_shampoo) and in the [optimi/distributed-shampoo](https://github.com/facebookresearch/optimizers) Meta release. Recent benchmarks ([SOAP](https://lucasjanson.fas.harvard.edu/papers/SOAP_Improving_And_Stabilizing_Shampoo_Using_Adam-Vyas_ea-2024.pdf)) suggest Shampoo outperforms diagonal-second-moment optimizers on practical workloads.

Why include it: Shampoo gives us a second Kronecker-style baseline beyond K-FAC, with a different derivation. If both K-FAC and Shampoo work on our anchor and our raw-Hessian Newton does not, the issue is plausibly that any reasonable factored full-curvature method works while the raw Hessian without saddle-aware damping does not. Shampoo is the lowest-priority of the four baselines in this plan, because its closest comparison is to K-FAC rather than to our own method.

### 2.6 Methods we are deliberately not including

AdaHessian (Yao et al. 2021) and Sophia (Liu et al. 2023) are diagonal-Hessian preconditioners and are excluded by the user-imposed constraint that we only consider methods richer than diagonal. L-BFGS is a low-rank approximation to the inverse Hessian built from past gradient differences rather than from any actual Hessian or Hessian-vector product, so it tests "low-rank inverse curvature from gradient history" rather than "full Hessian"; it can be added later as a sanity check on what a non-Hessian quasi-Newton method achieves, but it is not part of the main plan. Cubic-regularized Newton ([Tripuraneni et al. 2018, arXiv 1711.02838](https://arxiv.org/abs/1711.02838)) is conceptually adjacent to SFN, because both target saddle escape and both depend only on HVPs, but it has no clean reference PyTorch implementation we trust, so we treat it as a follow-up to SFN if **Damping** turns out to be the relevant axis.

## 3. Sanity check before any baseline — verify the algorithm against autograd

This step must come before the rest of the plan, because every later experiment depends on knowing whether our algorithm computes what it claims to. The check is to materialize the full Hessian via PyTorch autograd on a sufficiently small problem, invert it densely, and compare the result against [hessian_inverse_product](../src/hessian.py#L287).

Procedure:

1. Configure a tiny instance of `SequenceOfDenseBlocks` (e.g. `image_size=4, hidden_dim=4, num_layers=2`, batch size 8) so the parameter count `P` stays under a few hundred and the dense Hessian `H ∈ R^{P×P}` fits in memory.
2. Build `H` densely with `torch.autograd.functional.hessian` on the cross-entropy loss with respect to a flat parameter vector.
3. Pick a random vector `g` and compare `torch.linalg.solve(H + ε I, g)` against `model.hessian_inverse_product(x, y, g, ε)` across several values of ε in `{1e-3, 1e-1, 1.0, 10.0}`.
4. Report the relative error `||x_dense - x_ours|| / ||x_dense||` per ε.

Decision rule: if the relative error is `< 1e-4` for all ε tested, the algorithm is correct on tiny problems and **Bug** is unlikely on small models, so we proceed to Phase A. If the error is large for any ε, fix the bug first and re-run this check before spending compute on baselines. The Section 2.4 sub-sampled Newton baseline extends this sanity check from "tiny model + dense `torch.linalg.solve`" to "the Phase 5 anchor model + dense `torch.linalg.solve` on a 64-sample batch", because the anchor's `P` is small enough that a dense `P × P` Hessian still fits.

Per [CLAUDE.md](../CLAUDE.md), there is a test suite under [src/tests/](../src/tests); the new sanity check should land there as a pytest so we keep it green after future edits.

## 4. Reproduction plan

The plan is ordered to maximize signal per unit of compute and engineering effort, and each phase has an explicit decision rule that determines whether the next phase changes target.

### Phase A — Sub-sampled Newton on our anchor (cheapest decisive signal, ~1 day)

Implement a one-file sub-sampled Newton driver that materializes the full per-batch Hessian via `torch.autograd.functional.hessian` on the cross-entropy loss of `SequenceOfDenseBlocks`, adds `ε I`, calls `torch.linalg.solve`, and applies the step. The anchor has `P ≈ 6k` parameters, so the dense Hessian is about `144 MB` in float32 and runs in seconds per step. Three configurations:

- **A1**: sub-sampled Newton on the Phase 5 anchor, same `--num-steps` and `--batch-size=64` as our linear-inverse Newton, on full CIFAR-10. Compare trailing-average training loss and probe loss against [summary-so-far.md](../experiments/summary-so-far.md)'s Newton runs head-to-head.
- **A2**: sub-sampled Newton on the fixed-batch memorization diagnostic (same protocol as [exp-048b](../experiments/runs/exp-048b-sgd-reuse-batch-lr0.01)). The single 64-sample batch is held for 1000 steps.
- **A3**: numerical agreement check. On a held-fixed `(x, y, g, ε)` triple from A2 at step 0, compare the dense-solve step against `model.hessian_inverse_product(x, y, g, ε)` element-wise. Report `||δ_dense - δ_ours|| / ||δ_dense||`.

Decision rule on A3:

- Relative error `< 1e-4` → our linear-inverse algorithm agrees with the dense ground truth on the anchor. **Bug** is strongly down-weighted. Proceed to A1/A2 to interpret the optimization trajectories.
- Relative error large → **Bug** is confirmed on a real-sized problem even though Section 3 passed. Investigate before continuing.

Decision rule on A1:

- Sub-sampled Newton reaches `probe_loss < 1.97` in `≤ 1000` steps → our linear-inverse Newton stalls even though the *same* preconditioner direction works when computed densely. Either A3 caught a discrepancy (back to **Bug**), or the discrepancy is small but the trajectory is sensitive in a way we have not yet measured (a more subtle **Bug**). Proceed to Phase B and Phase C to triangulate.
- Sub-sampled Newton plateaus the way our Newton does → **Damping** and **Scale** are now the leading hypotheses. Proceed to Phase C (HF on the Gauss-Newton matrix) and Phase D (saddle-free Newton) to discriminate.

Decision rule on A2:

- Sub-sampled Newton drives the fixed batch to `loss < 0.1` → optimization on a single batch is possible with a full-Newton step, and our linear-inverse stalling at 1.16 on the same setup is a **Bug** or **Damping** signal.
- Sub-sampled Newton also stalls above 1.0 on the fixed batch → the diagnostic is harder than we thought, likely because the loss surface is genuinely non-quadratic in the relevant region and any single full-Newton step over-shoots even with LM damping. The Newton plateau is then less anomalous.

### Phase B — Reproduce K-FAC on its published benchmark (medium, ~2–3 days)

Implement (or check out) the canonical MNIST deep autoencoder of Hinton & Salakhutdinov 2006: input 784, encoder `[1000, 500, 250, 30]`, mirror decoder, sigmoid activations, cross-entropy reconstruction loss. Train K-FAC against SGD-momentum for `~100k` updates with the K-FAC paper's hyperparameters (batch size 1000, learning rate adapted by the quadratic-model rule, damping adapted by the LM rule). The success bar is the published reconstruction error of approximately `0.96` test error on MNIST within a budget where SGD-momentum is at roughly `2.0`–`2.5`, matching Figure 1 of Martens & Grosse 2015.

PyTorch K-FAC options, in priority order:

- [Thrandis/kfac-pytorch](https://github.com/Thrandis/kfac-pytorch) — small reference impl, MLP-friendly.
- The auto-differentiable variant in [asdfghjkl](https://github.com/kazukiosawa/asdl) — more recent, supports CNNs as well.
- Re-port the [tensorflow/kfac autoencoder script](https://github.com/tensorflow/kfac/blob/master/kfac/examples/autoencoder_mnist.py) directly if neither of the above reproduces cleanly. Reproduction notes are in [Yaroslav Bulatov's writeup](https://yaroslavvb.medium.com/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0).

Decision rule:

- We hit the published autoencoder numbers (test reconstruction within ±20% of `0.96`) → the reference is reproducible on our hardware and we move on with a known-good baseline.
- We cannot reproduce after a reasonable tuning pass → switch to Hessian-Free's autoencoder reproduction (Phase C1) before declaring the K-FAC reference broken on our stack. Do not move to Phase E until at least one reference baseline reproduces.

### Phase C — Hessian-Free on the autoencoder and on our anchor (~3–4 days)

Use the existing [ltatzel/PyTorchHessianFree](https://github.com/ltatzel/PyTorchHessianFree) implementation rather than writing CG and LM damping from scratch. The library exposes a single `HessianFree` PyTorch optimizer that needs a closure returning the model output and the loss; it computes the Gauss-Newton-vector product internally and runs CG with LM damping using the recipe from Martens 2010.

Apply HF to:

- **C1**: the canonical autoencoder from Phase B, to verify HF itself reproduces Martens 2010 numbers (~0.9 train reconstruction in `~50` epochs at batch size 1000). This is the K-FAC fallback reference if Phase B fails.
- **C2**: our `SequenceOfDenseBlocks` CIFAR-10 anchor, same `num-steps` budget as Phase 5. Run HF in both modes: with the Gauss-Newton matrix (the default) and with the raw Hessian (`use_gnm=False` or equivalent). The raw-Hessian HF iterate is the closest published method to our own algorithm and is the direct A/B against our linear-inverse Newton.

Decision rule on C1:

- HF reproduces Martens 2010 autoencoder numbers → HF is a working second-order baseline we trust.
- HF fails to reproduce → debug the GNV-P call against autograd-materialized `J^T H_loss J v` on a tiny problem (same protocol as Section 3). Do not move on until C1 succeeds.

Decision rule on C2:

- HF in Gauss-Newton mode reaches `probe_loss < 1.97` on our CIFAR-10 anchor but raw-Hessian HF does not → **Damping** is confirmed, specifically that swapping the raw Hessian for the Gauss-Newton matrix is the fix. The remedy for our own algorithm is to precondition with `J^T H_loss J` instead of `H`.
- Both HF modes succeed → our linear-inverse Newton is the only thing that fails, which points to **Bug** on our code path. Compare HF's CG iterate to our linear-inverse iterate on the *same* gradient and damping (the Phase A3 A/B extended to a full training trajectory) to localize the discrepancy.
- Both HF modes fail like our Newton → **Scale** is the leading hypothesis. Phase D (saddle-free Newton) becomes the next test; if SFN also fails, re-anchor on a larger model.

### Phase D — Saddle-Free Newton on our anchor (~3 days)

The low-rank SFN paper ([arXiv 2002.02881](https://arxiv.org/abs/2002.02881)) gives an algorithm that only needs Hessian-vector products, which we already have via [SequenceOfBlocks.hessian_vector_product](../src/hessian.py#L260). Implementation steps:

1. Run Lanczos for `k = min(20, P)` iterations on the per-batch Hessian to extract its top-`k` eigenpairs `(λ_i, v_i)`. SciPy's `scipy.sparse.linalg.eigsh` accepts a `LinearOperator` wrapper around our HVP, so this is a small adapter rather than a from-scratch Lanczos implementation.
2. Build the SFN step: split the gradient into the Krylov subspace component `g_K = ∑_i (v_i^T g) v_i` and the orthogonal complement `g_⊥ = g - g_K`. Apply `|λ_i| + ε` rescaling in the subspace and SGD-style damping `1 / ε_⊥` in the complement.
3. Wire this as a third mode in [train_newton.py](../src/train_newton.py) so it shares the LM accept/reject and metric logging machinery with the existing `sgd` and `newton` modes.

Apply SFN to the Phase 5 anchor (D1) and to the fixed-batch memorization diagnostic (D2).

Decision rule:

- SFN clearly beats our raw-Hessian Newton on D1 or D2 → **Damping** is confirmed: the raw Hessian's negative eigenvalues are what `H + ε I` mishandles, and `|H| + ε I` fixes it. The remedy for our own algorithm is to add an absolute-value transform to the block-factored solver (non-trivial but well-defined).
- SFN matches or marginally beats our Newton → **Damping** is partially in play but not the dominant issue.
- SFN performs no better than our Newton → **Damping** is unlikely; **Bug** or **Scale** is dominant. The Phase A3 numerical check plus Phase C2's HF A/B should already have discriminated between them.

### Phase E — Head-to-head on our anchor (cheap once Phases A–D are working)

Run the same `SequenceOfDenseBlocks` Phase 5 anchor under every method that has produced a working baseline by this point: SGD, our linear-inverse Newton, sub-sampled Newton (Phase A), HF in Gauss-Newton mode (Phase C), HF in raw-Hessian mode (Phase C), SFN (Phase D), and optionally K-FAC ported to the MLP (Phase B) and Shampoo. Plot trailing-average training loss and probe loss against both wall-clock and step count. The full comparison is the final read on whether our linear-inverse algorithm is competitive with the published methods on this exact model.

Shampoo is the lowest priority addition and only runs if Phases A through D leave the diagnosis ambiguous, because its closest comparison is to K-FAC rather than to our own method.

## 5. What we expect to learn

Each combination of phase outcomes points at one of **Bug**, **Damping**, or **Scale**:

| Phase outcome                                                                                                  | Implication                                                                                                                  |
| -------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Section 3 sanity check fails, or Phase A3 numerical agreement fails                                            | **Bug**. Fix and restart.                                                                                                    |
| Sub-sampled Newton (A1) succeeds on our anchor, our linear-inverse Newton does not, and A3 agrees numerically  | **Bug**, of a subtle kind whose effect on the trajectory only appears in training rather than in a single-step output check. |
| HF in Gauss-Newton mode (C2) succeeds and HF in raw-Hessian mode (C2) does not                                  | **Damping**. The remedy is to switch our own preconditioner from `H` to `J^T H_loss J`.                                       |
| SFN (D1) succeeds where our Newton fails                                                                        | **Damping**. The remedy is `|H| + ε I` rather than `H + ε I`.                                                                |
| All of sub-sampled Newton (A1), HF (C2), and SFN (D1) fail on our anchor, but K-FAC autoencoder (B) reproduces | **Scale**. The model is too small for stochastic curvature at this batch size. Re-anchor on a larger model.                  |
| K-FAC autoencoder (B) does not reproduce on our stack                                                          | We do not have a working reference baseline. Stop and re-evaluate the plan rather than running Phase E.                      |

Each row above is a question the current sweep-based approach cannot answer, and is the reason for taking the detour through published baselines.

## 6. Out-of-scope

This plan is not about improving the existing linear-Hessian-inverse algorithm or its LM recipe. It is also not about claiming a final "Newton works" or "Newton fails" verdict on the project's own anchor. The deliverable is a set of comparisons with published methods that *localizes* the failure to **Bug**, **Damping**, or **Scale**, so a follow-up plan can target the right one.

## Sources

Primary references:

- Martens, J. *Deep learning via Hessian-free optimization*. ICML 2010. [PDF](https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf). PyTorch reference implementation: [ltatzel/PyTorchHessianFree](https://github.com/ltatzel/PyTorchHessianFree).
- Martens, J. & Grosse, R. *Optimizing neural networks with Kronecker-factored approximate curvature*. ICML 2015. Reference code: [tensorflow/kfac](https://github.com/tensorflow/kfac), specifically [autoencoder_mnist.py](https://github.com/tensorflow/kfac/blob/master/kfac/examples/autoencoder_mnist.py). PyTorch ports: [Thrandis/kfac-pytorch](https://github.com/Thrandis/kfac-pytorch), [asdfghjkl](https://github.com/kazukiosawa/asdl). PyTorch reproduction notes: [Bulatov medium post](https://yaroslavvb.medium.com/optimizing-deeper-networks-with-kfac-in-pytorch-4004adcba1b0).
- Dauphin, Y. et al. *Identifying and attacking the saddle point problem in high-dimensional non-convex optimization*. NeurIPS 2014. [arXiv 1406.2572](https://arxiv.org/abs/1406.2572). Scalable low-rank follow-up: [arXiv 2002.02881](https://arxiv.org/abs/2002.02881). HVP-only variant: [Series of Hessian-Vector Products for Tractable Saddle-Free Newton](https://openreview.net/pdf/d1b7bb20a106d5e9a94ace29f3baaf1c8ee43c90.pdf).
- Roosta-Khorasani, F. & Mahoney, M. *Sub-sampled Newton methods I: Globally convergent algorithms*. [arXiv 1601.04737](https://arxiv.org/abs/1601.04737). GPU-accelerated follow-up: [arXiv 1802.09113](https://arxiv.org/abs/1802.09113). Concrete recipe and code: [Subsampled Hessian Newton for Supervised Learning, NTU](https://www.csie.ntu.edu.tw/~cjlin/papers/sub_hessian/sample_hessian.pdf).
- Gupta, V., Koren, T. & Singer, Y. *Shampoo: Preconditioned stochastic tensor optimization*. ICML 2018. [arXiv 1802.09568](https://arxiv.org/abs/1802.09568). Implementations: [google-research/scalable_shampoo](https://github.com/google-research/google-research/tree/master/scalable_shampoo), [facebookresearch/optimizers (distributed Shampoo)](https://github.com/facebookresearch/optimizers). Recent benchmark with stabilization: [SOAP](https://lucasjanson.fas.harvard.edu/papers/SOAP_Improving_And_Stabilizing_Shampoo_Using_Adam-Vyas_ea-2024.pdf).

Adjacent methods, mentioned but deliberately not in the active plan:

- Tripuraneni, N. et al. *Stochastic cubic regularization for fast nonconvex optimization*. NeurIPS 2018. [arXiv 1711.02838](https://arxiv.org/abs/1711.02838). Same family as SFN; revisit only if SFN turns out to be the load-bearing baseline.
- Yao, Z. et al. *AdaHessian: An adaptive second order optimizer*. AAAI 2021. Diagonal-Hessian, excluded by Section 2.6.
- Liu, H. et al. *Sophia: A scalable stochastic second-order optimizer for language model pre-training*. arXiv 2305.14342. Diagonal-Hessian, excluded by Section 2.6.

Survey index for further candidates: [riverstone496/awesome-second-order-optimization](https://github.com/riverstone496/awesome-second-order-optimization).
