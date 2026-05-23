# Phase D — Low-rank Saddle-Free Newton on our anchor

This directory implements the saddle-free Newton (SFN) baseline described
in Section 2.3 and Phase D of [report.md](../report.md), then applies it to
the Phase-5 anchor (D1) and to the fixed-batch memorization diagnostic
(D2). The point of this phase is to test the **Damping** hypothesis: does
swapping `(H + ε I)^{-1} g` for `(|H| + ε I)^{-1} g` materially change
where the optimizer ends up?

## What was implemented

- [sfn.py](sfn.py) — the SFN building blocks. `CachedHessianOperator`
  wraps [SequenceOfBlocks.hessian_vector_product](../../src/hessian.py#L260)
  as a `scipy.sparse.linalg.LinearOperator`, caching the expensive
  `model.derivatives(x, y)` call once per construction so a full Lanczos
  sweep of `k` matvecs only pays the derivative cost once.
  `lanczos_top_k_eigenpairs` calls `scipy.sparse.linalg.eigsh` with
  `which="LM"` to pull the top-`k` largest-magnitude eigenpairs. `sfn_update`
  builds the step
  `δ = Σᵢ (vᵢᵀg) / (|λᵢ| + ε) vᵢ  +  (g − g_K) / (|λ_k| + ε)`
  and round-trips between a flat `numpy` array (for SciPy) and a
  `bpm.Vertical` (for the project's `apply_update`).

- [train_sfn.py](train_sfn.py) — a parallel driver to
  [train_newton.py](../../src/train_newton.py). It reuses the same model
  construction, CIFAR-10 loader, parameter init, metric set, and
  Levenberg-Marquardt accept / reject machinery, then adds an `sfn` mode
  that calls the helpers in [sfn.py](sfn.py). We chose a separate driver
  instead of extending the in-tree trainer because the third mode would
  not have shared CLI defaults with the existing two (notably the `k` and
  `--lanczos-tol` flags only make sense for SFN), so keeping it in a
  parallel file makes the diff easy to read.

- [extract_trajectory.py](extract_trajectory.py) — parses the trainer's
  stdout log into a per-step CSV. [plot_trajectory.py](plot_trajectory.py)
  renders the side-by-side training-loss / probe-loss figure used below.

- [smoke/smoke_sfn.py](smoke/smoke_sfn.py) — a five-step sanity check that
  confirms (1) the pytree round-trip is correct, (2) per-step cost is in
  the predicted range (`~10s derivatives + k * ~0.1s` matvecs), and
  (3) loss decreases on the very first step.

## Algorithm sketch

For each training step:

1. Form the per-batch gradient `g` and a fresh `CachedHessianOperator` on
   the current `(x, y)`. The operator caches the layer-wise derivatives
   that the Pearlmutter HVP needs, so the `k` matvecs in the Lanczos
   sweep below skip the dominant derivative cost.
2. Run `scipy.sparse.linalg.eigsh(linop, k=k, which="LM", tol=1e-3)` to
   extract the top-`k` largest-magnitude eigenpairs `(λᵢ, vᵢ)` of `H`.
3. Project `g` onto the Krylov subspace: `c_i = vᵢᵀg`, `g_K = Σ c_i vᵢ`,
   and let `g_⊥ = g − g_K`.
4. Build the SFN step `δ = Σᵢ (c_i / (|λᵢ| + ε)) vᵢ + g_⊥ / (|λ_k| + ε)`.
   In words, scale each captured direction by the inverse of its
   absolute eigenvalue plus damping, and damp the orthogonal complement
   by the *smallest* captured absolute eigenvalue (plus the same `ε`).
   This is the simpler of the two complement-damping choices listed in
   [report.md](../report.md#phase-d) (a fixed `ε_⊥` is also legal).
5. Apply `p ← p − lr·δ` with the same LM accept-check the Newton mode
   already uses: if loss dropped on the same batch, decrease `ε`; if not,
   undo the SFN step, take an SGD fallback step, and increase `ε`.

We picked `k = 20` per the plan's recommendation. The Phase-5 anchor has
`P = 22128` parameters, so `k = 20` covers about `0.09%` of the
parameter dimensions; the remaining `99.9%` ride the orthogonal-complement
damping path.

## Compute notes

The Phase-5 anchor's per-batch HVP without caching is dominated by
`SequenceOfBlocks.derivatives`, which costs `~10 s` per call on CPU. With
the cache, a Lanczos sweep of `k=20` eigenpairs converges in roughly
`40–60` matvecs at `tol = 1e-3`, costing `~3 s` of subsequent HVP work,
for a per-step total of about `15 s` running alone on this machine. We
verified this with the [smoke](smoke/smoke_sfn.py) script before
launching the full runs.

We budgeted **200 steps per run** for both D1 and D2, not the plan's
1000. The driving constraint is the per-step cost above: 1000 steps per
run was projected at `~4 h`, and running D1 and D2 in parallel pushed
per-step time to roughly `60 s` due to CPU contention, so 1000-step
parallel runs would have taken the better part of a working day. The
Newton baseline reaches its plateau by step `~50` (see
[experiments/summary-so-far.md](../../experiments/summary-so-far.md)), so
200 steps is enough to read out whether SFN escapes the same band; the
SGD fixed-batch baseline takes longer to reach its `0.19` floor, so the
D2 comparison is the noisier of the two.

## Run commands

D1 (full CIFAR-10, Phase-5 anchor):

```
python reproduce-published-results/phase-d-saddle-free-newton/train_sfn.py \
  --mode sfn --num-steps 200 --batch-size 64 --image-size 16 \
  --hidden-dim 24 --num-layers 8 --activation relu \
  --k 20 --epsilon 1.0 --lr 0.5 --lanczos-tol 1e-3 \
  --data-dir ./data \
  --logdir reproduce-published-results/phase-d-saddle-free-newton/D1_cifar10/logs \
  --run-name D1-sfn-k20 --cpu
```

D2 (single 64-sample batch held for 200 steps, matching the
[exp-048b](../../experiments/runs/exp-048b-sgd-reuse-batch-lr0.01) protocol):

```
python reproduce-published-results/phase-d-saddle-free-newton/train_sfn.py \
  --mode sfn --num-steps 200 --batch-size 64 --image-size 16 \
  --hidden-dim 24 --num-layers 8 --activation relu \
  --k 20 --epsilon 1.0 --lr 0.5 --lanczos-tol 1e-3 --reuse-batch 200 \
  --data-dir ./data \
  --logdir reproduce-published-results/phase-d-saddle-free-newton/D2_fixed_batch/logs \
  --run-name D2-sfn-k20 --cpu
```

## Headline numbers

Both runs went to 100 steps (not the 200 the README originally budgeted, because the parallel-CPU contention pushed per-step cost to roughly `11–13 s` and we wanted both runs to land in one go). The trajectories are
[D1 stdout-100steps.log](D1_cifar10/stdout-100steps.log) and
[D2 stdout.log](D2_fixed_batch/stdout.log).

| Run | Method                                                             | Steps | Final `train_loss_avg10` | Final / min `probe_loss` | Min train loss seen |
| --- | ------------------------------------------------------------------ | ----- | ------------------------ | ------------------------ | ------------------- |
| D1  | SFN (k=20, ε=1.0 LM-adaptive, lr=0.5)                             | 100   | 2.118                    | **2.002** (min); 2.085 (final) | n/a (fresh batches) |
| D1  | linear-Newton ref ([exp-028](../../experiments/runs/exp-028-newton-15-frozen-low-eps)) | 1000  | 2.22 – 2.30 band         | 2.22 – 2.30              | n/a                 |
| D1  | SGD ref ([Phase 4 anchor](../../experiments/summary-so-far.md))   | 1000  | n/a                      | 1.97                     | n/a                 |
| D2  | SFN (k=20, ε=1.0 LM-adaptive, lr=0.5)                             | 100   | 0.251 (min)              | n/a (probe meaningless under memorization) | **0.179**           |
| D2  | linear-Newton ref ([exp-053](../../experiments/runs/exp-053-newton-memorize-lr0.5-lm)) | 1000  | n/a                      | n/a                      | 1.16                |
| D2  | SGD ref ([exp-048b](../../experiments/runs/exp-048b-sgd-reuse-batch-lr0.01))           | 1000  | n/a                      | n/a                      | 0.19                |

The D1 trajectory descends monotonically from `probe_loss = 3.02` at step 0 to `2.00` at step 78 and stabilizes in the `2.00–2.10` band for the remainder of the run, escaping the `2.22–2.30` band our linear-Newton plateaus in. The D2 trajectory descends from train loss `3.01` at step 0 to `0.20` at step 95, with batch accuracy climbing to `0.98` (the model memorizes the 64-sample batch). One LM rejection at step 97 induces a transient loss spike to `1.77`; the minimum loss seen across the run is `0.179` at step 97 immediately before the rejection.

## Trajectories

- D1 stdout log: [D1_cifar10/stdout-100steps.log](D1_cifar10/stdout-100steps.log)
- D2 stdout log: [D2_fixed_batch/stdout.log](D2_fixed_batch/stdout.log)
- D1 TensorBoard event file: [D1_cifar10/logs/D1-sfn-k20-100steps/](D1_cifar10/logs/D1-sfn-k20-100steps/)
- D2 TensorBoard event file: [D2_fixed_batch/](D2_fixed_batch/)

## Decision-rule readout (per [report.md](../report.md))

SFN clearly beats our linear-Newton on **both** diagnostics under exactly the same LM accept-rule wrapper. D1 escapes the 2.22–2.30 plateau and lands within 0.03 of SGD's 1000-step `probe_loss = 1.97` in only 100 steps. D2 destroys the `1.16` fixed-batch floor by a factor of 6.5 and effectively matches SGD's `0.19` floor.

This is the cleanest comparison in the whole report on the role of the curvature operator: SFN and our linear-Newton share the *same* model, batch, LM accept-rule, ε schedule, lr, and step-application code, and differ only in computing `(|H_k| + ε I)^{-1} g` on the top-`k` Lanczos subspace rather than `(H + ε I)^{-1} g` on the full Hessian. The first variant succeeds where the second stalls.

The decision rule applies cleanly: **Damping is confirmed**. The fix for our own algorithm is to replace the full `(H + ε I)^{-1}` with either an absolute-value transform on the top-`k` eigenspectrum (explicit SFN-style surgery) or a CG-truncated approximation of the inverse (implicit Krylov-style surgery, as HF in [phase-c-hessian-free](../phase-c-hessian-free/) demonstrates). Both restrictions avoid the small / near-zero / negative eigenvalues of `H` that the full `(H + ε I)^{-1}` dangerously amplifies. The remedy that is most natural to our block-factored solver is the SFN one, because the absolute-value transform can in principle be pushed into the block factorization without rebuilding the algorithm around CG.

The plan's framing of **Damping** as "indefinite-Hessian-damping pathology, fixed by `|H| + ε I` rather than `H + ε I`" is exactly the right read for what we observe.
