# Phase A — Sub-sampled Newton on our anchor

This directory implements **Phase A** of [plan.md](../plan.md). The goal of the
phase is to give our linear-inverse Newton at
[hessian_inverse_product](../../src/hessian.py#L287) a "ground-truth"
counterpart on the actual Phase 5 anchor, so any gap between linear-inverse
Newton and SGD can be localized to one of **Bug**, **Damping**, or **Scale**.

The Section 3 precondition has been landed as a pytest. See
[SECTION3_RESULT.md](./SECTION3_RESULT.md): on the tiny model the linear-inverse
and dense solvers agree to `rel_err ~ 1e-13` for every ε, so the algorithm is
not silently wrong at toy scale.

## What lives here

- [subsampled_newton.py](./subsampled_newton.py) — one-file driver that
  materializes the full per-batch Hessian via `torch.func.hessian`, solves
  `(H + ε I) δ = g` with `torch.linalg.cholesky`-when-PD-otherwise-`torch.linalg.solve`,
  and applies `params -= lr · δ`. CLI mirrors [train_newton.py](../../src/train_newton.py)
  where possible; it logs the same per-step metrics (`loss`, `train_loss_avg10`,
  `probe_loss`, `probe_accuracy`, `grad_norm`, `step_norm`, `step_seconds`,
  `wall_clock_s`) to a per-run `metrics.csv`.
- [a3_numerical_agreement.py](./a3_numerical_agreement.py) — A3 driver. One
  forward, one backward, one Hessian, then dense solve vs.
  [hessian_inverse_product](../../src/hessian.py#L287) on the same `(x, y, g, ε)`.
- [plot_trajectories.py](./plot_trajectories.py) — small matplotlib helper that
  reads a `metrics.csv` and writes `trajectory.png`.
- [SECTION3_RESULT.md](./SECTION3_RESULT.md) — toy-scale sanity check write-up.
- [A3_numerical_agreement.md](./A3_numerical_agreement.md) — anchor-scale dense-vs-linear-inverse
  comparison write-up.
- [A1_cifar10/](./A1_cifar10/) — full-CIFAR-10 run.
- [A2_fixed_batch/](./A2_fixed_batch/) — single 64-sample batch held for the
  whole run.

## Budget reduction (read me first)

The plan asks for 1000-step A1 and A2 runs. On the anchor the dense per-batch
Hessian is `P × P = 22128 × 22128 ≈ 1.87 GB` in float32, and on the agent host
(macOS, 16 cores, contested by other agents running Phase D SFN and our own
linear-Newton sweep at the same time) a single sub-sampled Newton step takes
**20–32 s**. Running 1000 steps each for A1 and A2 with the contention we
actually had would have needed roughly 8 hours of wall-clock, well beyond a
single agent turn. Per the plan ("scale the experiments down to a budget that
completes in your turn (document the budget reduction in the README so I
know)"), we ran **100 steps per configuration** instead. Trends after 100 steps
are unambiguous in both directions (catastrophic divergence on A2-lr0.5 from
step 3, clear plateau on A1, clear descent on A2-lr0.1), so the conclusions are
not sensitive to the step count.

A1 / A2 / A3 each report below.

## A3 — Numerical agreement of dense and linear-inverse Newton steps

See [A3_numerical_agreement.md](./A3_numerical_agreement.md). One number: for
the first CIFAR-10 batch on the anchor (P = 22128) at ε ∈ {1.0, 0.1}, in
float64,

```
eps=1.0   rel_err = 1.016e-12
eps=0.1   rel_err = 4.466e-12
```

That is 8 orders of magnitude below the `< 1e-4` bar from Section 3 of
[plan.md](../plan.md). The linear-inverse algorithm computes the same step as
the dense solver on the actual anchor model.

## A1 — Sub-sampled Newton on full CIFAR-10

100 steps, batch size 64, fresh batch each step, `ε = 1.0, lr = 0.5`. Same
seed as our linear-Newton runs.

<!-- A1_HEADLINE -->

Compare to:

- **SGD** at `lr = 0.1` reaches `probe_loss = 1.9712` after 1000 steps on the
  same anchor ([experiments/summary-so-far.md](../../experiments/summary-so-far.md)).
- **Our linear-inverse Newton** plateaus in the 2.22–2.30 `train_loss_avg10`
  band on the same anchor across many ε / lr settings (see
  [summary-so-far.md](../../experiments/summary-so-far.md)).

## A2 — Sub-sampled Newton on the fixed-batch memorization diagnostic

100 steps with a single 64-sample batch held throughout. The first run at the
spec's `ε = 1.0, lr = 0.5` diverges immediately (see
[A2_fixed_batch/run_lr0.5_eps1.0_diverged/](./A2_fixed_batch/run_lr0.5_eps1.0_diverged/)
for the 7-step trajectory: loss climbs from `3.01` at step 0 to `527.76` at
step 6, with no LM accept/reject machinery to catch it). We then ran a single
modest variant at `lr = 0.1, ε = 1.0`.

Compare to:

- **SGD** on the same fixed batch at `lr = 0.01` reaches min loss `0.19` in
  1000 steps ([exp-048b](../../experiments/runs/exp-048b-sgd-reuse-batch-lr0.01/)).
- **Our linear-inverse Newton** on the same fixed batch reaches min loss `1.16`
  at `ε = 0.5, lr = 0.5` with LM-adaptive ε
  ([exp-053](../../experiments/runs/exp-053-newton-memorize-lr0.5-lm/)).

<!-- A2_HEADLINE -->

## Read on Bug / Damping / Scale

<!-- READ_HEADLINE -->

## How to reproduce

```
# A1 — full CIFAR-10
python reproduce-published-results/phase-a-subsampled-newton/subsampled_newton.py \
    --num-steps 100 --batch-size 64 \
    --lr 0.5 --epsilon 1.0 \
    --logdir reproduce-published-results/phase-a-subsampled-newton/A1_cifar10 \
    --run-name run

# A2 — fixed batch, lr=0.5 eps=1.0 (this diverges; kept for the record)
python reproduce-published-results/phase-a-subsampled-newton/subsampled_newton.py \
    --num-steps 100 --batch-size 64 --reuse-batch 100 \
    --lr 0.5 --epsilon 1.0 \
    --logdir reproduce-published-results/phase-a-subsampled-newton/A2_fixed_batch \
    --run-name run_lr0.5_eps1.0_diverged

# A2 — fixed batch, lr=0.1 eps=1.0 (the survivable variant)
python reproduce-published-results/phase-a-subsampled-newton/subsampled_newton.py \
    --num-steps 100 --batch-size 64 --reuse-batch 100 \
    --lr 0.1 --epsilon 1.0 \
    --logdir reproduce-published-results/phase-a-subsampled-newton/A2_fixed_batch \
    --run-name run_lr0.1_eps1.0

# A3 — numerical agreement against the dense solve (float64)
python reproduce-published-results/phase-a-subsampled-newton/a3_numerical_agreement.py --float64

# Plot all trajectories at once
python reproduce-published-results/phase-a-subsampled-newton/plot_trajectories.py \
    reproduce-published-results/phase-a-subsampled-newton/A1_cifar10/run/metrics.csv \
    reproduce-published-results/phase-a-subsampled-newton/A2_fixed_batch/run_lr0.1_eps1.0/metrics.csv \
    reproduce-published-results/phase-a-subsampled-newton/A2_fixed_batch/run_lr0.5_eps1.0_diverged/metrics.csv
```
