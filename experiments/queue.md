# Newton experiment queue

Each experiment is a top-level YAML frontmatter block followed by a hypothesis paragraph. Status transitions: `pending` → `running` → `done`.

**Fixed config across all experiments (do not change):**
- model: `--hidden-dim 8 --num-layers 8 --image-size 16 --activation tanh`
- training budget: `--num-steps 15`
- probe loss metric: `probe_loss` from a 256-sample held-out batch (seeded fixed)

**Success criterion:** `probe_loss` at step 14 < 2.20 (decisively below the random-guess 2.30 plateau). Stretch: < 1.95 (matches SGD's long-run number).

**Git workflow:**
- Working tree must be clean before the Executor runs an experiment (only `experiments/queue.md` may be dirty due to status flip).
- Executor commits any `code_patch` + the queue status update before running, captures `git rev-parse HEAD`, and records it as `commit_hash` on the entry.
- After the run finishes, Executor commits the result artifacts (`experiments/runs/<id>/...`, queue status=done) as a second commit. The recorded `commit_hash` always refers to the *training-time* commit.

---

```yaml
id: exp-001-very-high-eps
status: done
commit_hash: 49c429ce21de99fd6aa6c737b78ca33619b44e54
hypothesis: |
  At random init the per-batch Hessian has many tiny eigenvalues whose inverses dominate the Newton step.
  Heavy damping (epsilon ≫ typical Hessian eigenvalue) reduces Newton toward gradient descent along
  well-conditioned directions while still preconditioning the well-conditioned ones. Should descend like
  scaled SGD with lr = lr/eps in the worst case.
flags:
  --mode: newton
  --epsilon: 100.0
  --lr: 1.0
  --lm-up: 1.0
  --lm-down: 1.0
  --batch-size: 64
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-001-very-high-eps
  --log-every: 1
code_patch: null
predicted_outcome: probe_loss should descend slowly but monotonically; final near 2.25.
```

---

```yaml
id: exp-002-batch-reuse-K3
status: done
commit_hash: 34aa5ec725a70513bf0f00a1ffaf1c6c311f581e
hypothesis: |
  The LM accept/reject check is per-batch, but the Hessian and gradient on different batches disagree.
  Reusing the same batch for several consecutive Newton steps lets the optimizer converge on that batch's
  loss surface, where the Hessian is locally accurate. Should produce visible per-step descent on probe_loss
  during reuse, even if descent stalls when the batch switches.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.5
  --batch-size: 64
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-002-batch-reuse-K3
  --log-every: 1
  --reuse-batch: 3
code_patch: |
  Add a CLI flag `--reuse-batch` (int, default 1). In the training loop, hold the (x, y) batch
  fixed for that many consecutive steps before drawing a new one. Default 1 preserves current
  behavior. Minimal patch — just modify the batch-fetching block at the top of the while loop
  and add the argparse line.
predicted_outcome: probe_loss drops sharply within each 3-step batch-reuse window, then plateaus on switch. Final near 2.10.
```

---

```yaml
id: exp-003-epsilon-sweep-down
status: pending
commit_hash: null  # filled by Executor at run start
hypothesis: |
  exp-001 showed that epsilon=100 over-damps the Newton step, so the effective learning rate along
  well-conditioned directions is roughly lr/epsilon = 0.01, which is too small to clear the random-guess
  plateau in 15 steps. The interpretation noted the trajectory flattens with |g| dropping from 6 to 1.2,
  indicating stalling rather than convergence. Sweeping epsilon downward toward the natural Hessian
  eigenvalue scale should give a larger effective step on well-conditioned directions while still
  damping the ill-conditioned ones. To isolate the damping variable from exp-001, every other knob
  (lr, lm-up, lm-down, batch-size) is held identical to exp-001; only epsilon changes from 100 to 10.
  If descent improves, damping level was the bottleneck; if it gets noisier or LM rejection rate rises,
  the natural Hessian scale is below 10 and we should sweep further.
flags:
  --mode: newton
  --epsilon: 10.0
  --lr: 1.0
  --lm-up: 1.0
  --lm-down: 1.0
  --batch-size: 64
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-003-epsilon-sweep-down
  --log-every: 1
code_patch: null
predicted_outcome: probe_loss descends faster than exp-001 and ends near 2.10, with possibly one or two LM rejections appearing as small upticks.
```

---

```yaml
id: exp-004-large-batch-moderate-eps
status: pending
commit_hash: null  # filled by Executor at run start
hypothesis: |
  exp-001's interpretation attributed the step-to-step bumpiness to per-batch Hessian/gradient mismatch
  against the held-out probe, because we step on minibatch directions but measure on a fixed probe.
  exp-002 attacks this by reusing the same small batch; this experiment attacks it from the other side
  by enlarging the batch so a single step already averages over more samples, which should reduce both
  the gradient variance and the Hessian estimation noise without requiring any code change. To isolate
  the batch-size variable, lr, lm-up, and lm-down match exp-001, and epsilon is set to 10 rather than
  100 so the step is not over-damped (otherwise larger batches would not help, since the bottleneck
  would still be effective step size, not noise). If exp-004 outperforms exp-003 (same epsilon, smaller
  batch), batch noise was a dominant contributor to bumpiness.
flags:
  --mode: newton
  --epsilon: 10.0
  --lr: 1.0
  --lm-up: 1.0
  --lm-down: 1.0
  --batch-size: 256
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-004-large-batch-moderate-eps
  --log-every: 1
code_patch: null
predicted_outcome: probe_loss descends more smoothly than exp-001 with fewer upticks, ending near 2.05.
```

---

```yaml
id: exp-005-small-lr-damped-newton
status: pending
commit_hash: null  # filled by Executor at run start
hypothesis: |
  Comparing exp-001 (epsilon=100, lr=1.0, zero rejections, descent to 2.38) against exp-002
  (epsilon=1.0, lr=0.5, 20 percent rejections, drift to 2.49) suggests the bottleneck is that
  at low epsilon the Newton step is large enough that the per-batch quadratic model is wrong
  on its own batch, so LM rejects. Rather than fight this by raising epsilon, classical damped
  Newton with line search shrinks the global step instead. If we hold epsilon at 1.0 but cut lr
  to 0.1, the Newton direction still preconditions across well-conditioned directions, yet the
  actual displacement per step is one-tenth as large, so the local quadratic model should remain
  valid and rejections should drop. This isolates the step-size variable from exp-002: only lr
  changes (0.5 to 0.1), with epsilon, batch-size, lm-up, and lm-down unchanged. If exp-005
  outperforms exp-002, the issue at low epsilon was global step magnitude rather than damping
  level; if rejection rate stays near 0.2, low epsilon itself is the problem and damping needs
  to rise regardless of lr.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.1
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-005-small-lr-damped-newton
  --log-every: 1
code_patch: null
predicted_outcome: probe_loss descends nearly monotonically with rejection rate below 0.1, ending near 2.15.
```

---

```yaml
id: exp-006-high-eps-fast-relax
status: pending
commit_hash: null  # filled by Executor at run start
hypothesis: |
  exp-001 with epsilon=100 and lm-down=1.0 (no adaptation) descended cleanly but stalled at 2.38
  because the effective step on well-conditioned directions was roughly lr/epsilon = 0.01, which
  is too small to clear the random-guess plateau. The interpretation noted |g| collapsing from 6
  to 1.2 indicates the trajectory entered a flat region with no remaining budget to escape. If
  instead we start at the same safe epsilon=100 (so initial steps are scaled-SGD-like and never
  rejected) but let LM aggressively relax damping after each accepted step via lm-down=0.5, the
  effective epsilon halves every accepted step and reaches roughly 100 * 0.5^15 ~ 0.003 by step
  15 in the best case, transitioning the run from scaled-SGD into true Newton once the iterate
  is in a region where the quadratic model is reliable. This isolates the lm-down variable from
  exp-001: only lm-down changes (1.0 to 0.5), with epsilon, lr, lm-up, and batch-size unchanged.
  If exp-006 outperforms exp-001 with low rejection rate, the bottleneck was static over-damping
  and LM adaptation can fix it without retuning epsilon; if rejection rate climbs sharply mid-run,
  epsilon collapses faster than the iterate can absorb and lm-down=0.5 is too aggressive.
flags:
  --mode: newton
  --epsilon: 100.0
  --lr: 1.0
  --lm-up: 2.0
  --lm-down: 0.5
  --batch-size: 64
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-006-high-eps-fast-relax
  --log-every: 1
code_patch: null
predicted_outcome: probe_loss starts descending like exp-001 then accelerates after step 8 as effective epsilon drops below 10, ending near 2.00.
```
