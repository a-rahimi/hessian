# Newton experiment queue

Each experiment is a top-level YAML frontmatter block followed by a hypothesis paragraph. Status transitions: `pending` → `running` → `done`.

**Starting model config (Phase 1, exp-001 through exp-007):** `--hidden-dim 8 --num-layers 8 --image-size 16 --activation tanh --num-steps 15`. All Phase 1 runs reached probe_loss in [2.33, 2.49] at step 14 — Newton and SGD indistinguishable inside this budget.

**Phase 2 — model-scale sweep (this section onwards):** the Planner may now vary model architecture AND training params. The goal is to find a model + hyperparameter combination where the optimizers actually separate on the metric. Vary one knob per experiment when possible.

Hard constraints (machine limits):
- `--batch-size <= 128` (256 OOMs).
- First-layer Hessian block memory = (input_dim × hidden_dim)² × 4 bytes must stay under ~600 MB. At image_size=16 this caps `hidden_dim <= ~32`. At image_size=8 it caps `hidden_dim <= ~50`. Increasing image_size beyond 16 forces hidden_dim downward.
- Newton step time scales super-linearly with batch_size; batch=64 is fast (~1 s/step), batch=128 ~10 s/step.

Sensible scaling axes:
- Wider: bump `--hidden-dim` from 8 → 16 → 24 (mind the memory cap).
- Deeper: bump `--num-layers` from 8 → 16 → 24 (inner-layer Hessian blocks are cheap since they scale as `hidden_dim²`, but more layers mean a bigger augmented system in `hessian_inverse_product`).
- Longer training horizon: `--num-steps` from 15 → 60 → 200. At 60 steps Newton runs take ~2 min, still tractable.
- Probe-batch metric is now `probe_accuracy` as well as `probe_loss` (committed in this round).

**Success criterion (Phase 2):** Newton's `probe_loss` at the final step is at least 0.05 lower than SGD's `probe_loss` at the final step, on the same model and training horizon. Tie or worse means Newton offers no benefit at that scale.

**Phase 3 outcome:** Newton beat SGD by 0.030 at (h8, l16, 60 steps) and by 0.021 at (h8, l24, 60 steps). Direction correct, magnitudes below the 0.05 threshold. Deeper did NOT widen the gap as predicted.

**Phase 4 — find the smallest SGD-trainable model (new goal, replaces the Phase 3 path).** Before more Newton experiments, find a (hidden_dim, num_layers, num_steps, lr) where SGD reaches `probe_loss < 2.00` AND `probe_accuracy > 0.20` — meaningfully above chance (probe_loss = log(10) = 2.3026, probe_accuracy = 0.10). Preferred shape: deep and narrow (`hidden_dim = 8` stays fixed). Tune by growing `num_layers` and `num_steps`, possibly with higher `lr`. Newton runs resume only after we land in this regime.

**Phase 4 outcome:** ReLU with hidden-dim=8 is dead at every depth tested (ReLU saturation at narrow width). The smallest SGD-trainable config is `--num-layers 8 --hidden-dim 24 --activation relu --lr 0.1 --num-steps 1000`, reaching probe_loss=1.9712 / probe_accuracy=0.281. Phase 5 anchors on this model.

**Phase 5 — Newton tuning on the Phase 4 anchor.** Model: `--num-layers 8 --hidden-dim 24 --image-size 16 --activation relu`. SGD wall-clock to reach probe_loss < 2.0 is ~3 seconds at num-steps=1000. Newton step time at this config is ~27 s/step (measured smoke), so Newton experiments use shorter horizons (num-steps=30 → ~13 min per run). Comparison metric: probe_loss at the final step at the same num-steps. Phase 5 success: Newton at num-steps=30 reaches probe_loss at least 0.05 lower than SGD's probe_loss at num-steps=30.

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
status: done
commit_hash: e6adc8a7cf4238679d4a114b95aa249f7d70b417
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
status: failed
commit_hash: 147f050ae787a492b8e0a05aeba46e4a55ed545e
failure_reason: |
  OOM at step 0 on three attempts. Hessian intermediates at batch_size=256 exceeded
  available RAM (~10 GB RSS, SIGKILL by jetsam). batch_size=128 had already been
  observed earlier in this project to push step time to ~22s; 256 is not runnable on
  this machine. Future plans should keep batch_size <= 128 (and probably <= 64 for
  fast iteration).
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
status: done
commit_hash: f453954ed2221b25c1d798d90f58001e5a865b62
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
status: done
commit_hash: 7e875d3b8bdb5324852e2eae506e019fdcc7d652
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

---

```yaml
id: exp-007-sgd-baseline-15-steps
status: done
commit_hash: a2f5985e2da0bfaf0210f95a9b8d5d0f0d761a9f
hypothesis: |
  Every Newton run so far settles in the narrow band [2.33, 2.49] regardless of damping or step
  size, which is suspiciously close to the random-guess plateau of 2.30 and well above the 2.20
  target. The earlier project note that SGD reached ~1.9 in 1000 steps tells us nothing about
  what SGD can achieve in 15 steps on this exact hidden-dim 8, num-layers 8, image-size 16, tanh
  configuration. If a 15-step SGD baseline also lands in the [2.33, 2.49] band, the plateau is a
  property of the model+budget pair (a true noise floor at this depth and batch size) rather than
  a failure of Newton, and chasing further Newton hyperparameters is wasted effort. If SGD clears
  2.20 in 15 steps, Newton genuinely lags scaled gradient descent here and the question becomes
  why the Hessian preconditioning is not buying us anything. Either way the result tightly bounds
  the rest of the search. We use lr=0.1 (the SGD default in train_newton.py) and batch-size=64
  to match exp-001/003/005/006 so the only difference from the Newton runs is the optimizer.
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-007-sgd-baseline-15-steps
  --log-every: 1
  --hidden-dim: 8
  --num-layers: 8
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss ends in [2.30, 2.40], roughly tied with the best Newton runs, confirming the band is a budget-imposed noise floor.
```

---

```yaml
id: exp-008-tiny-lr-large-batch
status: skipped
skip_reason: |
  Predates Phase 2 framing. Holds Phase 1 model fixed (hidden_dim=8, num_layers=8, num_steps=15),
  so its result would only re-sample the [2.33, 2.49] noise floor that exp-001 through exp-007
  already established. Phase 2 is about scaling up the model, not refining hyperparameters at
  the Phase 1 scale.
commit_hash: null  # filled by Executor at run start
hypothesis: |
  exp-005 is the current best because cutting lr from 0.5 to 0.1 at epsilon=1.0 kept the per-step
  displacement inside the per-batch quadratic trust region, and it was the only run still descending
  at step 14 (slope -0.061). That run's rejection rate of 0.133 (two rejections clustered early)
  shows the step is still slightly too large for some early-iterate batches; cutting lr another 10x
  to 0.01 should drive rejection rate to near zero. At the same time, exp-002, exp-003, and exp-006
  all showed that the trajectory bounces by 0.1 to 0.34 between adjacent steps, which is the
  signature of single-batch Hessian/gradient noise dominating the signal on the held-out probe.
  exp-004 tried to attack this with batch-size=256 and OOM'd, but batch-size=128 is the hardware
  ceiling and has never been tried, so doubling batch size from 64 to 128 should roughly halve the
  per-step noise. Combining the two changes (lr 0.1 -> 0.01 to keep the step inside the trust
  region without help from LM, and batch 64 -> 128 to denoise both gradient and Hessian) isolates
  whether the exp-005 plateau is set by step size or by batch noise: if lr=0.01 at batch=128 beats
  exp-005's 2.3302, both knobs were still active at exp-005's setting; if it stalls higher because
  the step is too small to clear the random-guess region in 15 steps, lr=0.1 was already at the
  right magnitude and the binding constraint is something else (likely batch noise alone).
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.01
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 128
  --num-steps: 15
  --logdir: runs/auto
  --run-name: exp-008-tiny-lr-large-batch
  --log-every: 1
  --hidden-dim: 8
  --num-layers: 8
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss descends smoothly with rejection rate below 0.05, ending near 2.20 with the trajectory still descending at step 14.
```

---

```yaml
id: exp-009-width-sgd-baseline
status: done
commit_hash: ef0cf50e49d38144eff41b606a4b982d2d41e69b
hypothesis: |
  Phase 1 established that at the small model (hidden_dim=8, num_layers=8, image_size=16) Newton
  and SGD are indistinguishable inside a 15-step budget, because both saturate the same noise
  floor near probe_loss=2.36 set by held-out probe variance against single-batch updates. The
  Phase 2 hypothesis is that Newton's preconditioning matters more in larger or deeper models
  with worse-conditioned Hessians, so we need to grow the model before re-comparing. This run
  doubles the width to hidden_dim=16 while holding num_layers=8 and image_size=16 fixed, so the
  first-layer Hessian block stays at the memory budget (768*16)^2*4 = 600 MB. We also extend
  num_steps to 60 (4x Phase 1) so any optimizer has room to clear the random-guess plateau
  before the comparison is made; exp-007 showed that 15 steps is too short for the SGD
  trajectory to separate from the plateau on this dataset. This is the SGD half of the paired
  width-scale comparison against exp-010; the gap exp-010.probe_loss_final - exp-009.probe_loss_final
  is the Phase 2 success metric for the width axis.
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 60
  --logdir: runs/auto
  --run-name: exp-009-width-sgd-baseline
  --log-every: 5
  --hidden-dim: 16
  --num-layers: 8
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss descends past the 2.30 random-guess plateau and ends in the 2.05-2.20 range, giving Newton a real target to beat.
```

---

```yaml
id: exp-010-width-newton
status: done-truncated
commit_hash: 9c44725a0b8c6ab11782bf16671dd610c6209f59
truncation_note: |
  Process was killed at step 45 of 60 when the spawning agent's subshell was reaped at
  agent context exit. No NaN, no OOM, no Python exception — the process simply lost its
  parent and was terminated. The trajectory up to step 45 is intact and is a reliable
  read of Newton's behavior at this scale; the remaining 15 steps are missing. Final
  recorded probe_loss at step 45 is 2.5870, well above SGD's 2.2839 at step 55 on the
  paired exp-009 run, so the comparison is unambiguous even with truncation.
hypothesis: |
  Paired with exp-009. Same model (hidden_dim=16, num_layers=8, image_size=16) and same
  training horizon (60 steps, batch=64), but Newton instead of SGD. Settings are the best
  recipe from Phase 1: epsilon=1.0, lr=0.1, lm-up=1.1, lm-down=0.9 (exp-005's configuration),
  which on the small model gave the lowest final probe_loss of any Newton run with a -0.061
  end-of-run slope still descending at step 14. The Phase 2 hypothesis is that doubling width
  worsens Hessian conditioning enough that Newton's preconditioning produces a measurable gap
  over SGD: with 60 steps Newton should clear the plateau faster than SGD because it scales
  steps along ill-conditioned directions, whereas SGD uses one global lr. If exp-010 beats
  exp-009 by at least 0.05 in final probe_loss, the Phase 2 success criterion is met at the
  width axis; if the gap is smaller or reversed, width alone does not surface Newton's edge
  and we should pivot to depth (exp-012) or to longer horizons.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.1
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 60
  --logdir: runs/auto
  --run-name: exp-010-width-newton
  --log-every: 5
  --hidden-dim: 16
  --num-layers: 8
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss descends faster than exp-009 in the first 30 steps and ends at least 0.05 below exp-009's final, in the 1.95-2.10 range.
```

---

```yaml
id: exp-011-depth-sgd-baseline
status: done
commit_hash: 03823d792a4ed5c59daac4661a3b31de84f1da17
hypothesis: |
  The other natural scaling axis is depth rather than width. This run holds hidden_dim=8 (the
  Phase 1 width) but doubles num_layers from 8 to 16, with image_size=16 and 60 training steps.
  Deeper networks at fixed width have worse-conditioned loss surfaces because gradients and
  curvature compound through more nonlinearities, so the Hessian's condition number grows
  faster with depth than with width at the small scales we can afford here. Memory is fine
  because inner-layer Hessian blocks scale as hidden_dim^2 (only the first layer carries the
  input_dim factor), so 16 layers at hidden_dim=8 cost roughly the same memory as 8 layers at
  hidden_dim=8 plus a slightly larger augmented system in hessian_inverse_product. This is the
  SGD half of the paired depth-scale comparison against exp-012; pairing the two lets the
  Interpreter compute the gap directly without depending on cross-experiment baselines.
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 60
  --logdir: runs/auto
  --run-name: exp-011-depth-sgd-baseline
  --log-every: 5
  --hidden-dim: 8
  --num-layers: 16
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss descends more slowly than exp-009 because deeper-but-thinner SGD struggles with vanishing gradients through tanh; ends in the 2.20-2.35 range, possibly stuck near the plateau.
```

---

```yaml
id: exp-012-depth-newton
status: done
commit_hash: 7f0991f88a4189edbcd4ef8ec455d4eee806927d
hypothesis: |
  Paired with exp-011. Same model (hidden_dim=8, num_layers=16, image_size=16) and same
  horizon (60 steps, batch=64), but Newton with Phase 1's best recipe (epsilon=1.0, lr=0.1,
  lm-up=1.1, lm-down=0.9). The depth axis is where Newton's preconditioning is most likely
  to pay off, because as depth grows with fixed tanh activation the per-layer Jacobian
  products shrink and the Hessian's small eigenvalues dominate the SGD direction, leaving
  most coordinates under-stepped. Newton rescales each direction by its inverse curvature,
  so directions that SGD cannot move should still be reachable in 60 steps. If exp-012 beats
  exp-011 by at least 0.05 in final probe_loss, the Phase 2 success criterion is met at the
  depth axis. Comparing the (exp-012 - exp-011) gap against the (exp-010 - exp-009) gap then
  tells us which scaling axis surfaces Newton's preconditioning advantage more strongly,
  which is the next decision point for Phase 3.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.1
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 60
  --logdir: runs/auto
  --run-name: exp-012-depth-newton
  --log-every: 5
  --hidden-dim: 8
  --num-layers: 16
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss descends visibly past exp-011's trajectory after roughly step 15 and ends at least 0.05 below exp-011's final, in the 2.05-2.20 range.
```

---

```yaml
id: exp-013-deeper-sgd-baseline
status: done
commit_hash: d4f32d6867eb0f6b1e63d3c5e6a71419269de7c8
hypothesis: |
  Phase 2 closed with exp-012 beating exp-011 by 0.0302 on probe_loss at depth=16, which is
  directionally correct for the Newton-helps-on-depth thesis but below the 0.05 success
  threshold. The interpretation argued the preconditioning advantage should grow monotonically
  with depth, because deeper tanh stacks have more severe Jacobian-product attenuation along
  small-curvature directions that Newton rescales but SGD cannot. This run is the SGD half of
  the paired deeper-depth comparison against exp-014: it holds hidden_dim=8 (the Phase 1 width
  where Newton's recipe is well-tuned) and bumps num_layers from 16 to 24 with the same
  60-step budget. Memory remains fine because inner-layer Hessian blocks scale as hidden_dim^2
  and only the augmented system in hessian_inverse_product grows linearly with depth. The
  prediction is that SGD's stall on probe_loss should worsen relative to exp-011 (2.3348)
  because gradients have to propagate through 24 tanh stages instead of 16, so |g| should
  collapse even faster than exp-011's 9.05 -> 0.3-1.2 trajectory. If SGD lands above 2.35 at
  depth=24, the Newton advantage in exp-014 has a wider gap to open up; if SGD still reaches
  the 2.28-2.34 band, depth=24 is not appreciably harder for SGD than depth=16 and the
  depth-axis scaling story needs revisiting.
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 60
  --logdir: runs/auto
  --run-name: exp-013-deeper-sgd-baseline
  --log-every: 5
  --hidden-dim: 8
  --num-layers: 24
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss stalls higher than exp-011, ending in the 2.32-2.42 range with |g| collapsing below 1.0 by step 5 and the trajectory oscillating around the 2.30 plateau without sustained descent.
```

---

```yaml
id: exp-014-deeper-newton
status: done
commit_hash: 49a37ac4a7c68dc60c6b030bfc0141ae3f04798c
hypothesis: |
  Paired with exp-013. Same model (hidden_dim=8, num_layers=24, image_size=16) and same horizon
  (60 steps, batch=64), but Newton with the recipe that has produced every Newton win so far
  (epsilon=1.0, lr=0.1, lm-up=1.1, lm-down=0.9). The depth axis is the only axis in this study
  where Newton has beaten SGD at all, and exp-012's interpretation predicts the gap widens with
  depth because the vanishing-gradient regime that Newton's preconditioner fixes intensifies as
  more tanh nonlinearities are stacked. If exp-014 beats exp-013 by at least 0.05 in final
  probe_loss, the Phase 3 success criterion is cleanly met and the Newton-helps-on-depth story
  is no longer a partial-support reading. If the gap is between 0.03 and 0.05, the trend is
  real but slow and we need either more depth (exp-015/016 would need a deeper follow-up) or a
  longer horizon (which exp-016 already tests). If exp-014 fails to beat exp-013 at all, the
  exp-012 gap was an artifact and the depth axis is no better than the width axis was.
  EXECUTION NOTE: Newton at depth=24, batch=64, 60 steps is estimated at ~8 min wall clock
  (50 percent longer per step than depth=16 because the augmented system in
  hessian_inverse_product carries 50 percent more layer blocks). This exceeds the agent
  context-window for foreground execution; exp-010 was killed mid-run at the agent boundary
  for exactly this reason. The Executor must launch this run via Bash with run_in_background
  and use Monitor to wait for completion, or hand the run to the user.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.1
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 60
  --logdir: runs/auto
  --run-name: exp-014-deeper-newton
  --log-every: 5
  --hidden-dim: 8
  --num-layers: 24
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss descends past exp-013's stall band and ends at least 0.05 below exp-013's final, in the 2.20-2.30 range, with |g| holding in the 0.3-3.0 band rather than collapsing the way SGD's does.
```

---

```yaml
id: exp-015-longer-horizon-sgd-baseline
status: done
commit_hash: e2a2f2d9a421a2d61f9f16a63824d97a7c8152f9
hypothesis: |
  The exp-012 interpretation also predicted that a longer budget at depth=16 should widen the
  Newton gap purely from the slope differential, because exp-012's last-four slope was -0.00427
  per logged step (still descending) while exp-011's was +0.011 (already ascending). If both
  slopes hold over an additional 60 steps, the gap at step 119 would be roughly
  0.0302 + (0.011 - (-0.00427)) * 12 = 0.0302 + 0.185 = 0.215, comfortably above the 0.05
  threshold. That estimate is almost certainly optimistic because both trajectories will flatten,
  but even a 30 percent realization of it (0.06-0.07) would clear the threshold. This run is the
  SGD half of the paired longer-horizon comparison against exp-016: it keeps the exp-011
  configuration (hidden_dim=8, num_layers=16, batch=64) exactly identical and only doubles the
  step budget to 120. If SGD's final probe_loss at step 119 is meaningfully higher than its
  exp-011 step-55 final (2.3348), the stall hypothesis is reinforced because more budget cannot
  recover what saturated gradients have already lost; if SGD descends further on the longer
  budget, the depth=16 plateau is partly a budget issue rather than a pure vanishing-gradient
  ceiling. Either way, this run gives exp-016 a clean SGD target to beat.
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 120
  --logdir: runs/auto
  --run-name: exp-015-longer-horizon-sgd-baseline
  --log-every: 5
  --hidden-dim: 8
  --num-layers: 16
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss ends in the 2.30-2.40 range, similar to or slightly worse than exp-011's 2.3348, because vanishing-gradient stall is not a budget problem and extra steps mostly oscillate around the plateau.
```

---

```yaml
id: exp-016-longer-horizon-newton
status: deferred
commit_hash: null
defer_reason: |
  Phase 3 redirect (user direction): before pairing more Newton runs, find the SMALLEST
  model where SGD clearly beats chance (probe_loss < 2.00 AND probe_accuracy > 0.20).
  Without that anchor, every paired Newton run is competing against an SGD that barely
  learned, so the comparison can't tell us whether Newton helps in the regime that matters.
  exp-016 can be re-queued (status flipped back to pending) once Phase 4 finds the right
  model size.
hypothesis: |
  Paired with exp-015. Same model as exp-012 (hidden_dim=8, num_layers=16, image_size=16) and
  same Newton recipe (epsilon=1.0, lr=0.1, lm-up=1.1, lm-down=0.9), but the horizon doubles to
  120 steps. The exp-012 trajectory was still descending at step 55 (last-four slope -0.00427),
  so extending the horizon should let Newton continue past its 2.3046 final while SGD on
  exp-015 stays parked near 2.33. The success criterion has two readings. First, if exp-016's
  final probe_loss is at least 0.05 below exp-015's final, the Phase 3 criterion is met along
  the horizon axis. Second, if exp-016 beats exp-012's 2.3046 by 0.05 or more, the slope
  differential prediction is validated and the right way to surface Newton's advantage on this
  problem is simply to train longer rather than to scale model size further. Comparing exp-016
  against exp-014 then tells us whether width-of-depth-axis (more layers) or length-of-horizon
  (more steps) is the more cost-effective lever for surfacing Newton's edge.
  EXECUTION NOTE: Newton at depth=16, batch=64, 120 steps is estimated at ~11 min wall clock
  (twice exp-012's ~5.5 min). This is the largest single-experiment cost in Phase 3 and is
  well past the agent context-window foreground budget. The Executor MUST launch this run
  via Bash with run_in_background and use Monitor to wait for completion, or hand the run to
  the user. Foreground execution will produce a truncation like exp-010's.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.1
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 120
  --logdir: runs/auto
  --run-name: exp-016-longer-horizon-newton
  --log-every: 5
  --hidden-dim: 8
  --num-layers: 16
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss continues to descend past exp-012's step-55 endpoint of 2.3046 and ends at least 0.05 below exp-015's final, in the 2.18-2.28 range, with rejection rate similar to exp-012's 0.4 because the LM controller is doing its expected work.
```

---

```yaml
id: exp-017-deep-long-horizon-sgd
status: done
commit_hash: 3a0cf9c4b415fe897efc87b02f2f1dba0adfc9fe
hypothesis: |
  The Phase 4 bar (probe_loss < 2.00 AND probe_accuracy > 0.20) has not been cleared by any SGD
  run so far, and exp-015 showed that doubling the budget at depth=16 from 60 to 120 steps barely
  moved the needle (2.3348 -> 2.2978). The interpretation of exp-011 and exp-015 attributed the
  stall to vanishing-gradient saturation at depth=16 with tanh, where |g| collapses to 0.3-1.2
  within the first 5 steps and stays there. If the bottleneck is mostly that we never give SGD
  enough wall-clock to crawl out of the saturation regime, then quadrupling the budget at the
  deepest configuration we have data for (depth=24, exp-013 stalled at 2.3103/0.141 after 60
  steps) should let the slow descent compound. The trajectory of exp-013 was already nearly flat
  at step 55, so going to 500 steps is a strong stress test: if SGD lands below 2.20 with
  accuracy >0.18, the stall is partly budget-limited and a 2000-step run would likely clear the
  bar; if it stays near 2.30/0.14, depth=24 is genuinely saturated and pushing horizon further
  is wasted. SGD is fast (<1s/step), so 500 steps is ~5-8 min wall clock.
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 500
  --logdir: runs/auto
  --run-name: exp-017-deep-long-horizon-sgd
  --log-every: 10
  --hidden-dim: 8
  --num-layers: 24
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss ends around 2.20-2.27 with probe_accuracy around 0.15-0.18, still short of the Phase 4 bar but with a measurable slope improvement over exp-013's stall, telling us depth=24 is partly saturation and partly budget.
```

---

```yaml
id: exp-018-extreme-deep-long-horizon-sgd
status: done
commit_hash: 1919f277ccbf7f2a32f65841c5ec0bd07c89f922
hypothesis: |
  Pairs with exp-017 along the depth axis. exp-013 (depth=24, 60 steps) stalled at 2.3103/0.141
  and exp-017 tests whether more budget rescues depth=24. This run instead asks whether even
  deeper (depth=32) can clear the Phase 4 bar given the same long horizon. The deep-narrow
  preferred shape from Phase 4 means we should push num_layers as far as we can; depth=32 has
  not been explored yet and lies just outside the range exp-011/exp-013 covered. The theory is
  ambiguous here. On one hand, deeper tanh networks have more representational capacity for the
  CIFAR-10 probe and could in principle reach lower probe_loss given enough steps. On the other,
  vanishing-gradient saturation gets sharply worse with depth at hidden_dim=8, so SGD might find
  itself unable to escape the chance plateau at all. Comparing exp-018 against exp-017 isolates
  depth-at-fixed-budget: if exp-018 beats exp-017, deeper helps even when saturated; if exp-018
  is worse than exp-017, depth=24 is already past the optimum and the search should pivot back
  to shallower configurations with bigger budgets.
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 500
  --logdir: runs/auto
  --run-name: exp-018-extreme-deep-long-horizon-sgd
  --log-every: 10
  --hidden-dim: 8
  --num-layers: 32
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss ends around 2.25-2.32 with probe_accuracy around 0.13-0.16, slightly worse than exp-017 because vanishing-gradient saturation at depth=32 dominates the extra-capacity benefit at hidden_dim=8.
```

---

```yaml
id: exp-019-high-lr-depth16-sgd
status: done
commit_hash: 67d2fa0e98f22975d7967e69565fd2673013d740
hypothesis: |
  Every prior SGD experiment fixed lr=0.1, which is the train_newton.py default. exp-015 showed
  that at depth=16 with 120 steps and lr=0.1, SGD only crawls to 2.2978/0.145, which is barely
  better than exp-011's 60-step run at the same lr (2.3348/0.141). If the bottleneck at depth=16
  is that effective per-step displacement is too small once gradients have attenuated through 16
  tanh stages (|g| in the 0.3-1.2 range per exp-011), then tripling lr should triple the step
  magnitude in the saturated regime and let the trajectory escape the chance plateau in
  comparable wall-clock. The risk is that early in training |g| is large (exp-011 logged 9.05 at
  step 0) and lr=0.3 would produce a 3x-larger initial step that could blow up. To buffer against
  that we also extend num_steps to 500 so the run has plenty of room to recover even if the first
  few steps are erratic. This isolates the learning-rate axis at depth=16, which has been
  underexplored: comparing against exp-015 (same depth, same hidden_dim, lr=0.1, 120 steps)
  tells us whether lr is the dominant lever or just a multiplier.
flags:
  --mode: sgd
  --lr: 0.3
  --batch-size: 64
  --num-steps: 500
  --logdir: runs/auto
  --run-name: exp-019-high-lr-depth16-sgd
  --log-every: 10
  --hidden-dim: 8
  --num-layers: 16
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss ends around 2.10-2.20 with probe_accuracy around 0.17-0.22, brushing or just clearing the Phase 4 accuracy bar while staying above the 2.00 loss bar, because higher lr lets the iterate descend past the chance plateau but cannot escape the underlying saturation noise.
```

---

```yaml
id: exp-020-aggressive-lr-stress-sgd
status: done
commit_hash: df47aac33aea811b33507296072d9fe300951398
hypothesis: |
  exp-019 takes a moderate step up the lr ladder (0.1 -> 0.3). This run takes the aggressive
  end of that ladder (lr=0.5) at depth=16 with a 300-step budget, to flank the search from the
  high-lr side and find out whether the optimizer simply diverges at this lr on this model. The
  reasoning is that if exp-019 at lr=0.3 lands at probe ~ 2.15 and exp-020 at lr=0.5 either
  lands lower (say ~2.05) or diverges, then we have a sharp upper bound on the useful lr range
  for this configuration. If exp-020 actually clears the Phase 4 bar (probe < 2.00, acc > 0.20)
  while exp-019 does not, lr was the dominant lever all along and Phase 4 has a clean answer.
  If exp-020 diverges (probe_loss climbing above the chance plateau or NaN), then lr=0.5 is too
  large at this depth and the useful range is in [0.1, 0.3]. The 300-step budget is shorter
  than exp-019's 500 so the experiments are not redundant: this is a stress test on lr, not a
  long-horizon run. Pairs informatively with exp-019: same depth, same hidden_dim, only lr and
  budget differ.
flags:
  --mode: sgd
  --lr: 0.5
  --batch-size: 64
  --num-steps: 300
  --logdir: runs/auto
  --run-name: exp-020-aggressive-lr-stress-sgd
  --log-every: 10
  --hidden-dim: 8
  --num-layers: 16
  --image-size: 16
  --activation: tanh
code_patch: null
predicted_outcome: probe_loss ends around 1.95-2.10 with probe_accuracy around 0.20-0.25, narrowly clearing the Phase 4 bar if the optimizer stays stable, or alternatively the run diverges with probe oscillating above 2.35 because lr=0.5 is past the stability boundary at depth=16 with tanh.
```

---

```yaml
id: exp-021-newton-anchor-sgd-control
status: running
commit_hash: TBD
hypothesis: |
  SGD control at the Phase 5 anchor model and the same num-steps Newton will use (30).
  Phase 4 showed SGD needs ~1000 steps to clear probe_loss=2.0 at this config, so 30 steps
  should leave it well above that. The point is to know exactly what bar Newton needs to
  clear at the same step budget. Predicted probe_loss in the 2.20-2.30 band based on the
  Phase 4 scan's intermediate readings (at scan-l8-h24-relu, step 250 ≈ 2.13).
flags:
  --mode: sgd
  --lr: 0.1
  --batch-size: 64
  --num-steps: 30
  --logdir: runs/auto
  --run-name: exp-021-newton-anchor-sgd-control
  --log-every: 1
  --num-layers: 8
  --hidden-dim: 24
  --image-size: 16
  --activation: relu
code_patch: null
predicted_outcome: probe_loss ends around 2.20-2.30, probe_accuracy around 0.10-0.15.
```

---

```yaml
id: exp-022-newton-anchor-baseline
status: pending
commit_hash: null  # filled by Executor at run start
hypothesis: |
  Newton at the Phase 5 anchor with the Phase 1 best recipe (epsilon=1.0, lr=0.1, lm-up=1.1,
  lm-down=0.9). Smoke test showed probe drops 3.02 → 2.42 in 5 steps with this recipe, which
  is a much steeper slope than SGD at the same step count. If the trend holds, 30 steps should
  put Newton well under SGD's same-step probe_loss. ~13 min wall clock.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.1
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 30
  --logdir: runs/auto
  --run-name: exp-022-newton-anchor-baseline
  --log-every: 1
  --num-layers: 8
  --hidden-dim: 24
  --image-size: 16
  --activation: relu
code_patch: null
predicted_outcome: probe_loss ends around 2.05-2.15, beating SGD-at-same-num-steps by >= 0.05.
```

---

```yaml
id: exp-023-newton-anchor-low-eps
status: pending
commit_hash: null  # filled by Executor at run start
hypothesis: |
  Sweep epsilon downward from 1.0 to 0.1. Lower damping means the Newton step is closer to the
  pure inverse-Hessian direction, which should help more when the per-batch Hessian is reasonably
  well-conditioned. At hidden-dim=24 with ReLU, the active subnetwork is less narrow than Phase 1
  so the Hessian's small eigenvalues are less likely to be near zero. Risk: low epsilon at this
  scale could produce huge |Δ| values like exp-002/exp-003 did at small models.
flags:
  --mode: newton
  --epsilon: 0.1
  --lr: 0.1
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 30
  --logdir: runs/auto
  --run-name: exp-023-newton-anchor-low-eps
  --log-every: 1
  --num-layers: 8
  --hidden-dim: 24
  --image-size: 16
  --activation: relu
code_patch: null
predicted_outcome: probe_loss ends around 2.00-2.10 if stable; alternatively jumps above 2.5 with high rejection rate if the lower damping over-amplifies small eigenvalues.
```

---

```yaml
id: exp-024-newton-anchor-high-lr
status: pending
commit_hash: null  # filled by Executor at run start
hypothesis: |
  Tripling lr from 0.1 to 0.3 at the Phase 1 best epsilon=1.0. Newton's |Δ| in exp-022's smoke
  was 0.3-1.3 at lr=0.1, so lr=0.3 should produce ~3x larger displacement per step. The Phase 4
  ReLU-SGD scan showed lr=0.3 KILLED SGD (gradient went to zero) on the narrow tanh model, but
  the wider ReLU model here may be more robust. If lr=0.3 helps without blowing up, the
  step-size axis is the dominant lever and the next round should push further.
flags:
  --mode: newton
  --epsilon: 1.0
  --lr: 0.3
  --lm-up: 1.1
  --lm-down: 0.9
  --batch-size: 64
  --num-steps: 30
  --logdir: runs/auto
  --run-name: exp-024-newton-anchor-high-lr
  --log-every: 1
  --num-layers: 8
  --hidden-dim: 24
  --image-size: 16
  --activation: relu
code_patch: null
predicted_outcome: probe_loss ends around 1.95-2.10 if stable, beating exp-022. Alternatively diverges with probe rising above 2.5 if lr=0.3 is past stability.
```
