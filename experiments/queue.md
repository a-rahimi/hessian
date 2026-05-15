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
status: pending
commit_hash: null  # filled by Executor at run start
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
status: pending
commit_hash: null  # filled by Executor at run start
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
