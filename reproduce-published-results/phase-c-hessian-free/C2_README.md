# C2 — Hessian-Free on the Phase 5 anchor in GGN and raw-Hessian modes

## What this is

[train_hf_anchor.py](C2_anchor/train_hf_anchor.py) wires the same
[ltatzel/PyTorchHessianFree](https://github.com/ltatzel/PyTorchHessianFree)
optimizer onto a parameter-matched re-build of our project's Phase 5 anchor —
`num_layers=8, hidden_dim=24, image_size=16, activation=relu`, no bias,
cross-entropy loss, batch size 64 on full CIFAR-10. The model is rewritten
as a plain `nn.Sequential` of `Linear+ReLU` blocks so BackPACK can hook into
it, with the same `kaiming_normal_(nonlinearity="relu")` init that
[src/train_newton.py](../../src/train_newton.py#L196) uses. The number of
trainable parameters is 22,128, matching the project's
[SequenceOfDenseBlocks](../../src/hessian.py#L349) anchor exactly.

We log the same metrics the project's
[StepScalars](../../src/train_newton.py#L44) emit:
`loss`, `train_loss_avg10`, `probe_loss`, `probe_accuracy`, `batch_accuracy`,
plus `step_seconds` and `wall_clock_s`. The probe batch is seeded identically
to `train_newton.py`'s `seed=999_999`, so `probe_loss` is comparable across
optimizers.

## Training commands

Full CIFAR-10 anchor, 1000-step budget (matching our Newton sweeps):

```
python train_hf_anchor.py --curvature-opt ggn     --num-steps 1000 \
  --batch-size 64 --data-dir ../../../data --logdir ./logs \
  --run-name hf_ggn_full --cpu

python train_hf_anchor.py --curvature-opt hessian --num-steps 1000 \
  --batch-size 64 --data-dir ../../../data --logdir ./logs \
  --run-name hf_hessian_full --cpu
```

Fixed-batch memorization diagnostic (single 64-sample batch held for the
entire run), matching the [exp-048b](../../experiments/runs/) protocol:

```
python train_hf_anchor.py --curvature-opt ggn     --fixed-batch \
  --num-steps 1000 --run-name hf_ggn_fixedbatch ...

python train_hf_anchor.py --curvature-opt hessian --fixed-batch \
  --num-steps 1000 --run-name hf_hessian_fixedbatch ...
```

A higher-damping sweep (`--damping 10.0 --cg-max-iter 30`) is included as
`hf_ggn_damp10` and `hf_hessian_damp10`; both extend stable training slightly
but still diverge before 1000 steps.

## Headline numbers

### Full CIFAR-10 anchor

The HF library blows past the local basin and diverges to NaN well before
1000 steps in every config we tried. The training script catches the first
NaN and stops, so the numbers below are best-step readings over the stable
window. The project's Newton plateau is `probe_loss ≈ 2.22-2.30` from
[experiments/summary-so-far.md](../../experiments/summary-so-far.md) and SGD
hits `probe_loss = 1.97`.

| run                               | best probe_loss | at step | NaN at step |
| --------------------------------- | --------------- | ------- | ----------- |
| HF-GGN, damping=1.0               | **2.189**       |  90     | 188         |
| HF-Hessian, damping=1.0           | **2.086**       | 253     | 447         |
| HF-GGN, damping=10.0              | 2.139           |  44     | 105         |
| HF-Hessian, damping=10.0          | 2.137           | 138     | 157         |
| Our Newton plateau (reference)    | 2.22 - 2.30     | -       | -           |
| SGD (Phase 4 reference)           | 1.97            | -       | -           |

Both HF modes reach the boundary of our Newton's plateau before they blow up,
and **HF-Hessian** in particular pushes meaningfully past it (2.086 < 2.22)
before the line search loses its handle. Neither HF mode reaches SGD's 1.97
in the stable window. Neither HF mode shows a clear Gauss-Newton-versus-raw-
Hessian gap in its best probe loss; if anything the raw-Hessian mode produces
the lower probe loss in the stable window.

### Fixed-batch memorization

The fixed-batch run holds a single 64-sample batch for 1000 steps. Our Newton
floor is `loss = 1.16`, SGD reaches `0.19` per `summary-so-far.md`.

| run                  | min training loss on fixed batch | at step |
| -------------------- | -------------------------------- | ------- |
| HF-GGN               | **8.4e-8**  (effectively zero)   | 106     |
| HF-Hessian           | **0.21**                         | 247     |
| Our Newton (ref)     | 1.16                             | -       |
| SGD (exp-048b ref)   | 0.19                             | -       |

Both HF modes memorize the fixed batch far below our Newton's `1.16` floor.
HF-GGN drives the training loss to single-precision zero in 106 steps;
HF-Hessian reaches `0.21`, which is comparable to SGD's `0.19` floor.
Both HF modes lift the probe loss to absurd values (12-20+ nats) during this
overfit, which is exactly what successful memorization looks like on a 64-
sample batch.

## Decision-rule readout

The C2 decision rule from the plan is:

| Phase outcome                                                                   | Diagnosis |
| ------------------------------------------------------------------------------- | --------- |
| HF-GGN succeeds, HF-raw-Hessian does not                                        | Damping   |
| Both HF modes succeed                                                           | Bug       |
| Both HF modes fail like our Newton                                              | Scale     |

On the full CIFAR-10 anchor, neither HF mode reaches SGD's `1.97` within the
stable window, but both modes clearly enter the Newton plateau region and HF-
Hessian crosses below it. There is **no clean GGN-vs-Hessian gap** in our
results; raw Hessian does at least as well as GGN on the stable window.

On the fixed-batch diagnostic, **both** HF modes destroy our Newton's `1.16`
floor — HF-GGN drives the loss to `8e-8`, HF-Hessian to `0.21`. By the table
above this is the "Both HF modes succeed" row, pointing at **Bug**: a
published full-Hessian-vector-product Newton-like method memorizes a 64-
sample batch using only matrix-vector products with the raw Hessian, while
our linear-inverse algorithm on the same model and batch stalls at `1.16`.

The fact that HF-GGN reaches `8e-8` faster than HF-Hessian reaches `0.21` is
the only Gauss-Newton signal we see, and it is much weaker than the
Bug-pointing signal of "raw-Hessian HF crushes our Newton's floor".

## Files

- [train_hf_anchor.py](C2_anchor/train_hf_anchor.py) — training driver,
  mirroring [src/train_newton.py](../../src/train_newton.py)'s metric schema.
- [plot_c2.py](C2_anchor/plot_c2.py) — plot generator.
- [c2_full_anchor.png](C2_anchor/c2_full_anchor.png),
  [c2_fixed_batch.png](C2_anchor/c2_fixed_batch.png) — comparison plots.
- [logs/](C2_anchor/logs/) — per-step CSV logs.
