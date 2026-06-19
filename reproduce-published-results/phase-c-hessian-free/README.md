# Phase C — Hessian-Free on the autoencoder and on our Phase 5 anchor

This is the Phase C deliverable from
[reproduce-published-results/report.md](../report.md). It runs the published
Hessian-Free optimizer of Martens 2010 on two problems and compares it to
both SGD-with-momentum and to our project's linear-inverse Newton in
[src/hessian.py](../../src/hessian.py#L287). The result localizes our
Newton's plateau in the
**Bug / Damping / Scale** taxonomy from Section 5 of the plan.

## HF implementation and install

We use [ltatzel/PyTorchHessianFree](https://github.com/ltatzel/PyTorchHessianFree),
which provides a PyTorch `HessianFree` optimizer that wraps the GGN/Hessian-
vector products from [BackPACK](https://backpack.pt/) inside a CG solver with
LM damping and Martens-style backtracking. We clone it into
[third_party/PyTorchHessianFree](third_party/PyTorchHessianFree) and install
it editable against the project venv:

```
git clone https://github.com/ltatzel/PyTorchHessianFree.git \
  reproduce-published-results/phase-c-hessian-free/third_party/PyTorchHessianFree
/Users/arahimi/hessian/.venv/bin/python -m pip install backpack-for-pytorch
cd reproduce-published-results/phase-c-hessian-free/third_party/PyTorchHessianFree
/Users/arahimi/hessian/.venv/bin/python -m pip install -e .
```

A standalone sanity check
[C1_autoencoder/sanity_check_gnvp.py](C1_autoencoder/sanity_check_gnvp.py)
materializes `J^T H_loss J v` and `H v` with explicit autograd and compares
against the BackPACK products HF uses internally. Both products agree to
1.99e-7 (GGN) and 9.70e-8 (Hessian) on a tiny 4-layer test net — well below
the `1e-4` correctness gate the plan asks for.

## C1 — Hinton & Salakhutdinov 2006 autoencoder on MNIST

Full details in [C1_README.md](C1_README.md). Headline:

```
# default kaiming-uniform init
HF       (200 steps, ~150s wall): final test BCE/pixel = 0.332
SGD+mom (2000 steps, ~ 45s wall): final test BCE/pixel = 0.255

# Martens 2010 small-Gaussian init (N(0, 0.01^2))
HF       (150 steps, ~130s wall): final test BCE/pixel = 0.335
SGD+mom (2000 steps, ~ 35s wall): final test BCE/pixel = 0.255
```

HF reaches 0.282 transiently at step 50 of the default-init run, but the line
search starts failing afterwards and the BCE drifts back up. **HF does not
beat SGD on this CPU-budget setup under either init**. SGD-with-momentum
descends fast enough that it escapes the vanishing-gradient regime even with
`N(0, 0.01^2)` weights, leaving no room for HF to win.

Per the C1 decision rule we ran the C1 debugging gate
([sanity_check_gnvp.py](C1_autoencoder/sanity_check_gnvp.py)). HF's
curvature operators agree with the autograd ground truth to single-precision
tolerance, so the lack of a published-style HF-beats-SGD result is an
experimental-regime issue (CPU budget too small, no RBM pre-training, SGD
escaping the basin too quickly), not a library bug. We treat HF's curvature
operator as trustworthy and proceed to C2.

Training commands and plots:
[c1_autoencoder_default_init.png](C1_autoencoder/c1_autoencoder_default_init.png),
[c1_autoencoder_martens_init.png](C1_autoencoder/c1_autoencoder_martens_init.png).

## C2 — HF on our Phase 5 anchor

Full details in [C2_README.md](C2_README.md). The anchor is
[SequenceOfDenseBlocks](../../src/hessian.py#L349) at
`num_layers=8, hidden_dim=24, image_size=16, activation=relu`, rebuilt as a
plain `nn.Sequential` of `Linear+ReLU` with the same parameter count
(22,128) so BackPACK can attach to it. Training uses full CIFAR-10 at batch
size 64, matching [src/train_newton.py](../../src/train_newton.py) defaults.

### Headlines

Best `probe_loss` in the stable window of each run (NaN-stopped before 1000
steps):

| run                    | best probe_loss | step | NaN at step |
| ---------------------- | --------------- | ---- | ----------- |
| HF-GGN, damping=1.0    | **2.189**       |  90  | 188         |
| HF-Hessian, damping=1.0| **2.086**       | 253  | 447         |
| Our Newton (ref)       | 2.22 - 2.30     |  -   | -           |
| SGD (ref)              | 1.97            |  -   | -           |

Fixed-batch memorization (`--fixed-batch`, single 64-sample batch held for
1000 steps; protocol matches [exp-048b](../../experiments/runs/)):

| run             | min training loss on fixed batch | step |
| --------------- | -------------------------------- | ---- |
| HF-GGN          | **8.4e-8** (effectively zero)    | 106  |
| HF-Hessian      | **0.21**                         | 247  |
| Our Newton ref  | 1.16                             | -    |
| SGD ref         | 0.19                             | -    |

### Decision-rule readout (Section 5 of [report.md](../report.md))

The C2 row of the plan's decision table reads:

| Phase outcome                                  | Diagnosis |
| ---------------------------------------------- | --------- |
| HF-GGN succeeds, HF-raw-Hessian doesn't        | Damping   |
| Both HF modes succeed                          | Bug       |
| Both HF modes fail like our Newton             | Scale     |

What we see:

- **Both HF modes succeed on the fixed-batch diagnostic**, where our Newton
  stalls at 1.16. HF-GGN reaches single-precision zero, HF-Hessian reaches
  0.21 (right at SGD's reference floor). This is the unambiguous "Both HF
  modes succeed" row.
- On the full anchor, both HF modes reach the Newton plateau region and
  HF-Hessian crosses below it (2.086 < our Newton's 2.22-2.30) before
  diverging. Neither mode reaches SGD's 1.97 in the stable window, but the
  raw-Hessian mode does at least as well as the GGN mode, so there is **no
  Damping signal** in the C2 data.

**Lean: Bug.** A published full-Hessian-vector-product method, configured
to use the same `H` we use, drives the same model on the same batch to
loss `≈ 0`. Our linear-inverse algorithm stalls at `1.16` on identical input.
The likely explanations are:

1. A wrong sign, transposition, or damping placement in the augmented block-
   tri-diagonal system in
   [hessian_inverse_product](../../src/hessian.py#L287).
2. A subtle issue in the
   [block_partitioned_matrices](../../src/block_partitioned_matrices.py)
   solver that only shows up at the depth of our anchor model.

The fact that HF-GGN reaches `8e-8` an order of magnitude faster than HF-
Hessian reaches `0.21` is the only Gauss-Newton signal in the data, and it is
much weaker than the Bug-pointing signal of "raw-Hessian HF beats our Newton
by a factor of ~5.5 on the same fixed batch".

## Constraints and known artifacts

- All runs are CPU-only. The HF library blows up to NaN inside the LM damping
  loop before the 1000-step budget on the full anchor; we catch the first
  NaN in [train_hf_anchor.py](C2_anchor/train_hf_anchor.py) and stop. A
  higher-damping sweep (`--damping 10.0 --cg-max-iter 30`) extends stable
  training only marginally. The fixed-batch runs are stable for the full
  1000 steps.
- The HF library's `cg_efficient_backtracking` raises
  `UnboundLocalError: cannot access local variable 'best_iter'` when CG
  terminates with no candidate iterates, which happens regularly on our
  small 22k-parameter anchor. We disable backtracking
  (`use_cg_backtracking=False`) in [train_hf_anchor.py](C2_anchor/train_hf_anchor.py)
  to work around it. The final CG iterate is used directly.
- For C1 we use the default `use_cg_backtracking=True` in
  [train_autoencoder.py](C1_autoencoder/train_autoencoder.py); the larger
  2.8M-parameter autoencoder does not hit the empty-CG-iterates edge case.

## Files

```
phase-c-hessian-free/
  README.md                this file
  C1_README.md             C1 autoencoder details
  C2_README.md             C2 anchor details
  C1_autoencoder/
    train_autoencoder.py
    sanity_check_gnvp.py
    plot_c1.py
    c1_autoencoder_*.png
    logs/{hf,sgd_momentum,hf_martens_init,sgd_martens_init}/
  C2_anchor/
    train_hf_anchor.py
    plot_c2.py
    c2_*.png
    logs/{hf_ggn_full,hf_hessian_full,hf_*_fixedbatch,hf_*_damp10}/
  third_party/PyTorchHessianFree/   editable HF install
```
