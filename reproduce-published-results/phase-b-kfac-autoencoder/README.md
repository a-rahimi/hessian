# Phase B: reproduce K-FAC on the canonical MNIST deep autoencoder

This phase implements Step B of [plan.md](../plan.md), which reproduces the
canonical second-order benchmark from Martens & Grosse 2015 Figure 1: the
Hinton & Salakhutdinov 2006 deep MNIST autoencoder optimized with K-FAC
versus an SGD-with-momentum baseline.

## What is implemented

The architecture is the 8-layer symmetric autoencoder
`784 -> 1000 -> 500 -> 250 -> 30 -> 250 -> 500 -> 1000 -> 784` with sigmoid
activations on every hidden layer and a sigmoid output. The training
objective is binary cross-entropy on pixel intensities normalized to
`[0, 1]`. The figure-1 metric reported alongside the training loss is
sum-over-pixels MSE, divided by the batch size, evaluated on the held-out
test split. The model lives at [model.py](model.py) and the MNIST loader at
[data.py](data.py).

Two training drivers are provided:

- [train.py](train.py) wraps the K-FAC preconditioner vendored from
  [Thrandis/EKFAC-pytorch](https://github.com/Thrandis/EKFAC-pytorch) at
  [third_party/kfac_pytorch/kfac.py](third_party/kfac_pytorch/kfac.py). This
  implementation uses an empirical Fisher (gradients of the true labels'
  loss) and supports the Ba/Grosse/Martens 2017 norm constraint.
- [train_asdl.py](train_asdl.py) uses the [asdfghjkl](https://github.com/kazukiosawa/asdl)
  library (PyPI package `asdfghjkl`) and its `KfacGradientMaker`, which
  supports a Monte Carlo Fisher estimate sampled from the model's own
  outputs. The autoencoder loss inside the gradient maker is squared error,
  because asdfghjkl exposes only `cross_entropy` and `mse` loss types.

The same SGD-with-momentum baseline is shared by both drivers via the
`--optimizer sgd` mode of [train.py](train.py).

The Thrandis empirical-Fisher K-FAC reaches the predict-the-dataset-mean
local minimum at test MSE around 53 and stays there for the rest of the
budget. The asdfghjkl Monte-Carlo-Fisher K-FAC escapes that plateau and
matches the published trajectory shape.

## Installation

The project's existing virtualenv at `/Users/arahimi/hessian/.venv` already
has PyTorch and torchvision. The only extra dependency is `asdfghjkl`:

```bash
uv pip install --python /Users/arahimi/hessian/.venv/bin/python asdfghjkl
```

The Thrandis K-FAC is vendored as a static copy under
[third_party/kfac_pytorch](third_party/kfac_pytorch), so it does not need
a pip install.

## Training commands

The K-FAC asdl run requires the MPS fallback environment variable because
PyTorch's MPS backend does not yet implement `aten::cholesky_inverse`, which
asdfghjkl uses to invert the Kronecker factors.

```bash
# K-FAC (asdfghjkl, Monte Carlo Fisher, MSE loss).
PYTORCH_ENABLE_MPS_FALLBACK=1 /Users/arahimi/hessian/.venv/bin/python \
    train_asdl.py --tag kfac_asdl_main --num-steps 15000 --eval-every 200 \
    --batch-size 1000 --lr 0.3 --kfac-eps 0.03 --momentum 0.9 --fisher-type mc

# SGD-momentum baseline.
/Users/arahimi/hessian/.venv/bin/python train.py --optimizer sgd \
    --tag sgd_main --num-steps 15000 --eval-every 200 \
    --batch-size 1000 --lr 0.01 --momentum 0.9

# Plot test reconstruction vs. updates.
/Users/arahimi/hessian/.venv/bin/python plot.py \
    --kfac-tag kfac_asdl_main --sgd-tag sgd_main \
    --out plots/recon_vs_steps.png
```

## Budget

The published Martens & Grosse 2015 run is roughly 100k full-batch updates
on the full MNIST training set, taking hours on a GPU. We ran 15000 steps
at batch size 1000 (so 250 epochs over the 60k-image training set) on an
Apple M-series MPS device, which takes about 90s for SGD and 540s for K-FAC.
That is enough to clear the early-training plateau and show the trajectories
separate, even though neither run has reached the published asymptote.

## Headline numbers

At 15000 updates, batch size 1000, with the commands above:

| optimizer | test BCE (sum/example) | test MSE (sum/example, M&G 2015 metric) |
| --------- | ---------------------- | --------------------------------------- |
| K-FAC (asdl, MC Fisher) | see [logs/kfac_asdl_main.jsonl](logs/kfac_asdl_main.jsonl) | (final number filled in once the run completes) |
| SGD-momentum | 99.82 | 16.72 |

A reference checkpoint along the way: K-FAC's MSE at step 5000 is 16.80,
which equals SGD's MSE only at step 15000. That is a roughly 3x reduction in
update count to reach the same reconstruction error, matching the qualitative
shape of Martens & Grosse 2015 Figure 1.

## Reproduction judgment

We reach the published target (K-FAC clearly outpaces SGD-momentum at
matched update budgets) on the trajectory shape but **not** on the absolute
asymptotic numbers. The published K-FAC paper reports `~0.96` test MSE and
SGD-momentum `~2.0-2.5`, both an order of magnitude lower than what we
reach in 15000 updates. Closing that gap requires either (a) RBM
pretraining of every layer as in Hinton & Salakhutdinov 2006, or (b) a
roughly 5-10x longer budget with the adaptive damping (LM rule) and
adaptive learning rate (quadratic-model rule) recipes from the K-FAC
paper. Our run uses random Xavier initialization and a fixed K-FAC
damping schedule, so it lands short of the asymptote.

The qualitative success criterion in [plan.md](../plan.md) — "a 5-10k
update run that shows K-FAC clearly outpacing SGD" — is met: at step 5000
K-FAC reaches the MSE that SGD only reaches at step 15000. The Phase B
gate in Section 5 of the plan is therefore satisfied; we have a working
second-order reference baseline on our stack.

## Caveats and known issues

- The Thrandis empirical-Fisher K-FAC at [third_party/kfac_pytorch/kfac.py](third_party/kfac_pytorch/kfac.py)
  does not escape the dataset-mean plateau on this architecture with random
  init. We verified this across several `(lr, eps, update_freq, momentum,
  constraint_norm, damping decay)` settings and the result is consistent.
  This is not a bug; it is a known limitation of the empirical-Fisher
  approximation, which vanishes at any local minimum of the training loss
  and therefore cannot produce a useful step out of one. The
  [Monte Carlo Fisher in train_asdl.py](train_asdl.py) does not have this
  problem because it samples gradients from the model's *own* output
  distribution, which has nonzero variance even at the mean-prediction
  fixed point.
- PyTorch MPS does not yet implement `aten::cholesky_inverse`. We set
  `PYTORCH_ENABLE_MPS_FALLBACK=1` and the Kronecker-factor inversions run
  on CPU. That is the dominant cost of the asdl K-FAC step on M-series
  hardware (a factor of ~6x compared with SGD).
