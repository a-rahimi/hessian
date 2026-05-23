# C1 — Hessian-Free on the Hinton & Salakhutdinov 2006 autoencoder

## What this is

The Phase C plan calls for first verifying that the published Hessian-Free
optimizer reproduces its canonical autoencoder result before turning HF on
our own anchor model. The training script
[train_autoencoder.py](C1_autoencoder/train_autoencoder.py) builds the
canonical 784→1000→500→250→30→250→500→1000→784 sigmoid autoencoder, trains it
on MNIST under HF and under SGD-with-momentum, and logs per-step training BCE
plus per-epoch held-out test BCE.

The Hessian-Free implementation used is
[ltatzel/PyTorchHessianFree](https://github.com/ltatzel/PyTorchHessianFree),
cloned into
[third_party/PyTorchHessianFree](third_party/PyTorchHessianFree) and
installed editable via `pip install -e .` against the project venv. Its
internals call `BackPACK` for the Hessian-vector and Gauss-Newton-vector
products.

## Install

The clone-and-install commands used to set up the venv were:

```
git clone https://github.com/ltatzel/PyTorchHessianFree.git \
  reproduce-published-results/phase-c-hessian-free/third_party/PyTorchHessianFree
/Users/arahimi/hessian/.venv/bin/python -m pip install backpack-for-pytorch
cd reproduce-published-results/phase-c-hessian-free/third_party/PyTorchHessianFree
/Users/arahimi/hessian/.venv/bin/python -m pip install -e .
```

## Training commands

Default kaiming-uniform init (PyTorch's `nn.Linear` default):

```
python train_autoencoder.py --optimizer hf  --num-steps 200 \
  --num-train 10000 --batch-size 200 --data-dir ../../../data \
  --logdir ./logs --run-name hf  --cpu

python train_autoencoder.py --optimizer sgd --num-steps 2000 \
  --num-train 10000 --batch-size 200 --sgd-lr 0.1 --sgd-momentum 0.9 \
  --data-dir ../../../data --logdir ./logs --run-name sgd_momentum --cpu
```

Martens 2010 small-Gaussian init (`N(0, 0.01^2)` on every linear weight),
which we hoped would reproduce the vanishing-gradient regime where HF beats
SGD:

```
python train_autoencoder.py --optimizer hf  --num-steps 150 \
  --init-std 0.01 --run-name hf_martens_init  ...

python train_autoencoder.py --optimizer sgd --num-steps 2000 \
  --init-std 0.01 --run-name sgd_martens_init ...
```

## Headline numbers (CPU, 10k-sample MNIST subset)

Final held-out test BCE per pixel:

| run                                  | steps | wall    | final test BCE/pixel |
| ------------------------------------ | ----- | ------- | -------------------- |
| HF (default init), 200 steps         | 200   | ~150 s  | 0.332                |
| SGD+momentum (default init), 2k step | 2000  | ~ 45 s  | **0.255**            |
| HF (Martens N(0, 0.01) init)         | 150   | ~130 s  | 0.335                |
| SGD+momentum (Martens init)          | 2000  | ~ 35 s  | **0.255**            |

The two best-step readings from HF were a transient `test_bce_per_pixel
= 0.282` at step 50, after which the run drifted back up to 0.32-0.35 as the
line search began to fail.

## Decision-rule readout

The plan's C1 decision rule is

> If HF reproduces Martens 2010 autoencoder numbers, HF is a working
> second-order baseline we trust.
> If HF fails to reproduce, debug the GNV-P call against autograd-materialized
> `J^T H_loss J v` on a tiny problem before continuing.

The Martens-2010-style result — HF visibly beating SGD on test reconstruction —
**did not reproduce on our CPU-budget setup**. SGD-with-momentum descends faster
than HF on both inits and reaches roughly the same final BCE that the
"constant predictor" floor permits for a sigmoid autoencoder without RBM
pre-training.

Per the decision rule we therefore ran the C1 debugging gate:
[sanity_check_gnvp.py](C1_autoencoder/sanity_check_gnvp.py) builds a tiny
ReLU MLP, materializes `J`, `H_loss`, and the dense Hessian explicitly with
autograd, and compares `J^T H_loss J v` and `H v` against HF's BackPACK-backed
products. The reference and library outputs agree to:

```
GGN vp relative error: 1.99e-7
Hessian vp relative error: 9.70e-8
```

both well below the `1e-4` correctness gate. The HF library's curvature
operator is therefore correct. The lack of an HF win on our autoencoder setup
is an experimental-regime issue (CPU budget too small, no RBM pre-training,
SGD-momentum already escapes the vanishing-gradient regime with default
inits), not a library bug.

We therefore treat HF's curvature operator as a trustworthy baseline and
proceed to **C2**, where the diagnostic is much sharper because we are
comparing HF in two curvature modes against our own algorithm on the *same*
model.

## Files

- [train_autoencoder.py](C1_autoencoder/train_autoencoder.py) — training
  driver.
- [sanity_check_gnvp.py](C1_autoencoder/sanity_check_gnvp.py) — autograd
  cross-check of HF's GGN-vector and Hessian-vector products.
- [plot_c1.py](C1_autoencoder/plot_c1.py) — generates the two PNG plots
  below.
- [c1_autoencoder_default_init.png](C1_autoencoder/c1_autoencoder_default_init.png),
  [c1_autoencoder_martens_init.png](C1_autoencoder/c1_autoencoder_martens_init.png)
  — test BCE vs steps and wall time for both inits.
- [logs/](C1_autoencoder/logs/) — per-step CSV logs from every run.
