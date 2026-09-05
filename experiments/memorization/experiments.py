"""Single source of truth for the fixed-batch memorization benchmark configs.

Both run.py and plot.py import CONFIGS from here so the six runs (method x
activation) stay in sync. Each config carries a stable `name`, its `method` and
`activation`, and the exact `train_newton.py` CLI args used to produce it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import Config  # noqa: E402

# Model/data config shared by every run: a tall-skinny MLP (width 8, depth 16)
# memorizing one fixed 32-example minibatch. --reuse-batch is set absurdly high
# so the same first minibatch is reused for the whole run.
SHARED_ARGS = [
    "--image-size", "8",
    "--hidden-dim", "8",
    "--num-layers", "16",
    "--batch-size", "32",
    "--seed", "0",
    "--reuse-batch", "1000000",
    "--data-dir", "./data",
]

ACTIVATIONS = ["tanh", "gelu", "relu"]

# Best SGD learning rate per activation (tuned separately from this suite).
SGD_LR = {"tanh": "0.01", "gelu": "0.03", "relu": "0.01"}


def _trust_region(activation: str) -> Config:
    args = [
        "--mode", "trust-region",
        "--tr-solver", "dense",
        "--delta-init", "1.0",
        "--delta-max", "100",
        "--tr-eta", "0.1",
        "--num-steps", "400",
        "--activation", activation,
        *SHARED_ARGS,
    ]
    return Config(f"tr_{activation}", "trust-region", activation, args)


def _sgd(activation: str) -> Config:
    args = [
        "--mode", "sgd",
        "--lr", SGD_LR[activation],
        "--num-steps", "30000",
        "--activation", activation,
        *SHARED_ARGS,
    ]
    return Config(f"sgd_{activation}", "sgd", activation, args)


CONFIGS = [_trust_region(a) for a in ACTIVATIONS] + [_sgd(a) for a in ACTIVATIONS]

CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}
