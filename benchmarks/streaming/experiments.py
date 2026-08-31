"""Single source of truth for the streaming benchmark configs.

Both run.py and plot.py import CONFIGS from here. This is the sibling of the
memorization benchmark: the model, activation, and optimizer settings are the
ones that worked best there (gelu, and the tuned trust-region radius), but a
fresh minibatch is drawn every step instead of reusing one, so the runs are
actually training on CIFAR-10 rather than memorizing a single batch.
"""

from __future__ import annotations

import dataclasses

# Same tall-skinny MLP as the memorization benchmark, minus --reuse-batch: with
# the default reuse of 1 a new minibatch is drawn every step.
SHARED_ARGS = [
    "--image-size", "8",
    "--hidden-dim", "8",
    "--num-layers", "16",
    "--batch-size", "32",
    "--seed", "0",
    "--data-dir", "./data",
]

ACTIVATION = "gelu"


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    method: str
    activation: str
    args: list[str]


TR = Config(
    "tr_gelu",
    "trust-region",
    ACTIVATION,
    [
        "--mode", "trust-region",
        "--tr-solver", "dense",
        "--delta-init", "1.0",
        "--delta-max", "100",
        "--tr-eta", "0.1",
        "--num-steps", "400",
        "--activation", ACTIVATION,
        *SHARED_ARGS,
    ],
)

SGD = Config(
    "sgd_gelu",
    "sgd",
    ACTIVATION,
    [
        "--mode", "sgd",
        "--lr", "0.1",
        "--num-steps", "30000",
        "--activation", ACTIVATION,
        *SHARED_ARGS,
    ],
)

CONFIGS = [TR, SGD]
CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}
