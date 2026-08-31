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


# Everything the two trust-region runs share, so that the only thing separating
# them is which curvature matrix the subproblem is built on.
TR_ARGS = [
    "--mode", "trust-region",
    "--tr-solver", "dense",
    "--delta-init", "1.0",
    "--delta-max", "100",
    "--tr-eta", "0.1",
    "--num-steps", "400",
    "--activation", ACTIVATION,
    *SHARED_ARGS,
]

TR = Config(
    "tr_gelu",
    "trust-region",
    ACTIVATION,
    [*TR_ARGS, "--curvature", "hessian"],
)

# The same trust region on the Gauss-Newton matrix instead of the Hessian. G
# drops the network's own curvature, which is what makes the Hessian indefinite,
# so the subproblem is solved on a positive semidefinite matrix.
GGN = Config(
    "ggn_gelu",
    "trust-region-ggn",
    ACTIVATION,
    [*TR_ARGS, "--curvature", "ggn"],
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

CONFIGS = [TR, GGN, SGD]
CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}
