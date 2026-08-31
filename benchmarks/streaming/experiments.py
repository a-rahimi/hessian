"""Single source of truth for the streaming benchmark configs.

Both run.py and plot.py import CONFIGS from here. This is the sibling of the
memorization benchmark, crossing the same three methods with the same three
activations, and the trust-region settings are the ones that worked best there.
The difference is that a fresh minibatch is drawn every step instead of reusing
one, so the runs are actually training on CIFAR-10 rather than memorizing a
single batch.
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

ACTIVATIONS = ["tanh", "gelu", "relu"]

# Best SGD learning rate per activation, swept on this streaming objective rather
# than carried over from memorization, because the best rate differs between the
# two: gelu wants 0.1 here against 0.03 there.
SGD_LR = {"tanh": "0.03", "gelu": "0.1", "relu": "0.1"}


@dataclasses.dataclass(frozen=True)
class Config:
    name: str
    method: str
    activation: str
    args: list[str]


def _trust_region_args(activation: str) -> list[str]:
    """The arguments the two trust-region runs share.

    They differ only in `--curvature`, so anything separating their curves is the
    curvature matrix rather than the optimizer.
    """
    return [
        "--mode", "trust-region",
        "--tr-solver", "dense",
        "--delta-init", "1.0",
        "--delta-max", "100",
        "--tr-eta", "0.1",
        "--num-steps", "400",
        "--activation", activation,
        *SHARED_ARGS,
    ]


def _trust_region(activation: str) -> Config:
    args = [*_trust_region_args(activation), "--curvature", "hessian"]
    return Config(f"tr_{activation}", "trust-region", activation, args)


def _gauss_newton(activation: str) -> Config:
    """The same trust region on the Gauss-Newton matrix instead of the Hessian.

    G drops the network's own curvature, which is the term that makes the Hessian
    indefinite, so the subproblem is built on a positive semidefinite matrix.
    """
    args = [*_trust_region_args(activation), "--curvature", "ggn"]
    return Config(f"ggn_{activation}", "trust-region-ggn", activation, args)


def _sgd(activation: str) -> Config:
    args = [
        "--mode", "sgd",
        "--lr", SGD_LR[activation],
        "--num-steps", "30000",
        "--activation", activation,
        *SHARED_ARGS,
    ]
    return Config(f"sgd_{activation}", "sgd", activation, args)


CONFIGS = (
    [_trust_region(a) for a in ACTIVATIONS]
    + [_gauss_newton(a) for a in ACTIVATIONS]
    + [_sgd(a) for a in ACTIVATIONS]
)

CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}
