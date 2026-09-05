"""Single source of truth for the curvature-batch experiment (exp-059).

The trust region fits its quadratic model and then scores that model's accuracy
on the same minibatch, so the curvature is fit to exactly the samples the step is
judged on and the accept/reject ratio cannot tell a good step from one that
overfit the curvature. These runs estimate the Hessian on a batch of its own
while leaving the gradient, the trial loss, and the ratio on the original batch,
so the only thing that moves between the two arms is which samples the quadratic
term saw.

Both arms are otherwise the streaming experiment's trust-region config verbatim.
gelu is replicated across three seeds, because at seed 0 alone the gap looked far
larger than it turned out to be.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import Config  # noqa: E402

# The streaming experiment's model and data: a tall-skinny MLP drawing a fresh
# minibatch every step, so the curvature really is being re-estimated each time.
SHARED_ARGS = [
    "--image-size", "8",
    "--hidden-dim", "8",
    "--num-layers", "16",
    "--batch-size", "32",
    "--data-dir", "./data",
]

TR_ARGS = [
    "--mode", "trust-region",
    "--tr-solver", "dense",
    "--delta-init", "1.0",
    "--delta-max", "100",
    "--tr-eta", "0.1",
    "--num-steps", "400",
]

# `same` is the control, which is what the trust region has always done.
ARMS = ["same", "fresh"]

# gelu carries the seed replication; tanh is here because the streaming trust
# region ends worse than the random-guess loss on it, so it is the case where a
# curvature fit to the wrong samples should be doing the most damage.
SEEDS = {"gelu": [0, 1, 2], "tanh": [0]}


def _config(activation: str, seed: int, arm: str) -> Config:
    # `method` carries the arm, because the arm is what this experiment varies.
    return Config(
        name=f"{activation}_seed{seed}_{arm}",
        method=arm,
        activation=activation,
        args=[
            *TR_ARGS,
            "--activation", activation,
            "--curvature-batch", arm,
            "--seed", str(seed),
            *SHARED_ARGS,
        ],
    )


def _sgd(activation: str) -> Config:
    """SGD at the streaming experiment's tuned rate, as the reference line.

    Without it the figure only shows the two arms tracking each other, which says
    nothing about whether either is any good.
    """
    return Config(
        name=f"{activation}_seed0_sgd",
        method="sgd",
        activation=activation,
        args=[
            "--mode", "sgd",
            "--lr", {"gelu": "0.1", "tanh": "0.03"}[activation],
            "--num-steps", "30000",
            "--activation", activation,
            "--seed", "0",
            *SHARED_ARGS,
        ],
    )


CONFIGS = [
    _config(activation, seed, arm)
    for activation, seeds in SEEDS.items()
    for seed in seeds
    for arm in ARMS
] + [_sgd(activation) for activation in SEEDS]

CONFIGS_BY_NAME = {config.name: config for config in CONFIGS}
