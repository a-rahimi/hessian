"""Plot the curvature-batch experiment.

Two panels for the seed-0 runs on each activation, and a third overlaying all
three gelu seeds. The third is the one that matters, because the seed-0 gap does
not survive replication.

    python experiments/curvature-batch/plot.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import harness  # noqa: E402
from experiments import CONFIGS, SEEDS  # noqa: E402

# Okabe-Ito colorblind-safe palette: arm -> color.
ARM_COLOR = {"same": "#0072B2", "fresh": "#D55E00"}
SEED_ALPHA = {0: 1.0, 1: 0.5, 2: 0.3}
RANDOM_GUESS_LOSS = math.log(10)


def by_name(name: str):
    return harness.load_csv(RESULTS_DIR, name)


def decorate(ax, title: str) -> None:
    ax.axhline(RANDOM_GUESS_LOSS, ls=":", color="gray", lw=1, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("iteration")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=8, loc="upper right")


def build_figure() -> plt.Figure:
    """Build the three-panel figure and return it (for inline display)."""
    fig, (ax_gelu, ax_tanh, ax_seeds) = plt.subplots(1, 3, figsize=(15, 4.6))

    for ax, activation in ((ax_gelu, "gelu"), (ax_tanh, "tanh")):
        for arm, color in ARM_COLOR.items():
            data = by_name(f"{activation}_seed0_{arm}")
            ax.plot(data["step"], data["probe_loss"], color=color, lw=1.6,
                    label=f"curvature batch: {arm}")
        decorate(ax, f"{activation}, seed 0")
    ax_gelu.set_ylabel("held-out probe loss")

    for seed in SEEDS["gelu"]:
        for arm, color in ARM_COLOR.items():
            data = by_name(f"gelu_seed{seed}_{arm}")
            ax_seeds.plot(data["step"], data["probe_loss"], color=color, lw=1.4,
                          alpha=SEED_ALPHA[seed], label=f"{arm}, seed {seed}")
    decorate(ax_seeds, "gelu, all three seeds")

    fig.suptitle(
        "Trust-region curvature estimated on a batch of its own "
        "(gradient and accept/reject stay on the original batch); "
        "dotted line is the random-guess loss"
    )
    fig.tight_layout()
    return fig


def save_figure(out: Path | None = None) -> Path:
    out = out or (RESULTS_DIR / "curvature_batch.png")
    build_figure().savefig(out, dpi=130)
    return out


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    print(f"wrote {save_figure()}")
    print(f"\n{'run':22s} {'best probe':>11s} {'best acc':>9s}")
    for config in CONFIGS:
        data = by_name(config.name)
        print(f"{config.name:22s} {min(data['probe_loss']):11.4f} "
              f"{max(data['probe_accuracy']):9.3f}")
