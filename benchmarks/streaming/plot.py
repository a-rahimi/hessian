"""Plot the streaming results, one row of panels per activation.

Reads the CSVs written by run.py and produces results/streaming_curves.png:
held-out probe loss vs iteration (left) and vs wall-clock (right), both
log-scaled. Each activation gets its own row, matching the memorization figure,
because with three methods crossed with three activations a single pair of panels
puts nine curves on top of each other and the methods stop being separable.
Colour therefore distinguishes the methods, and the rows separate the
activations. The y axis is shared across rows so the activations remain
comparable to each other.

    python benchmarks/streaming/plot.py
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

sys.path.insert(0, str(SCRIPT_DIR))
from experiments import CONFIGS  # noqa: E402

# gelu is the only activation here, so distinguish the runs by color.
METHOD_COLOR = {
    "trust-region": "#0072B2",
    "trust-region-ggn": "#009E73",
    "sgd": "#D55E00",
}

RANDOM_GUESS_LOSS = math.log(10)


def load_csv(name: str) -> dict[str, list[float]]:
    path = RESULTS_DIR / f"{name}.csv"
    cols: dict[str, list[float]] = {"step": [], "probe_loss": [], "wall_clock_s": []}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cols["step"].append(float(row["step"]))
            cols["probe_loss"].append(float(row["probe_loss"]))
            cols["wall_clock_s"].append(float(row["wall_clock_s"]))
    return cols


def build_figure() -> plt.Figure:
    """Build the per-activation probe-loss figure and return it (for inline display)."""
    # dict.fromkeys keeps the order the configs declare rather than sorting.
    activations = list(dict.fromkeys(config.activation for config in CONFIGS))
    fig, axes = plt.subplots(
        len(activations),
        2,
        figsize=(12, 3.6 * len(activations)),
        sharex="col",
        sharey=True,
        squeeze=False,
    )

    for row, activation in zip(axes, activations):
        ax_iter, ax_time = row
        for config in (c for c in CONFIGS if c.activation == activation):
            data = load_csv(config.name)
            color = METHOD_COLOR[config.method]
            ax_iter.plot(
                data["step"], data["probe_loss"], color=color, lw=2, label=config.method
            )
            ax_time.plot(
                data["wall_clock_s"],
                data["probe_loss"],
                color=color,
                lw=2,
                label=config.method,
            )

        for ax in (ax_iter, ax_time):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.axhline(RANDOM_GUESS_LOSS, ls=":", color="gray", alpha=0.5, lw=1)
            ax.grid(True, which="both", alpha=0.15)

        ax_iter.set_ylabel(f"{activation}\nheld-out probe loss")
        ax_iter.legend(fontsize=8, loc="lower left")

    axes[0][0].set_title("Probe loss vs iteration")
    axes[0][1].set_title("Probe loss vs wall-clock")
    axes[-1][0].set_xlabel("iteration")
    axes[-1][1].set_xlabel("wall-clock (s)")

    fig.suptitle(
        "Streaming fresh minibatches: width 8 x depth 16 MLP, batch 32"
    )
    fig.tight_layout()
    return fig


def save_figure(out: Path | None = None) -> Path:
    out = out or (RESULTS_DIR / "streaming_curves.png")
    build_figure().savefig(out, dpi=130)
    return out


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    print(f"wrote {save_figure()}")
