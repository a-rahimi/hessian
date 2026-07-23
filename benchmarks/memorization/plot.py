"""Plot the fixed-batch memorization results as a two-panel figure.

Reads the six CSVs written by run.py and produces
results/memorization_curves.png: loss vs iteration (left) and loss vs
wall-clock (right), both log-scaled.

    python benchmarks/memorization/plot.py
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

# Okabe-Ito colorblind-safe palette: activation -> color.
ACT_COLOR = {"tanh": "#0072B2", "gelu": "#009E73", "relu": "#D55E00"}
# Method -> linestyle.
METHOD_STYLE = {"trust-region": "-", "sgd": "--"}

LOSS_FLOOR = 1e-3
RANDOM_GUESS_LOSS = math.log(10)  # ln(10) for uniform 10-class prediction


def load_csv(name: str) -> dict[str, list[float]]:
    path = RESULTS_DIR / f"{name}.csv"
    cols: dict[str, list[float]] = {"step": [], "loss": [], "wall_clock_s": []}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cols["step"].append(float(row["step"]))
            cols["loss"].append(max(float(row["loss"]), LOSS_FLOOR))
            cols["wall_clock_s"].append(float(row["wall_clock_s"]))
    return cols


def build_figure() -> plt.Figure:
    """Build the two-panel loss figure and return it (for inline display)."""
    fig, (ax_iter, ax_time) = plt.subplots(1, 2, figsize=(12, 5))

    for config in CONFIGS:
        data = load_csv(config.name)
        color = ACT_COLOR[config.activation]
        style = METHOD_STYLE[config.method]
        label = f"{config.method} / {config.activation}"
        ax_iter.plot(data["step"], data["loss"], style, color=color, lw=2, label=label)
        ax_time.plot(
            data["wall_clock_s"], data["loss"], style, color=color, lw=2, label=label
        )

    for ax in (ax_iter, ax_time):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("training loss (floored at 1e-3)")
        ax.axhline(RANDOM_GUESS_LOSS, ls=":", color="gray", alpha=0.5, lw=1)
        ax.grid(True, which="both", alpha=0.15)

    ax_iter.set_xlabel("iteration")
    ax_iter.set_title("Loss vs iteration")
    ax_time.set_xlabel("wall-clock (s)")
    ax_time.set_title("Loss vs wall-clock")
    ax_iter.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "Fixed-batch memorization: width 8 x depth 16 MLP, 32-example fixed batch"
    )
    fig.tight_layout()
    return fig


def save_figure(out: Path | None = None) -> Path:
    out = out or (RESULTS_DIR / "memorization_curves.png")
    build_figure().savefig(out, dpi=130)
    return out


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    print(f"wrote {save_figure()}")
