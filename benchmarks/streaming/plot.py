"""Plot the streaming results as a two-panel figure.

Reads the CSVs written by run.py and produces results/streaming_curves.png:
held-out probe loss vs iteration (left) and vs wall-clock (right), both
log-scaled.

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

# gelu is the only activation here, so distinguish the two runs by color.
METHOD_COLOR = {"trust-region": "#0072B2", "sgd": "#D55E00"}

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
    """Build the two-panel probe-loss figure and return it (for inline display)."""
    fig, (ax_iter, ax_time) = plt.subplots(1, 2, figsize=(12, 5))

    for config in CONFIGS:
        data = load_csv(config.name)
        color = METHOD_COLOR[config.method]
        label = f"{config.method} / {config.activation}"
        ax_iter.plot(data["step"], data["probe_loss"], color=color, lw=2, label=label)
        ax_time.plot(
            data["wall_clock_s"], data["probe_loss"], color=color, lw=2, label=label
        )

    for ax in (ax_iter, ax_time):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("held-out probe loss")
        ax.axhline(RANDOM_GUESS_LOSS, ls=":", color="gray", alpha=0.5, lw=1)
        ax.grid(True, which="both", alpha=0.15)

    ax_iter.set_xlabel("iteration")
    ax_iter.set_title("Probe loss vs iteration")
    ax_time.set_xlabel("wall-clock (s)")
    ax_time.set_title("Probe loss vs wall-clock")
    ax_iter.legend(fontsize=9, loc="lower left")

    fig.suptitle(
        "Streaming fresh minibatches: width 8 x depth 16 MLP, gelu, batch 32"
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
