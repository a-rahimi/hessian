"""Plot trust-region vs SGD loss curves from tensorboard event files.

Produces two figures per run pair:
  loss_vs_step.png   — training loss and probe_loss vs step index
  loss_vs_time.png   — same curves but x-axis is wall_clock_s

Usage:
    python scripts/plot_tr_vs_sgd.py --tr-run runs/trust-region-<ts> --sgd-run runs/sgd-<ts>

Output is written next to this script (or to --outdir if specified).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str, tags: list[str]) -> dict[str, dict]:
    """Return {tag: {"steps": [...], "times": [...], "values": [...]}} for each tag."""
    ea = EventAccumulator(logdir)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag not in available:
            print(f"  warning: tag '{tag}' not found in {logdir}")
            result[tag] = {"steps": [], "times": [], "values": []}
            continue
        events = ea.Scalars(tag)
        result[tag] = {
            "steps": [e.step for e in events],
            "times": [e.wall_time for e in events],
            "values": [e.value for e in events],
        }
    # Normalize wall times to seconds from first event
    for tag, data in result.items():
        if data["times"]:
            t0 = data["times"][0]
            data["times"] = [t - t0 for t in data["times"]]
    return result


def plot_pair(
    tr_data: dict,
    sgd_data: dict,
    x_key: str,
    x_label: str,
    outpath: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    for ax, (loss_tag, ax_title) in zip(axes, [("loss", "Training loss"), ("probe_loss", "Probe loss")]):
        for label, data, color in [
            ("trust-region", tr_data, "tab:blue"),
            ("sgd", sgd_data, "tab:orange"),
        ]:
            d = data.get(loss_tag, {})
            xs = d.get(x_key, [])
            ys = d.get("values", [])
            if xs and ys:
                ax.plot(xs, ys, label=label, color=color, linewidth=1.2, alpha=0.85)
        ax.set_xlabel(x_label)
        ax.set_ylabel("loss")
        ax.set_title(ax_title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"saved {outpath}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tr-run", required=True, help="Path to trust-region tensorboard logdir.")
    p.add_argument("--sgd-run", required=True, help="Path to SGD tensorboard logdir.")
    p.add_argument("--outdir", default=None, help="Output directory for PNGs. Defaults to --tr-run.")
    args = p.parse_args()

    outdir = Path(args.outdir) if args.outdir else Path(args.tr_run)
    outdir.mkdir(parents=True, exist_ok=True)

    tags = ["loss", "probe_loss"]
    print(f"loading TR run: {args.tr_run}")
    tr_data = load_scalars(args.tr_run, tags)
    print(f"loading SGD run: {args.sgd_run}")
    sgd_data = load_scalars(args.sgd_run, tags)

    plot_pair(
        tr_data, sgd_data,
        x_key="steps", x_label="step",
        outpath=outdir / "loss_vs_step.png",
        title="Loss vs step",
    )
    plot_pair(
        tr_data, sgd_data,
        x_key="times", x_label="wall clock (s)",
        outpath=outdir / "loss_vs_time.png",
        title="Loss vs wall clock time",
    )


if __name__ == "__main__":
    main()
