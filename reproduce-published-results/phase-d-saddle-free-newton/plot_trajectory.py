"""Plot loss / probe trajectories from an SFN run.

Takes the CSV produced by `extract_trajectory.py` and a list of (label, csv)
pairs, and writes a single 2-panel PNG: training loss on the left, probe
loss on the right, both versus step. We also draw horizontal reference
lines for the linear-Newton plateau and the SGD bar described in the plan.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_csv(path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {
        "step": [],
        "loss": [],
        "avg10": [],
        "probe": [],
    }
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows["step"].append(int(row["step"]))
            rows["loss"].append(float(row["loss"]))
            rows["avg10"].append(float(row["avg10"]))
            rows["probe"].append(float(row["probe"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label-csv", nargs="+", required=True,
                    help="Repeated 'label=path/to.csv' arguments.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--newton-plateau", type=float, default=None,
                    help="Horizontal reference line for our linear-Newton plateau (probe).")
    ap.add_argument("--sgd-bar", type=float, default=None,
                    help="Horizontal reference line for SGD (probe).")
    ap.add_argument("--newton-floor-loss", type=float, default=None,
                    help="Horizontal reference line on the training-loss panel (e.g. fixed-batch min loss).")
    ap.add_argument("--sgd-floor-loss", type=float, default=None,
                    help="Horizontal reference line on the training-loss panel for SGD's fixed-batch min loss.")
    args = ap.parse_args()

    fig, (ax_loss, ax_probe) = plt.subplots(1, 2, figsize=(11, 4.2))
    for spec in args.label_csv:
        label, path = spec.split("=", 1)
        data = load_csv(Path(path))
        ax_loss.plot(data["step"], data["avg10"], label=f"{label} avg10")
        ax_probe.plot(data["step"], data["probe"], label=label)

    if args.newton_floor_loss is not None:
        ax_loss.axhline(args.newton_floor_loss, color="gray", linestyle="--",
                        label=f"linear-Newton floor ({args.newton_floor_loss:.2f})")
    if args.sgd_floor_loss is not None:
        ax_loss.axhline(args.sgd_floor_loss, color="green", linestyle="--",
                        label=f"SGD floor ({args.sgd_floor_loss:.2f})")
    if args.newton_plateau is not None:
        ax_probe.axhline(args.newton_plateau, color="gray", linestyle="--",
                         label=f"linear-Newton plateau ({args.newton_plateau:.2f})")
    if args.sgd_bar is not None:
        ax_probe.axhline(args.sgd_bar, color="green", linestyle="--",
                         label=f"SGD ({args.sgd_bar:.2f})")

    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("train loss (trailing avg 10)")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(True, alpha=0.3)

    ax_probe.set_xlabel("step")
    ax_probe.set_ylabel("probe loss")
    ax_probe.legend(fontsize=8)
    ax_probe.grid(True, alpha=0.3)

    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
