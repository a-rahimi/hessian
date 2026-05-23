"""Plot per-step loss and probe loss for the Phase A runs.

Reads metrics.csv files emitted by [subsampled_newton.py](./subsampled_newton.py)
and produces simple matplotlib figures under each run directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k, v in r.items():
                rows.setdefault(k, []).append(float(v))
    return rows


def plot_run(csv_path: Path, out_path: Path, title: str) -> None:
    rows = load_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(rows["step"], rows["loss"], label="loss", linewidth=0.8)
    if "train_loss_avg10" in rows:
        axes[0].plot(
            rows["step"], rows["train_loss_avg10"], label="train_loss_avg10",
            linewidth=1.2,
        )
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("training loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title(title + " - training loss")

    axes[1].plot(rows["step"], rows["probe_loss"], label="probe_loss")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("probe loss")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title(title + " - probe loss")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_paths", nargs="+", help="paths to metrics.csv files")
    p.add_argument("--out-dir", default=None,
                   help="output directory; defaults to each csv's parent")
    args = p.parse_args()

    for s in args.csv_paths:
        csv_path = Path(s)
        out_dir = Path(args.out_dir) if args.out_dir else csv_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "trajectory.png"
        title = csv_path.parent.name
        plot_run(csv_path, out_path, title)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
