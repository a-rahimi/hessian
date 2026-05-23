"""Plot C1 deep-autoencoder train and test reconstruction error vs steps and
wall time for HF and SGD-momentum, under the default kaiming-uniform init and
the Martens 2010 small-Gaussian init."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
LOGDIR = HERE / "logs"


def read_csv(path: Path) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    with open(path) as f:
        rdr = csv.reader(f)
        header = next(rdr)
        for h in header:
            cols[h] = []
        for row in rdr:
            for h, v in zip(header, row):
                try:
                    cols[h].append(float(v))
                except ValueError:
                    cols[h].append(float("nan"))
    return cols


def plot_run_set(runs: list[tuple[str, str, str]], suffix: str,
                 title: str) -> None:
    fig, (ax_step, ax_time) = plt.subplots(1, 2, figsize=(12, 4.5))
    for run, label, color in runs:
        ep = read_csv(LOGDIR / run / "epoch_log.csv")
        ax_step.plot(ep["step"], ep["test_bce_per_pixel"], "-o",
                      label=label, color=color, markersize=3)
        ax_time.plot(ep["wall_s"], ep["test_bce_per_pixel"], "-o",
                      label=label, color=color, markersize=3)
    for ax in (ax_step, ax_time):
        ax.set_ylabel("test BCE per pixel")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    ax_step.set_xlabel("step")
    ax_time.set_xlabel("wall time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    out = HERE / f"c1_autoencoder_{suffix}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    plot_run_set(
        [("hf", "HF (default init)", "tab:blue"),
         ("sgd_momentum", "SGD+momentum (default init)", "tab:orange")],
        "default_init",
        "C1 autoencoder, default kaiming-uniform init",
    )
    plot_run_set(
        [("hf_martens_init", "HF (Martens N(0, 0.01))", "tab:blue"),
         ("sgd_martens_init", "SGD+momentum (Martens N(0, 0.01))",
          "tab:orange")],
        "martens_init",
        "C1 autoencoder, Martens 2010 small-Gaussian init",
    )


if __name__ == "__main__":
    main()
