"""Make C2 comparison plots from the run logs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
LOGDIR = HERE / "logs"

REFERENCE_SGD_PROBE = 1.97        # Phase 4 anchor, 1000 steps.
REFERENCE_NEWTON_PROBE_LOW = 2.22  # Plateau lower bound from summary-so-far.
REFERENCE_NEWTON_PROBE_HIGH = 2.30
REFERENCE_SGD_FIXED_BATCH = 0.19   # exp-048b.
REFERENCE_NEWTON_FIXED_BATCH = 1.16  # Newton fixed-batch floor.


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


def plot_full_anchor() -> None:
    fig, (ax_probe, ax_train) = plt.subplots(1, 2, figsize=(12, 4.5))

    for run, label, color in [
        ("hf_ggn_full", "HF-GGN", "tab:blue"),
        ("hf_hessian_full", "HF-Hessian", "tab:red"),
    ]:
        d = read_csv(LOGDIR / run / "step_log.csv")
        ax_probe.plot(d["step"], d["probe_loss"], label=label, color=color)
        ax_train.plot(d["step"], d["train_loss_avg10"], label=label, color=color)

    for ax in (ax_probe, ax_train):
        ax.axhline(REFERENCE_SGD_PROBE, color="gray", ls="--", lw=1,
                    label="SGD probe = 1.97")
        ax.axhspan(REFERENCE_NEWTON_PROBE_LOW, REFERENCE_NEWTON_PROBE_HIGH,
                    color="orange", alpha=0.18, label="Our Newton plateau")
        ax.set_xlabel("step")
        ax.set_ylim(1.5, 3.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax_probe.set_ylabel("probe loss (fixed 256-sample probe)")
    ax_train.set_ylabel("train loss (10-step trailing mean)")
    fig.suptitle("C2 Phase 5 anchor — HF in GGN vs raw-Hessian mode")
    fig.tight_layout()
    fig.savefig(HERE / "c2_full_anchor.png", dpi=110)
    plt.close(fig)


def plot_fixed_batch() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, label, color in [
        ("hf_ggn_fixedbatch", "HF-GGN", "tab:blue"),
        ("hf_hessian_fixedbatch", "HF-Hessian", "tab:red"),
    ]:
        d = read_csv(LOGDIR / run / "step_log.csv")
        ax.semilogy(d["step"], d["loss"], label=label, color=color, lw=1)

    ax.axhline(REFERENCE_SGD_FIXED_BATCH, color="green", ls="--", lw=1,
                label="SGD floor = 0.19 (exp-048b)")
    ax.axhline(REFERENCE_NEWTON_FIXED_BATCH, color="orange", ls="--", lw=1,
                label="Our Newton floor = 1.16")
    ax.set_xlabel("step")
    ax.set_ylabel("train loss on fixed 64-sample batch (log scale)")
    ax.set_title("C2 fixed-batch memorization diagnostic")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "c2_fixed_batch.png", dpi=110)
    plt.close(fig)


def main() -> None:
    plot_full_anchor()
    plot_fixed_batch()
    print(f"wrote {HERE / 'c2_full_anchor.png'} and "
          f"{HERE / 'c2_fixed_batch.png'}")


if __name__ == "__main__":
    main()
