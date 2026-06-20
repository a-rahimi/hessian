"""Plot loss curves for the two trust-region subproblem solvers.

Overlays the dense (brute-force eigendecomposition) and efficient (linear-time
oracle) solvers on the batch-memorization problem, both training loss vs step
and vs wall-clock time.

Usage:
    python scripts/plot_tr_solvers.py \
        --efficient-run runs/memo-efficient \
        --dense-run runs/memo-dense \
        --outdir runs/memo-compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(logdir: str, tags: list[str]) -> dict[str, dict]:
    ea = EventAccumulator(logdir)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag not in available:
            print(f"  warning: tag '{tag}' not in {logdir}")
            result[tag] = {"steps": [], "times": [], "values": []}
            continue
        events = ea.Scalars(tag)
        result[tag] = {
            "steps": [e.step for e in events],
            "times": [e.wall_time for e in events],
            "values": [e.value for e in events],
        }
    for data in result.values():
        if data["times"]:
            t0 = data["times"][0]
            data["times"] = [t - t0 for t in data["times"]]
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--efficient-run", required=True)
    p.add_argument("--dense-run", required=True)
    p.add_argument("--outdir", default="runs/memo-compare")
    args = p.parse_args()

    tags = ["loss", "probe_loss"]
    eff = load_scalars(args.efficient_run, tags)
    dense = load_scalars(args.dense_run, tags)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for x_key, x_label, fname in [
        ("steps", "step", "loss_vs_step.png"),
        ("times", "wall-clock time (s)", "loss_vs_time.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            dense["loss"][x_key],
            dense["loss"]["values"],
            "o-",
            color="tab:red",
            label="dense (eigendecomposition)",
            markersize=3,
        )
        ax.plot(
            eff["loss"][x_key],
            eff["loss"]["values"],
            "s--",
            color="tab:blue",
            label="efficient (linear-time oracle)",
            markersize=3,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel("training loss (fixed batch)")
        ax.set_title("Trust-region batch memorization: dense vs efficient solver")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = outdir / fname
        fig.savefig(path, dpi=130)
        print(f"wrote {path}")

    # Quick numeric agreement summary on the overlapping steps.
    de, ef = dense["loss"]["values"], eff["loss"]["values"]
    n = min(len(de), len(ef))
    if n:
        max_abs = max(abs(de[i] - ef[i]) for i in range(n))
        print(f"max |loss_dense - loss_efficient| over {n} steps: {max_abs:.3e}")
        print(f"final loss  dense={de[n-1]:.4f}  efficient={ef[n-1]:.4f}")


if __name__ == "__main__":
    main()
