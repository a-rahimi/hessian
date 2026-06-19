"""Print summary statistics for a run's metrics.csv.

Used by the README and the A1 / A2 reports so the "headline numbers" are
not hand-rolled. Reads the same per-step CSV that
[subsampled_newton.py](./subsampled_newton.py) emits.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    args = p.parse_args()

    csv_path = Path(args.csv_path)
    rows: list[dict[str, float]] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})

    n = len(rows)
    losses = [r["loss"] for r in rows]
    avg10 = [r["train_loss_avg10"] for r in rows]
    probe = [r["probe_loss"] for r in rows]
    probe_acc = [r["probe_accuracy"] for r in rows]
    bacc = [r["batch_accuracy"] for r in rows]
    step_norms = [r["step_norm"] for r in rows]
    step_secs = [r["step_seconds"] for r in rows]

    print(f"=== {csv_path.parent.name} ===")
    print(f"steps:                 {n}")
    print(f"loss min:              {min(losses):.4f}  (step {losses.index(min(losses))})")
    print(f"loss max:              {max(losses):.4f}  (step {losses.index(max(losses))})")
    print(f"loss final:            {losses[-1]:.4f}")
    print(f"avg10 min:             {min(avg10):.4f}")
    print(f"avg10 final:           {avg10[-1]:.4f}")
    print(f"probe_loss min:        {min(probe):.4f}")
    print(f"probe_loss final:      {probe[-1]:.4f}")
    print(f"probe_acc max:         {max(probe_acc):.4f}")
    print(f"batch_acc max:         {max(bacc):.4f}")
    print(f"step_norm mean:        {statistics.mean(step_norms):.4f}")
    print(f"step_norm max:         {max(step_norms):.4f}")
    print(f"step_seconds mean:     {statistics.mean(step_secs):.2f}")
    print(f"total wall_clock_s:    {rows[-1]['wall_clock_s']:.1f}")


if __name__ == "__main__":
    main()
