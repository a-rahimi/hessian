"""Run the fixed-batch memorization benchmark and log per-step results to CSV.

For each config in experiments.CONFIGS, launch src/train_newton.py as a
subprocess, parse its per-step stderr lines, and write
results/<name>.csv with columns step,loss,batch_accuracy,wall_clock_s.

    python benchmarks/memorization/run.py            # all six configs
    python benchmarks/memorization/run.py tr_tanh    # a single config by name
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR = SCRIPT_DIR / "results"
TRAIN_SCRIPT = REPO_ROOT / "src" / "train_newton.py"

sys.path.insert(0, str(SCRIPT_DIR))
from experiments import CONFIGS, CONFIGS_BY_NAME  # noqa: E402

# Matches lines like:
#   [trust-region] step=  12 loss=1.2345 avg10=... probe=... probe_acc=...
#   bacc=0.812 ... t=4.5s (step 410ms)
STEP_RE = re.compile(
    r"step=\s*(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+.*?"
    r"bacc=(?P<bacc>[-+0-9.eE]+).*?"
    r"\bt=(?P<t>[-+0-9.eE]+)s"
)


def parse_line(line: str) -> dict | None:
    m = STEP_RE.search(line)
    if m is None:
        return None
    return {
        "step": int(m["step"]),
        "loss": float(m["loss"]),
        "batch_accuracy": float(m["bacc"]),
        "wall_clock_s": float(m["t"]),
    }


def run_config(name: str) -> list[dict]:
    config = CONFIGS_BY_NAME[name]
    cmd = [sys.executable, str(TRAIN_SCRIPT), *config.args]
    print(f"\n=== {name} ({config.method}, {config.activation}) ===", flush=True)
    print(" ".join(cmd), flush=True)

    rows: list[dict] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in proc.stderr:
        row = parse_line(line)
        if row is not None:
            rows.append(row)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} exited with code {proc.returncode}")

    write_csv(name, rows)
    summarize(name, rows)
    return rows


def write_csv(name: str, rows: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["step", "loss", "batch_accuracy", "wall_clock_s"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


def summarize(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{name}: NO ROWS PARSED", flush=True)
        return
    min_loss = min(r["loss"] for r in rows)
    best_bacc = max(r["batch_accuracy"] for r in rows)
    reached = next((r for r in rows if r["batch_accuracy"] >= 1.0), None)
    if reached is not None:
        hit = f"step {reached['step']} at {reached['wall_clock_s']:.2f}s"
    else:
        hit = "never"
    print(
        f"{name}: min_loss={min_loss:.4f} best_bacc={best_bacc:.3f} "
        f"reached_100%={hit}",
        flush=True,
    )


def main() -> None:
    names = sys.argv[1:] or [c.name for c in CONFIGS]
    for name in names:
        if name not in CONFIGS_BY_NAME:
            raise SystemExit(f"unknown config: {name}")
    for name in names:
        run_config(name)


if __name__ == "__main__":
    main()
