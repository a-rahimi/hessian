"""Run the streaming benchmark and log per-step results to CSV.

For each config in experiments.CONFIGS, launch src/train_newton.py as a
subprocess, parse its per-step stderr, and write results/<name>.csv with columns
step,train_loss,probe_loss,probe_accuracy,wall_clock_s. Because a fresh minibatch
is drawn every step, the held-out probe_loss (a fixed evaluation batch) is the
signal that matters, not the noisy per-step train_loss.

    python benchmarks/streaming/run.py            # both configs
    python benchmarks/streaming/run.py tr_gelu    # a single config by name
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

STEP_RE = re.compile(
    r"step=\s*(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+.*?"
    r"probe=(?P<probe>[-+0-9.eE]+)\s+"
    r"probe_acc=(?P<probe_acc>[-+0-9.eE]+).*?"
    r"\bt=(?P<t>[-+0-9.eE]+)s"
)


def parse_line(line: str) -> dict | None:
    m = STEP_RE.search(line)
    if m is None:
        return None
    return {
        "step": int(m["step"]),
        "train_loss": float(m["loss"]),
        "probe_loss": float(m["probe"]),
        "probe_accuracy": float(m["probe_acc"]),
        "wall_clock_s": float(m["t"]),
    }


def run_config(name: str) -> None:
    config = CONFIGS_BY_NAME[name]
    cmd = [sys.executable, str(TRAIN_SCRIPT), *config.args]
    print(f"\n=== {name} ({config.method}, {config.activation}) ===", flush=True)
    print(" ".join(cmd), flush=True)

    rows: list[dict] = []
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True,
    )
    for line in proc.stderr:
        row = parse_line(line)
        if row is not None:
            rows.append(row)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} exited with code {proc.returncode}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step", "train_loss", "probe_loss", "probe_accuracy", "wall_clock_s"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)
    summarize(name, rows)


def summarize(name: str, rows: list[dict]) -> None:
    if not rows:
        print(f"{name}: NO ROWS PARSED", flush=True)
        return
    best = min(rows, key=lambda r: r["probe_loss"])
    print(
        f"{name}: best probe_loss={best['probe_loss']:.4f} at step {best['step']} "
        f"({best['wall_clock_s']:.1f}s), best probe_acc="
        f"{max(r['probe_accuracy'] for r in rows):.3f}",
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
