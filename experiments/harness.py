"""The one way to run an experiment, and the one format its results take.

Every experiment is a directory under `experiments/` holding an `experiments.py`
that defines `CONFIGS`, a `plot.py` that reads the CSVs, and a `results/`
directory of one CSV per config. This module is the only thing in the repository
that launches a training run or parses one, so no experiment carries a runner or
a regex of its own and none of them can drift apart from the others.

To run an experiment:

    python experiments/<name>/run.py            # every config
    python experiments/<name>/run.py tr_gelu    # one config by name

The parser is deliberately generic. `train_newton.py` logs each step as a line of
`key=value` pairs, so rather than each experiment naming the handful of fields it
cares about, `parse_step_line` takes every pair it can read and `run_config`
writes all of them. An experiment's plot.py then asks for the columns it wants.
That way adding a diagnostic to the training loop makes it available to every
experiment at once, and an old CSV that predates a column simply lacks it.
"""

from __future__ import annotations

import csv
import dataclasses
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = REPO_ROOT / "src" / "train_newton.py"


@dataclasses.dataclass(frozen=True)
class Config:
    """One training run: a stable name, what it is, and the exact CLI args."""

    name: str
    method: str
    activation: str
    args: list[str]


# The log line reads, for example:
#   [trust-region] step=  12 loss=2.3026 probe=2.3026 probe_acc=0.094 bacc=0.094
#   |g|=2.8e-03 Δ=1.0e+00 ρ=+0.282 λ*=3.5e-02 ok boundary eig=[-3.3e-02,+1.4e-02]
#   t=8.10s (step 285ms)
# so every datum is a `key=value` pair and bare words like `ok` are flags. The
# whitespace after `=` is allowed because the step counter is right-aligned into
# a fixed width, which puts spaces between its `=` and its digits.
_PAIR_RE = re.compile(r"(?P<key>[^\s=]+)=\s*(?P<value>[^\s]+)")
_NUMBER = r"[-+]?[0-9.]+(?:[eE][-+]?[0-9]+)?"
_BRACKETED_PAIR_RE = re.compile(rf"^\[({_NUMBER}),({_NUMBER})\]$")

# Log keys are terse and some are non-ASCII, so map them to column names that a
# plot.py can refer to without the reader having to decode them.
COLUMN_NAMES = {
    "t": "wall_clock_s",
    "probe": "probe_loss",
    "probe_acc": "probe_accuracy",
    "bacc": "batch_accuracy",
    "avg10": "loss_avg10",
    "|g|": "grad_norm",
    "|Δ|": "step_norm",
    "Δ": "trust_radius",
    "ρ": "rho",
    "λ*": "lambda_star",
    "ε": "epsilon",
    "secular": "tr_secular_evals",
    "solves": "tr_solves",
}

# A value carrying a unit; strip it rather than dropping the column.
_UNIT_SUFFIXES = ("s", "ms")


def _to_float(value: str) -> float | None:
    for suffix in _UNIT_SUFFIXES:
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    try:
        return float(value)
    except ValueError:
        return None


def parse_step_line(line: str) -> dict[str, float] | None:
    """Every numeric field on one per-step log line, or None if it is not one."""
    if "step=" not in line:
        return None

    row: dict[str, float] = {}
    for match in _PAIR_RE.finditer(line):
        key = COLUMN_NAMES.get(match["key"], match["key"])
        value = match["value"]

        # An eigenvalue range logs as a bracketed pair; split it into two columns
        # so that the CSV stays one number per cell.
        bracketed = _BRACKETED_PAIR_RE.match(value)
        if bracketed:
            low, high = (_to_float(v) for v in bracketed.groups())
            if low is not None and high is not None:
                row[f"{key}_min"], row[f"{key}_max"] = low, high
            continue

        number = _to_float(value)
        if number is not None:
            row[key] = number

    return row if "step" in row else None


def parse_log(text: str) -> list[dict[str, float]]:
    """Every per-step row in a captured stderr log."""
    return [row for line in text.splitlines() if (row := parse_step_line(line))]


def write_csv(rows: list[dict[str, float]], path: Path) -> None:
    """Write rows to `path`, with `step` first and the rest in first-seen order."""
    columns: list[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    columns.sort(key=lambda name: (name != "step",))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(results_dir: Path, name: str) -> dict[str, list[float]]:
    """A results CSV as column name -> values.

    Columns a run did not log are simply absent, so a plot.py that wants one
    should check for it rather than assume every CSV has the same width.
    """
    with (results_dir / f"{name}.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        columns: dict[str, list[float]] = {name: [] for name in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                if value != "":
                    columns[key].append(float(value))
    return columns


def run_config(config: Config, results_dir: Path) -> list[dict[str, float]]:
    """Launch one config, stream-parse its stderr, and write results/<name>.csv."""
    cmd = [sys.executable, str(TRAIN_SCRIPT), *config.args]
    print(f"\n=== {config.name} ({config.method}, {config.activation}) ===", flush=True)
    print(" ".join(cmd), flush=True)

    rows: list[dict[str, float]] = []
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    for line in proc.stderr:
        row = parse_step_line(line)
        if row is not None:
            rows.append(row)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"{config.name} exited with code {proc.returncode}")

    path = results_dir / f"{config.name}.csv"
    write_csv(rows, path)
    print(f"wrote {path} ({len(rows)} rows)", flush=True)
    summarize(config.name, rows)
    return rows


def summarize(name: str, rows: list[dict[str, float]]) -> None:
    """One line per run: where it got to, on whichever loss it recorded."""
    if not rows:
        print(f"{name}: NO ROWS PARSED", flush=True)
        return

    # Prefer the held-out loss when the run tracked one, since on a streaming
    # objective the per-step training loss is measured on a different batch every
    # step and says little.
    metric = "probe_loss" if "probe_loss" in rows[0] else "loss"
    best = min(rows, key=lambda row: row[metric])
    parts = [f"best {metric}={best[metric]:.4f} at step {int(best['step'])}"]
    if "wall_clock_s" in best:
        parts.append(f"({best['wall_clock_s']:.1f}s)")
    for accuracy in ("probe_accuracy", "batch_accuracy"):
        if accuracy in rows[0]:
            parts.append(f"best {accuracy}={max(r[accuracy] for r in rows):.3f}")
            break
    print(f"{name}: " + " ".join(parts), flush=True)


def main(configs: list[Config], results_dir: Path) -> None:
    """The body of every experiment's run.py."""
    by_name = {config.name: config for config in configs}
    names = sys.argv[1:] or [config.name for config in configs]
    for name in names:
        if name not in by_name:
            raise SystemExit(f"unknown config: {name}")
    for name in names:
        run_config(by_name[name], results_dir)
