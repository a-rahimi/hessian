"""Parse a stdout.log from `train_sfn.py` into a per-step CSV.

We log a full line per step in `train_sfn.py`, of the form

    [sfn] step=  17 loss=2.2836 avg10=2.3845 probe=2.2934 probe_acc=0.086 bacc=0.125 lr=5.000e-01 ...

Pulling structured trajectories out of the rendered tensorboard event files
would also work, but the stdout log is the canonical artifact we keep under
each run directory, so parsing it directly keeps the analysis path readable
and free of TensorBoard-version assumptions.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


LINE_RE = re.compile(
    r"\[(?P<mode>\w+)\] step= *(?P<step>\d+) "
    r"loss=(?P<loss>[-\d.eE+]+) "
    r"avg10=(?P<avg10>[-\d.eE+]+) "
    r"probe=(?P<probe>[-\d.eE+]+) "
    r"probe_acc=(?P<probe_acc>[-\d.eE+]+) "
    r"bacc=(?P<bacc>[-\d.eE+]+) "
    r"lr=(?P<lr>[-\d.eE+]+) "
    r"\|g\|=(?P<gnorm>[-\d.eE+]+) "
    r"\|Δ\|=(?P<stepnorm>[-\d.eE+]+)"
    r"(?: ε=(?P<eps>[-\d.eE+]+) (?P<accept>ok|REJ)"
    r"(?: λmax=(?P<lam_max>[-\d.eE+]+) λmin=(?P<lam_min>[-\d.eE+]+) "
    r"neg=(?P<frac_neg>[-\d.eE+]+) mv=(?P<mv>\d+))?"
    r")? "
    r"t=(?P<wall>[-\d.eE+]+)s"
)


def parse(log_path: Path):
    rows = []
    for line in log_path.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        rows.append(m.groupdict())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        raise SystemExit(f"No rows parsed from {args.log}")
    fields = list(rows[0].keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
