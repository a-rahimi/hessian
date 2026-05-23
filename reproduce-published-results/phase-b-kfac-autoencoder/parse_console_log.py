"""Recover a JSONL log file from a console-format log.

The asdl-driven K-FAC run lost most of its JSONL output due to a buffering
quirk on macOS (the file descriptor's writes never made it to disk despite
explicit ``flush()`` calls; the file landed at 1029 bytes while ``lsof``
reported 5801 bytes allocated). The console log written via ``tee`` is
intact, so we rebuild the JSONL by regexing it.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LINE_RE = re.compile(
    r"step=\s*(\d+)\s+train_\w+=\s*([-\d.]+)\s+test_bce=\s*([-\d.]+)"
    r"\s+test_mse=\s*([-\d.]+)\s+t=\s*([-\d.]+)s",
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    for line in Path(args.input).read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        step, train_loss, test_loss, test_mse, wall = m.groups()
        rows.append(
            {
                "step": int(step),
                "train_loss": float(train_loss),
                "test_loss": float(test_loss),
                "test_mse": float(test_mse),
                "wall_seconds": float(wall),
            }
        )

    out = Path(args.out)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
