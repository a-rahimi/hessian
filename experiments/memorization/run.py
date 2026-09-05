"""Run the memorization experiment and write results/<name>.csv per config.

    python experiments/memorization/run.py            # every config
    python experiments/memorization/run.py tr_gelu    # one config by name

The running and parsing live in experiments/harness.py, which every experiment
shares, so this file only says which configs to run and where to put them.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

import harness  # noqa: E402
from experiments import CONFIGS  # noqa: E402

if __name__ == "__main__":
    harness.main(CONFIGS, SCRIPT_DIR / "results")
