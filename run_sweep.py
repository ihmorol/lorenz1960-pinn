"""Entry point: run the depth x width architecture sweep.

Run from the repository root:  python run_sweep.py

Trains nine networks (depth 3/4/5 x width 50/60/70) at one fixed seed and writes
each to runs/<depth>x<width>/, plus runs/comparison.csv and runs/figures/.
The run of record under src/fydp2/ is not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fydp2.sweep import sweep  # noqa: E402

if __name__ == "__main__":
    sweep()
