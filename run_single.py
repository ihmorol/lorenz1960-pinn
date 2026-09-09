"""Entry point: train one architecture, with the full per-point breakdown.

    python run_single.py          # the default 4x60
    python run_single.py 5 70     # depth 5, width 70

Writes runs/<depth>x<width>/ exactly as the sweep does, so a single run can be
added to or refreshed without retraining the other eight. Prints the run's
63-column summary. The run of record under src/fydp2/ is not touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fydp2.config import Config  # noqa: E402
from fydp2.sweep import run_one, sweep_config  # noqa: E402

if __name__ == "__main__":
    base = Config()
    depth, width = (int(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (base.depth, base.width)
    summary = run_one(sweep_config(base, depth, width))
    print(summary.T.to_string(header=False))
