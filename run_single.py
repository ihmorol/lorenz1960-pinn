"""Entry point: train one architecture, with the full per-point breakdown.

    python run_single.py          # the default 4x60
    python run_single.py 5 70     # depth 5, width 70
    python run_single.py 4 60 --sequential    # walk the collocation points in time

Writes runs/<depth>x<width>/ exactly as the sweep does, so a single run can be
added to or refreshed without retraining the other eight. Prints the run's
63-column summary. The run of record under src/pinn/ is not touched.

With ``--sequential`` the architecture tag becomes ``<depth>x<width>_seq``, so the
run lands beside the single-domain one instead of overwriting it and the two
summaries can be diffed directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402

if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    unknown = set(flags) - {"--sequential"}
    if unknown:
        raise SystemExit(f"unknown flag(s): {', '.join(sorted(unknown))}")

    base = Config(sequential="--sequential" in flags)
    depth, width = (int(a) for a in positional[:2]) if len(positional) >= 2 else (base.depth, base.width)
    summary = run_one(sweep_config(base, depth, width))
    print(summary.T.to_string(header=False))
