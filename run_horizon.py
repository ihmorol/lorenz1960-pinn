"""Train the same 4x60 network on a longer time window, parallel and sequential.

    python run_horizon.py            # t in [0, 10], both schemes
    python run_horizon.py 20         # t in [0, 20]

Writes runs/t<T>/4x60/ and runs/t<T>/4x60_seq/ with the full per-point breakdown,
so run_viz3d.py and the standard figure suite work on them unchanged. The [0, 1]
runs under runs/ are not touched.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402

if __name__ == "__main__":
    t_end = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    base = Config(t_span=(0.0, t_end), runs_dir=f"runs/t{t_end:g}")
    for sequential in (False, True):
        cfg = sweep_config(replace(base, sequential=sequential), base.depth, base.width)
        print(f"=== {cfg.label} -> {cfg.results_path} ===", flush=True)
        row = run_one(cfg, resume=True)
        print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
                   "wall_clock_s"]].to_string(index=False), flush=True)
