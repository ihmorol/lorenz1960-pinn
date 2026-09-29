"""Run the 27-window causal PINN at depths 1-3 and widths 10, 20, 30.

    python run_small_grid.py --plan       # show parameter counts without training
    python run_small_grid.py --workers 2  # train two independent models at once

Each architecture has its own checkpoint and can be resumed after interruption.
The 3x30 result is retained for the requested full grid but excluded from the
under-50,000-parameter winner.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.history import FLOAT_FORMAT  # noqa: E402
from pinn.sweep import run_one  # noqa: E402
from run_causal import CFG  # noqa: E402

DEPTHS = (1, 2, 3)
WIDTHS = (10, 20, 30)
PARAMETER_LIMIT = 50_000
RUNS_DIR = "runs/causal-window/small-grid"


def parameter_count(depth: int, width: int, windows: int) -> int:
    """Linear(1, width), hidden layers, Linear(width, 3), per window."""
    return windows * ((depth - 1) * width**2 + (depth + 4) * width + 3)


def configurations():
    base = replace(CFG, runs_dir=RUNS_DIR, snapshot_every=0)
    for depth in DEPTHS:
        for width in WIDTHS:
            arch = f"{depth}x{width}"
            root = f"{RUNS_DIR}/{arch}"
            yield replace(base, depth=depth, width=width,
                          results_dir=root, ckpt_dir=f"{root}/history")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2,
                        help="parallel training processes (default: 2 for one Colab GPU)")
    parser.add_argument("--plan", action="store_true", help="show the nine runs without training")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    configs = list(configurations())
    for cfg in configs:
        count = parameter_count(cfg.depth, cfg.width, cfg.n_windows)
        status = "within budget" if count < PARAMETER_LIMIT else "over budget"
        print(f"{cfg.depth}x{cfg.width}: {count:,} parameters ({status}) -> {cfg.results_path}",
              flush=True)
    if args.plan:
        return

    rows = []
    output = configs[0].runs_path
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        jobs = {pool.submit(run_one, cfg, True): cfg for cfg in configs}
        for job in as_completed(jobs):
            cfg = jobs[job]
            row = job.result()
            actual = int(row["n_params"].iloc[0])
            expected = parameter_count(cfg.depth, cfg.width, cfg.n_windows)
            if actual != expected:
                raise ValueError(f"{cfg.arch}: expected {expected} parameters, got {actual}")
            rows.append(row)
            comparison = pd.concat(rows, ignore_index=True)
            comparison["within_budget"] = comparison["n_params"] < PARAMETER_LIMIT
            comparison.sort_values(["depth", "width"]).to_csv(
                output / "comparison.csv", index=False, float_format=FLOAT_FORMAT)
            print(f"[done] {cfg.arch}: RMSE={row['rmse_combined_l2'].iloc[0]:.6g} "
                  f"({len(rows)}/{len(configs)})", flush=True)

    eligible = comparison[comparison["within_budget"]]
    winner = eligible.sort_values(["rmse_combined_l2", "n_params"]).iloc[0]
    print(f"[best under {PARAMETER_LIMIT:,}] {winner['arch']}: "
          f"{int(winner['n_params']):,} parameters, "
          f"RMSE={winner['rmse_combined_l2']:.6g}", flush=True)
    print(f"[results] {output / 'comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
