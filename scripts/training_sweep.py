"""PINN training sweep over the (k, l, initial state) grid of reference_sweep.

One default 4x60 run per cell, written to runs/training_sweep/<tag>/ with the
same resume semantics as the architecture sweep: finished runs are skipped,
interrupted ones are redone.

Usage:
    python scripts/training_sweep.py [--epochs N] [--limit N] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one  # noqa: E402
from reference_sweep import ICS, KS, LS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=Config().epochs)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N cells")
    ap.add_argument("--seeds", type=str, default="", help="comma-separated seeds; empty = seed 0")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else (0,)
    base = _REPO / "runs" / "training_sweep"
    base.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    cells = [(k, l, i, s) for k in KS for l in LS for i in range(len(ICS)) for s in seeds]
    for n, (k, l, ic_idx, seed) in enumerate(cells[:args.limit or None], start=1):
        ic = ICS[ic_idx]
        cfg = replace(
            Config(k=k, l=l, initial_state=ic, epochs=args.epochs, seed=seed,
                   lbfgs_iters=1000),
            results_dir=f"runs/training_sweep/k{k:g}_l{l:g}_ic{ic_idx + 1}_seed{seed}",
            ckpt_dir=f"runs/training_sweep/k{k:g}_l{l:g}_ic{ic_idx + 1}_seed{seed}/history",
        )
        print(f"\n=== [{n}/{len(cells)}] {cfg.results_dir} "
              f"({(time.perf_counter() - started) / 60:.1f} min elapsed) ===", flush=True)
        summary = run_one(cfg, resume=True).iloc[0]
        rows.append({"tag": cfg.results_dir.rsplit("/", 1)[-1], "k": k, "l": l,
                     "initial_state": ic, "seed": seed, "epochs": args.epochs,
                     "final_loss": summary["final_loss"],
                     "rmse_combined_l2": summary["rmse_combined_l2"],
                     "wall_clock_s": summary["wall_clock_s"]})
        pd.DataFrame(rows).to_csv(base / "training_sweep.csv", index=False)

    table = pd.DataFrame(rows)
    base.mkdir(parents=True, exist_ok=True)
    table.to_csv(base / "training_sweep.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
