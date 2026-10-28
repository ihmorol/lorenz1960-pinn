"""Run A: one 4x60 network over one closed orbit, float64, Adam then L-BFGS.

    python run_batch.py            # writes runs/batch-precision/4x60_f64_unit/
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402

T_LOOP = 13.26446   # rounded approximate return time; see saved reference endpoint

CFG = Config(t_span=(0.0, T_LOOP), points_per_unit=3000, eval_per_unit=1000,
             dtype="float64", ic_scale="unit", epochs=40000, lr_decay=0.9, lr_decay_every=5000,
             lbfgs_iters=5000, checkpoint_every=5000, runs_dir="runs/batch-precision")
SNAPSHOT_EVERY = 500   # 50 (the sweep default) would write ~10 GB of breakdown CSVs at 40k points

if __name__ == "__main__":
    cfg = replace(sweep_config(CFG, CFG.depth, CFG.width), snapshot_every=SNAPSHOT_EVERY)
    row = run_one(cfg, resume=True)
    print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
               "wall_clock_s"]].to_string(index=False))
