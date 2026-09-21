"""Run B: causal windowed training (Wang, Sankaran & Perdikaris 2024, Alg. 1 + App. E).

    python run_causal.py           # writes runs/4x60_f64_unit_win27_causal/
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402

T_LOOP = 13.26446

CFG = Config(t_span=(0.0, T_LOOP), points_per_unit=3000, eval_per_unit=1000,
             dtype="float64", ic_scale="unit", collocation="uniform", n_windows=27,
             causal_eps_schedule=(1e-2, 1e-1, 1.0, 10.0), causal_delta=0.99,
             causal_max_iters=4000, lr_decay=0.9, lr_decay_every=5000, lbfgs_iters=1500,
             warm_start=True)
SNAPSHOT_EVERY = 2000

if __name__ == "__main__":
    cfg = replace(sweep_config(CFG, CFG.depth, CFG.width), snapshot_every=SNAPSHOT_EVERY)
    row = run_one(cfg, resume=True)
    print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
               "wall_clock_s"]].to_string(index=False))
