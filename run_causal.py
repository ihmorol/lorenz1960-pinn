import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one  # noqa: E402

T_LOOP = 13.26446

CFG = Config(t_span=(0.0, T_LOOP), n_collocation=27 * 1535 + 1, eval_per_unit=1000,
             dtype="float64", ic_scale="unit", collocation="uniform", n_windows=27,
             causal_eps_schedule=(1e-2, 1e-1, 1.0, 10.0), causal_delta=0.99,
             causal_max_iters=4000, lr_start=1e-3, lr_end=1e-4,
             lr_decay=0.9, lr_decay_every=1000, lbfgs_iters=1000,
             warm_start=True, snapshot_every=2000,
             results_dir="runs/active/windowed-causal-1536pt",
             ckpt_dir="runs/active/windowed-causal-1536pt/history",
             runs_dir="runs/active")

if __name__ == "__main__":
    row = run_one(CFG, resume=True)
    print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
               "wall_clock_s"]].to_string(index=False))
