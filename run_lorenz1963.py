"""Lorenz-1963 through the unchanged pipeline.

    python run_lorenz1963.py           # RK4 baseline + run-of-record PINN -> runs/lorenz1963/
    python run_lorenz1963.py causal    # the run_causal.py configuration instead
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config, compute_error_metrics  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402
from lorenz1960_baseline import plot_error_curves, plot_state_time_series  # noqa: E402
from systems import rk4_vs_scipy  # noqa: E402
from run_causal import CFG as CAUSAL  # noqa: E402

# sigma, rho, beta from Lorenz (1963); start state and window from the DeepXDE Lorenz benchmark.
PROBLEM = dict(problem="lorenz1963", params=(10.0, 28.0, 8 / 3), initial_state=(-8.0, 7.0, 27.0),
               t_span=(0.0, 3.0))


def baseline(cfg: Config) -> None:
    out = cfg.runs_path / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    t, rk4, ref = rk4_vs_scipy(cfg.problem, cfg.params, cfg.initial_state, cfg.t_span)
    metrics = compute_error_metrics(ref, rk4)
    metrics.to_csv(out / "metrics.csv", index=False)
    plot_state_time_series(t, rk4, title="RK4 vs DOP853", overlay=ref)[0].savefig(out / "solution.png", dpi=200)
    plot_error_curves(t, ref, rk4, title="RK4 error vs DOP853")[0].savefig(out / "error.png", dpi=200)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    causal = sys.argv[1:] == ["causal"]
    cfg = replace(CAUSAL if causal else Config(), **PROBLEM,
                  runs_dir="runs/lorenz1963" + ("/causal-window" if causal else ""))
    baseline(cfg)
    row = run_one(sweep_config(cfg, cfg.depth, cfg.width), resume=True)
    print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
               "wall_clock_s"]].to_string(index=False))
