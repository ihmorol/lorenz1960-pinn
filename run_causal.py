"""Run B: causal windowed training (Wang, Sankaran & Perdikaris 2024, Alg. 1 + App. E).

    python run_causal.py                          # 27 windows, one network per window
    python run_causal.py --shared                 # 27 windows through ONE ~11k-param network
    python run_causal.py --out <dirname>          # write into runs/<dirname> (never clobber an archive)

Config changes under test on this pass (vs. the archived 4x60_f64_unit_win27_causal_warm,
which used 39793 points and 1500 L-BFGS iters/window):
  * collocation points  -> 1500 total
  * L-BFGS              -> 500 iterations per window
  * epsilon schedule    -> 5 stages (1e-2, 1e-1, 1.0, 10.0, 100.0)

``--shared`` keeps every other knob identical and swaps the architecture only:
one shared network is reused for all 27 windows, each window starting from the
previous window's end state (see :class:`pinn.pinn.SharedWindowPINN`). The run
then has ``depth x width`` parameters in total, not 27 copies of them.

``--out`` is required when running the per-window variant at this cheaper config:
its natural tag collides with the archived run of the same architecture, so an
explicit directory keeps both on disk. Without it, the config-derived tag is used.
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402

T_LOOP = 13.26446

CFG = Config(t_span=(0.0, T_LOOP), n_collocation=1500, eval_per_unit=1000,
             dtype="float64", ic_scale="unit", collocation="uniform", n_windows=27,
             causal_eps_schedule=(1e-2, 1e-1, 1.0, 10.0, 100.0), causal_delta=0.99,
             causal_max_iters=4000, lr_decay=0.9, lr_decay_every=5000, lbfgs_iters=500,
             warm_start=True)
SNAPSHOT_EVERY = 2000


def _arg(name: str) -> str | None:
    """Value of ``--name value``, or None when the flag is absent."""
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else None


if __name__ == "__main__":
    base = replace(CFG, share_network="--shared" in sys.argv)
    cfg = replace(sweep_config(base, base.depth, base.width), snapshot_every=SNAPSHOT_EVERY)
    out = _arg("--out")
    if out:
        runs_dir = Path(cfg.results_dir).parent
        cfg = replace(cfg, results_dir=str(runs_dir / out),
                      ckpt_dir=str(runs_dir / out / "history"))
    row = run_one(cfg, resume=True)
    print(row[["arch", "n_params", "rmse_combined_l2", "max_abs_error_combined_l2",
               "final_loss", "wall_clock_s"]].to_string(index=False))
