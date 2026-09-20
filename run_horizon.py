"""Train the same 4x60 network on a long time window under four schemes.

    python run_horizon.py            # t in [0, 10]
    python run_horizon.py 20         # t in [0, 20]

Schemes, each into runs/t<T>/<tag>/ so run_viz3d.py and run_compare.py read them
unchanged (the [0, 1] runs under runs/ are not touched):

    4x60          one network, all points at once            (baseline PINN)
    4x60_seq      this repo's trapezoid walk                 (not from the literature)
    4x60_causal   causal weighting, Wang, Sankaran & Perdikaris, CMAME 2024
    4x60_march    time-marching, Krishnapriyan et al., NeurIPS 2021 (sec. 5):
                  split [0, T] into windows of length 1, train a fresh network per
                  window with its initial state = previous window's end state.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config, reference_at  # noqa: E402
from pinn.history import SnapshotWriter  # noqa: E402
from pinn.pinn import ResidualParts, residual_parts  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402
from pinn.train import make_grid, train  # noqa: E402

WINDOW = 1.0            # marching window length; [0, 1] is where the plain PINN is exact
EPOCHS_PER_WINDOW = 5000


def march(base: Config, out: Path) -> None:
    """Krishnapriyan-style time marching. Writes breakdown/ + history/ like a normal run."""
    t0, tf = base.t_span
    edges = np.arange(t0, tf + 1e-9, WINDOW)
    state, losses, grids, parts = tuple(base.initial_state), [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        cfg = replace(base, t_span=(float(a), float(b)), initial_state=state,
                      epochs=EPOCHS_PER_WINDOW, snapshot_every=0, print_every=0)
        model, hist = train(cfg)
        losses += hist.loss
        grid = make_grid(cfg, next(model.parameters()).device)
        p = residual_parts(model, grid.clone().requires_grad_(True))
        grids.append(grid.detach()); parts.append(p)
        with torch.no_grad():
            state = tuple(float(v) for v in model(torch.tensor([[b]], dtype=torch.float32))[0])
        print(f"[march] window [{a:.0f}, {b:.0f}] loss {hist.loss[-1]:.2e} end state {np.round(state, 4)}",
              flush=True)

    grid_t = torch.cat(grids).cpu().numpy().reshape(-1)
    joined = ResidualParts(*[torch.cat([getattr(p, f) for p in parts]).detach()
                              for f in ResidualParts._fields])
    SnapshotWriter(out / "breakdown", grid_t, reference_at(base, grid_t)).write(len(losses) - 1, joined)
    (out / "history").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"iteration": range(len(losses)), "loss": losses}).to_csv(
        out / "history" / "loss_history.csv", index=False)
    err = np.linalg.norm(joined.u.numpy() - reference_at(base, grid_t), axis=1)
    print(f"[march] rmse {np.sqrt((err**2).mean()):.2e}  max err {err.max():.2e}", flush=True)


if __name__ == "__main__":
    t_end = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    base = Config(t_span=(0.0, t_end), runs_dir=f"runs/t{t_end:g}")
    for kw in ({}, {"sequential": True}, {"causal_eps": 1e-2}):
        cfg = sweep_config(replace(base, **kw), base.depth, base.width)
        print(f"=== {cfg.label} -> {cfg.results_path} ===", flush=True)
        row = run_one(cfg, resume=True)
        print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
                   "wall_clock_s"]].to_string(index=False), flush=True)
    print(f"=== time marching, {WINDOW}-unit windows -> {base.results_path.parent / '4x60_march'} ===",
          flush=True)
    march(base, base.runs_path / "4x60_march")
