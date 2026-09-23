"""Post-training reporting: prediction, error metrics, and the wide run summary.

Nothing here feeds back into the loss. These functions read a *finished* run —
its checkpoint, grid, and telemetry — and produce the metric table, the
63-column :func:`run_summary` row the sweep concatenates, and the
:class:`~pinn.viz.RunArtifacts` bundle the figure suite consumes.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import viz as figures
from ..config import Config, compute_error_metrics, reference_trajectory
from ..pinn import residual_of
from .collocation import make_grid

LOSS_THRESHOLDS = (1e-4, 1e-6, 1e-8)


def predict(model, t: np.ndarray) -> np.ndarray:
    p = next(model.parameters())
    tt = torch.as_tensor(t, dtype=p.dtype, device=p.device).reshape(-1, 1)
    with torch.no_grad():
        return model(tt).cpu().numpy().astype(np.float64)


def residual_at(model, t: np.ndarray) -> np.ndarray:
    """ODE residual r(t) = du/dt - f(u) evaluated at arbitrary times."""
    p = next(model.parameters())
    tt = torch.as_tensor(t, dtype=p.dtype, device=p.device).reshape(-1, 1).requires_grad_(True)
    return residual_of(model, tt).detach().cpu().numpy().astype(np.float64)


def evaluate(model, cfg: Config) -> pd.DataFrame:
    t, ys = reference_trajectory(cfg, n=cfg.n_eval)
    return compute_error_metrics(ys, predict(model, t))


def collect_artifacts(model, history, cfg: Config) -> figures.RunArtifacts:
    """Bundle a finished run into the plain-array record the figure suite reads."""
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
    grid = make_grid(cfg, next(model.parameters()).device).cpu().numpy().reshape(-1)
    return figures.RunArtifacts(
        t=t,
        pred=predict(model, t),
        ref=ref,
        history=history,
        coefficients=cfg.coefficients,
        collocation_t=grid,
        collocation_residual=residual_at(model, grid),
        dense_residual=residual_at(model, t),
        t_span=cfg.t_span,
        label=cfg.label,
    )


def epochs_to(loss: list[float], threshold: float) -> float:
    """First iteration index at which the loss dropped below ``threshold``."""
    for i, value in enumerate(loss):
        if value < threshold:
            return float(i)
    return float("nan")


def run_summary(model, history, cfg: Config,
                run: figures.RunArtifacts | None = None) -> pd.DataFrame:
    """One wide row describing a finished run: config, accuracy, physics, cost.

    This is the row the architecture sweep concatenates into ``comparison.csv``.
    Every column is computed after training; nothing here feeds back into the loss.
    """
    run = run if run is not None else collect_artifacts(model, history, cfg)
    pred, ref, t = run.pred, run.ref, run.t
    err = pred - ref
    err_norm = np.linalg.norm(err, axis=1)
    metrics = run.metrics.set_index("state")

    row: dict[str, object] = {
        # --- configuration
        "arch": cfg.arch, "problem": cfg.problem, "depth": cfg.depth, "width": cfg.width,
        "n_windows": cfg.n_windows, "share_network": cfg.share_network,
        "causal_eps_schedule": " ".join(f"{e:g}" for e in cfg.causal_eps_schedule),
        "causal_delta": cfg.causal_delta, "causal_max_iters": cfg.causal_max_iters, "warm_start": cfg.warm_start,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "activation": cfg.activation, "ic": cfg.ic, "gamma": cfg.gamma,
        "epochs": cfg.epochs, "lbfgs_iters": cfg.lbfgs_iters, "seed": cfg.seed,
        "n_collocation": cfg.n_collocation,
        "lr_start": cfg.lr_start, "lr_end": cfg.lr_end,
        "t_start": cfg.t_span[0], "t_end": cfg.t_span[1],
        "dtype": cfg.dtype, "ic_scale": cfg.ic_scale, "collocation": cfg.collocation,
        "n_eval": cfg.n_eval, "lr_decay": cfg.lr_decay if cfg.lr_decay is not None else float("nan"),
        "lr_decay_every": cfg.lr_decay_every,
    }

    # --- accuracy, per state and combined
    for state in ("x", "y", "z", "combined_l2"):
        for metric in ("mae", "rmse", "max_abs_error"):
            row[f"{metric}_{state}"] = float(metrics.loc[state, metric])

    for j, name in enumerate("xyz"):
        span = float(ref[:, j].max() - ref[:, j].min())
        ss_tot = float(((ref[:, j] - ref[:, j].mean()) ** 2).sum())
        row[f"rel_mae_{name}"] = float(np.abs(err[:, j]).mean() / span) if span else float("nan")
        row[f"r2_{name}"] = 1.0 - float((err[:, j] ** 2).sum()) / ss_tot if ss_tot else float("nan")
        # Endpoint state: where the trajectory actually landed versus the truth.
        row[f"final_{name}"] = float(pred[-1, j])
        row[f"final_ref_{name}"] = float(ref[-1, j])
        row[f"final_err_{name}"] = float(err[-1, j])

    for q in (50, 90, 99):
        row[f"err_norm_p{q}"] = float(np.percentile(err_norm, q))
    row["err_norm_max"] = float(err_norm.max())

    # --- physics: does the ODE hold away from the points we trained on?
    row["final_loss"] = float(history.loss[-1]) if history.loss else float("nan")
    row["best_loss"] = float(min(history.loss)) if history.loss else float("nan")
    coll = float(np.mean(np.asarray(run.collocation_residual) ** 2)) \
        if run.collocation_residual is not None else float("nan")
    dense = float(np.mean(np.asarray(run.dense_residual) ** 2)) \
        if run.dense_residual is not None else float("nan")
    row["residual_mse_collocation"] = coll
    row["residual_mse_dense"] = dense
    row["residual_generalisation_gap"] = dense / coll if coll else float("nan")

    # --- conserved quantities
    _, drift = figures.invariant_series(pred, run.coefficients)
    _, ref_drift = figures.invariant_series(ref, run.coefficients)
    for i in range(drift.shape[1]):
        row[f"invariant_{i + 1}_max_drift"] = float(drift[:, i].max())
        row[f"invariant_{i + 1}_max_drift_reference"] = float(ref_drift[:, i].max())

    # --- cost
    wall = float(history.wall_clock_s)
    row["wall_clock_s"] = wall
    row["ms_per_epoch"] = wall / cfg.epochs * 1e3 if cfg.epochs else float("nan")
    for threshold in LOSS_THRESHOLDS:
        row[f"epochs_to_{threshold:.0e}"] = epochs_to(history.loss, threshold)
    row["device"] = next(model.parameters()).device.type
    # CUDA-only; NaN on CPU, where torch exposes no equivalent counter.
    row["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
    )
    row["n_snapshots"] = len(history.snapshots.epochs) if history.snapshots else 0
    return pd.DataFrame([row])


def config_for(run_dir: str | Path) -> Config:
    """Rebuild the Config a finished run was trained with, from its own summary.

    Needed to reload a checkpoint: the weights only fit a network of the same
    shape. Falls back to the defaults for runs predating ``run_summary.csv``,
    which is correct for the 4x60 run of record.
    """
    run_dir = Path(run_dir)
    paths = {"results_dir": str(run_dir), "ckpt_dir": str(run_dir / "history")}
    summary = run_dir / "run_summary.csv"
    if not summary.exists():
        return replace(Config(), **paths) if run_dir != Config().results_path else Config()

    row = pd.read_csv(summary).iloc[0]
    return replace(
        Config(), **paths,
        depth=int(row["depth"]), width=int(row["width"]), activation=str(row["activation"]),
        ic=str(row["ic"]), seed=int(row["seed"]), epochs=int(row["epochs"]),
        n_collocation=int(row["n_collocation"]),
        t_span=(float(row["t_start"]), float(row["t_end"])),
        lbfgs_iters=int(row["lbfgs_iters"]), gamma=float(row["gamma"]),
        lr_start=float(row["lr_start"]), lr_end=float(row["lr_end"]),
        dtype=str(row.get("dtype", "float32")), ic_scale=str(row.get("ic_scale", "span")),
        collocation=str(row.get("collocation", "lhs")), n_eval=int(row.get("n_eval", 1001)),
        lr_decay=None if pd.isna(row.get("lr_decay", float("nan"))) else float(row["lr_decay"]),
        lr_decay_every=int(row.get("lr_decay_every", 5000)),
        problem=str(row.get("problem", "lorenz1960")), n_windows=int(row.get("n_windows", 1)),
        causal_eps_schedule=tuple(float(e) for e in str(row.get("causal_eps_schedule", "")).split()
                                  if e not in ("", "nan")),
        causal_delta=float(row.get("causal_delta", 0.99)), causal_max_iters=int(row.get("causal_max_iters", 0)),
        warm_start=bool(row.get("warm_start", False)),
        share_network=bool(row.get("share_network", False)),
    )
