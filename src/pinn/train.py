"""Train the Lorenz-1960 PINN, evaluate against the locked baseline, save results.

This module is orchestration only: build a :class:`~pinn.functions.trainer.Trainer`,
run it, then persist the checkpoint, telemetry and figures. The loops live in
:mod:`pinn.functions.trainer`; the reporting metrics live in
:mod:`pinn.functions.reporting`. Those names are re-exported below so existing
``from pinn.train import ...`` call sites keep working unchanged.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from . import viz as figures
from .config import Config, reference_at, reference_trajectory  # noqa: F401  (re-exported)
from .history import FLOAT_FORMAT, TrainHistory, residual_grid
from .pinn import PINN
from .functions.collocation import make_grid
from .functions.reporting import (  # noqa: F401  (re-exported public surface)
    LOSS_THRESHOLDS, collect_artifacts, config_for, evaluate, predict, residual_at, run_summary,
)
from .functions.trainer import Trainer, get_device, set_seed  # noqa: F401

__all__ = [
    "train", "save_results", "write_breakdown_figures", "load_run", "rebuild_figures", "main",
    "get_device", "set_seed", "make_grid", "predict", "residual_at", "evaluate",
    "collect_artifacts", "run_summary", "config_for", "LOSS_THRESHOLDS",
]


def train(cfg: Config) -> tuple[PINN, TrainHistory]:
    return Trainer(cfg).run()


def write_breakdown_figures(cfg: Config, formats=figures.FORMATS) -> dict[str, list]:
    """Render the per-epoch collocation figures from a run's ``breakdown/`` CSVs.

    Lives here rather than in :mod:`pinn.viz` because it reads the breakdown
    through :mod:`pinn.history`, and that module imports torch; ``figures`` stays
    a pure array-in, figure-out module.
    """
    out = cfg.results_path / "figures"
    epochs, centres, values = residual_grid(cfg.results_path / "breakdown")
    summary = pd.read_csv(cfg.results_path / "point_summary.csv")
    figures.set_style()

    written = {
        "residual_evolution": figures.save_figure(
            figures.fig_residual_evolution(epochs, centres, values, cfg.label),
            out, "residual_evolution", formats),
        "residual_profiles": figures.save_figure(
            figures.fig_residual_profiles(epochs, centres, values, label=cfg.label),
            out, "residual_profiles", formats),
        "point_convergence": figures.save_figure(
            figures.fig_point_convergence(summary, cfg.label),
            out, "point_convergence", formats),
    }
    html = figures.write_residual_surface_html(
        epochs, centres, values, out / "residual_surface.html", cfg.label)
    if html is not None:
        written["residual_surface_html"] = [html]
    return written


def save_results(model: PINN, history: TrainHistory, cfg: Config) -> pd.DataFrame:
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.ckpt_path / "pinn.pt")
    if history.param_trail:
        extra = {}
        if history.weight_profiles:
            width = max(len(w) for w in history.weight_profiles)
            extra["weights"] = np.stack([np.pad(w, (0, width - len(w)), constant_values=np.nan)
                                         for w in history.weight_profiles])
        np.savez_compressed(cfg.ckpt_path / "param_trail.npz", epochs=np.asarray(history.param_epochs),
                            params=np.stack(history.param_trail), grads=np.stack(history.grad_trail), **extra)
    if history.min_w:
        pd.DataFrame({"iteration": range(len(history.min_w)), "min_w": history.min_w}).to_csv(
            cfg.ckpt_path / "causal.csv", index=False)
    if history.window_marks or history.eps_marks:
        rows = [(i, e, "eps") for i, e in history.eps_marks] + \
               [(i, float("nan"), "window") for i in history.window_marks]
        pd.DataFrame(rows, columns=["iteration", "eps", "kind"]).to_csv(cfg.ckpt_path / "marks.csv", index=False)
    if cfg.print_every:
        print(f"[save]  checkpoint + telemetry -> {cfg.ckpt_path}", flush=True)
        print(f"[save]  rendering figures -> {cfg.results_path} ...", flush=True)
    run = collect_artifacts(model, history, cfg)
    metrics = figures.write_run_report(run, cfg.results_path, data_dir=cfg.ckpt_path)

    summary = run_summary(model, history, cfg, run)
    summary.to_csv(cfg.results_path / "run_summary.csv", index=False, float_format=FLOAT_FORMAT)
    if history.snapshots is not None:
        history.snapshots.finalize(cfg.results_path)
        breakdown = write_breakdown_figures(cfg)
        figures.generate_run_extras(cfg.results_path)
        if cfg.print_every:
            print(f"[save]  {len(history.snapshots.epochs)} snapshots + point_summary.csv + "
                  f"{len(breakdown)} breakdown figures"
                  f"{'' if 'residual_surface_html' in breakdown else ' (no plotly: 3-D HTML skipped)'}",
                  flush=True)
    if cfg.print_every:
        print(f"[save]  wrote metrics.csv, run_summary.csv + "
              f"{len(list((cfg.results_path / 'figures').glob('*')))} figure files", flush=True)
    return metrics


def load_run(cfg: Config | None = None) -> tuple[PINN, TrainHistory, Config]:
    """Reload a finished run's trained weights and telemetry, with no retraining."""
    cfg = cfg or Config()
    device = get_device()
    from .pinn import build_model

    model = build_model(cfg).to(device=device, dtype=cfg.torch_dtype)
    try:
        state = torch.load(cfg.ckpt_path / "pinn.pt", map_location=device, weights_only=True)
    except TypeError:  # older torch
        state = torch.load(cfg.ckpt_path / "pinn.pt", map_location=device)
    model.load_state_dict(state)
    return model, TrainHistory.from_saved(cfg.ckpt_path), cfg


def rebuild_figures(cfg: Config | None = None) -> pd.DataFrame:
    """Redraw every figure from a finished run's checkpoint and saved CSVs, no retraining."""
    model, history, cfg = load_run(cfg)
    return figures.write_run_report(
        collect_artifacts(model, history, cfg), cfg.results_path, data_dir=cfg.ckpt_path
    )


def main(cfg: Config | None = None) -> tuple[PINN, TrainHistory, pd.DataFrame]:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    model, history = train(cfg)
    metrics = save_results(model, history, cfg)
    if cfg.print_every:
        print(f"[done]  total {time.perf_counter() - t0:.1f}s", flush=True)
    print(metrics.to_string(index=False))
    return model, history, metrics


if __name__ == "__main__":
    main()
