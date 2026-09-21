"""All plotting for the PINN. Core code calls only what is exported here."""
from .figures import (FORMATS, RunArtifacts, fig_point_convergence, fig_residual_evolution,
                      fig_residual_profiles, generate_all, generate_sweep, invariant_series,
                      save_figure, set_style, write_residual_surface_html, write_run_report)

__all__ = ["FORMATS", "RunArtifacts", "fig_point_convergence", "fig_residual_evolution",
           "fig_residual_profiles", "generate_all", "generate_sweep", "invariant_series",
           "save_figure", "set_style", "write_residual_surface_html", "write_run_report"]

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import evaluation, landscape, trajectory3d, training


def generate_run_extras(run_dir) -> list[Path]:
    """Training- and evaluation-phase figures beyond the standard suite, from a saved run."""
    from ..sweep import config_for
    from ..train import load_run, make_grid, predict, reference_trajectory, residual_at

    run = Path(run_dir)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_for(run)
    model, history, _ = load_run(cfg)
    dtype = next(model.parameters()).dtype
    written: list[Path] = []

    if (run / "breakdown").exists():
        written += trajectory3d.write_all(run)
    trail_path = run / "history" / "param_trail.npz"
    if trail_path.exists():
        written += landscape.write_all(run)
        trail = np.load(trail_path)
        written += save_figure(training.fig_gradient_stability(trail["epochs"], trail["grads"]),
                               out, "gradient_stability", ("png",))
        sizes = [(n, p.numel()) for n, p in model.named_parameters()]
        written += save_figure(training.fig_gradient_histograms(trail["epochs"], trail["grads"], sizes),
                               out, "gradient_histograms", ("png",))
        grid = make_grid(cfg, next(model.parameters()).device)
        spectra = {}
        for k in np.unique(np.linspace(0, len(trail["epochs"]) - 1, 4).round().astype(int)):
            torch.nn.utils.vector_to_parameters(torch.as_tensor(trail["params"][k], dtype=dtype),
                                                model.parameters())
            spectra[int(trail["epochs"][k])] = training.ntk_eigenvalues(model, grid)
        written += save_figure(training.fig_ntk_spectrum(spectra), out, "ntk_spectrum", ("png",))
        model, history, _ = load_run(cfg)

    loss = np.asarray(history.loss)
    written += save_figure(training.fig_loss_phases(loss, history.adam_iters, history.eps_marks),
                           out, "loss_phases", ("png",))
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
    pred = predict(model, t)
    joints = getattr(model, "edges", [])[1:-1]
    written += save_figure(evaluation.fig_error_vs_t(t, pred, ref, residual_at(model, t), joints),
                           out, "error_vs_t", ("png",))
    written += save_figure(evaluation.fig_error_growth(t, pred, ref), out, "error_growth", ("png",))
    return written


def compare_runs(run_dirs, out=None) -> Path:
    from . import compare
    return compare.write_page([Path(r) for r in run_dirs], out)


def precision_floor(run32, run64, out) -> list[Path]:
    l32 = pd.read_csv(Path(run32) / "history" / "loss_history.csv").loss.to_numpy()
    l64 = pd.read_csv(Path(run64) / "history" / "loss_history.csv").loss.to_numpy()
    return save_figure(evaluation.fig_precision_floor(l32, l64), Path(out), "precision_floor", ("png",))
