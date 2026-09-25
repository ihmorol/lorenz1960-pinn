"""All plotting for the PINN. Core code calls only what is exported here."""
from .figures import (FORMATS, RunArtifacts, fig_point_convergence, fig_residual_evolution,
                      fig_residual_profiles, generate_all, generate_sweep, invariant_series,
                      save_figure, set_style, write_residual_surface_html, write_run_report)

__all__ = ["FORMATS", "RunArtifacts", "fig_point_convergence", "fig_residual_evolution",
           "fig_residual_profiles", "generate_all", "generate_sweep", "invariant_series",
           "save_figure", "set_style", "write_residual_surface_html", "write_run_report"]

from pathlib import Path

import numpy as np
import torch

from . import evaluation, landscape, trajectory3d, training


def generate_run_extras(run_dir) -> list[Path]:
    """Training- and evaluation-phase figures beyond the standard suite, from a saved run."""
    from ..pinn import WindowedPINN
    from ..sweep import config_for
    from ..train import load_run, make_grid, predict, reference_trajectory, residual_at

    run = Path(run_dir)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_for(run)
    model, history, _ = load_run(cfg)
    p = next(model.parameters())
    written: list[Path] = []

    if (run / "breakdown").exists():
        written += trajectory3d.write_all(run)
    trail_path = run / "history" / "param_trail.npz"
    causal_weights = None
    if trail_path.exists():
        if cfg.n_windows == 1:
            written += landscape.write_all(run)
        with np.load(trail_path) as trail:
            trail_epochs = trail["epochs"]
            written += save_figure(training.fig_gradient_stability(trail_epochs, trail["grads"], history.window_marks),
                                   out, "gradient_stability", ("png",))
            sizes = [(n, p.numel()) for n, p in model.named_parameters()]
            written += save_figure(training.fig_gradient_histograms(trail_epochs, trail["grads"], sizes),
                                   out, "gradient_histograms", ("png",))
            if cfg.n_windows == 1:
                grid = make_grid(cfg, next(model.parameters()).device)
                spectra = {}
                for k in np.unique(np.linspace(0, len(trail_epochs) - 1, 4).round().astype(int)):
                    torch.nn.utils.vector_to_parameters(
                        torch.as_tensor(trail["params"][k], dtype=p.dtype, device=p.device), model.parameters())
                    spectra[int(trail_epochs[k])] = training.ntk_eigenvalues(model, grid)
                written += save_figure(training.fig_ntk_spectrum(spectra), out, "ntk_spectrum", ("png",))
                model, history, _ = load_run(cfg)
            if "weights" in trail:
                causal_weights = (trail_epochs, trail["weights"])

    loss = np.asarray(history.loss)
    written += save_figure(training.fig_loss_phases(loss, history.adam_iters, history.eps_marks,
                                                    history.window_marks, history.loss_phase),
                           out, "loss_phases", ("png",))
    if history.min_w:
        written += save_figure(training.fig_min_w(np.asarray(history.min_w), cfg.causal_delta,
                                                  history.eps_marks, history.window_marks), out, "min_w", ("png",))
        if causal_weights is not None:
            written += save_figure(training.fig_causal_weights(*causal_weights),
                                   out, "causal_weights", ("png",))
    if history.window_marks:
        written += save_figure(training.fig_window_grid(loss, history.window_marks), out, "window_grid", ("png",))
    if isinstance(model, WindowedPINN):
        written += save_figure(evaluation.fig_joint_continuity(model, model.edges), out, "joint_continuity", ("png",))
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
    pred = predict(model, t)
    joints = model.edges[1:-1] if isinstance(model, WindowedPINN) else []
    written += save_figure(evaluation.fig_error_vs_t(t, pred, ref, residual_at(model, t), joints),
                           out, "error_vs_t", ("png",))
    written += save_figure(evaluation.fig_error_growth(t, pred, ref), out, "error_growth", ("png",))
    return written
