"""All plotting for the PINN, and nothing else: every function here takes plain
arrays/frames (or a finished-run CSV) and returns figures or files.

Orchestration that decides *which* figures a run gets lives in
:mod:`pinn.functions.reporting`; this package never imports it, so plotting
stays an isolated leaf the core calls in one line.
"""
from .figures import (FORMATS, RunArtifacts, fig_point_convergence, fig_residual_evolution,
                      fig_residual_profiles, generate_all, generate_sweep, invariant_series,
                      save_figure, set_style, write_residual_surface_html, write_run_report)

__all__ = ["FORMATS", "RunArtifacts", "fig_point_convergence", "fig_residual_evolution",
           "fig_residual_profiles", "generate_all", "generate_sweep", "invariant_series",
           "save_figure", "set_style", "write_residual_surface_html", "write_run_report",
           "compare_runs"]

from pathlib import Path

from . import compare


def compare_runs(run_dirs, out=None) -> Path:
    return compare.write_page([Path(r) for r in run_dirs], out)
