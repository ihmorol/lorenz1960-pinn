"""All plotting for the PINN. Core code calls only what is exported here."""
from .figures import (FORMATS, RunArtifacts, fig_point_convergence, fig_residual_evolution,
                      fig_residual_profiles, generate_all, generate_sweep, invariant_series,
                      save_figure, set_style, write_residual_surface_html, write_run_report)

__all__ = ["FORMATS", "RunArtifacts", "fig_point_convergence", "fig_residual_evolution",
           "fig_residual_profiles", "generate_all", "generate_sweep", "invariant_series",
           "save_figure", "set_style", "write_residual_surface_html", "write_run_report"]
