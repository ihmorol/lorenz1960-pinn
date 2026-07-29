from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "lorenz1960_baseline"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    path = BASE_DIR / filename
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


def build_problem_setup() -> list[dict]:
    return [
        markdown_cell(
            """# 01 Problem Setup

This notebook defines the exact Lorenz-1960 initial value problem used for the FYDP numerical baseline. The goal is to freeze the equations, parameters, initial conditions, time interval, and numerical comparison grid before any solver-specific work begins.

## Research objective

Establish a trustworthy numerical baseline for the Lorenz-1960 ODE system that can later support ANN-based experiments. At this stage, the task is strictly numerical: define the benchmark correctly, solve it with a custom RK4 method and a standard scientific-library solver, and validate that both are consistent on the chosen interval.

## Verified local context used

- `lorenz_1960_equations_learning_guide.md`
- `paper1.txt` Section 4.2
- `fydp_research_structure_plan.md`

Those local sources agree on the benchmark setup:

- `k = 2`, `l = 1`
- `x(0) = 0.5`, `y(0) = 0.75`, `z(0) = 1`
- `t in [0, 1]`

## Lorenz-1960 equations

General form from the benchmark description:

\\[
\\frac{dx}{dt} = kl\\left(\\frac{1}{k^2+l^2}-\\frac{1}{k^2}\\right)yz,
\\]
\\[
\\frac{dy}{dt} = kl\\left(\\frac{1}{l^2}-\\frac{1}{k^2+l^2}\\right)xz,
\\]
\\[
\\frac{dz}{dt} = \\frac{kl^2}{2}\\left(\\frac{1}{k^2}-\\frac{1}{l^2}\\right)xy.
\\]

After substituting `k = 2` and `l = 1`, the system becomes:

\\[
\\frac{dx}{dt} = -0.1yz, \\qquad
\\frac{dy}{dt} = 1.6xz, \\qquad
\\frac{dz}{dt} = -0.75xy.
\\]

## Numerical configuration used in this baseline

- Uniform comparison grid: 1001 time points on `[0, 1]`
- Primary fixed RK4 step: `h = 1e-3`
- Library baseline: `scipy.integrate.solve_ivp`
- High-accuracy library settings: `method="DOP853"`, `rtol=1e-10`, `atol=1e-12`

## Why these settings were chosen

- The local project notes require the Lorenz-1960 benchmark from `paper1` Section 4.2, not the appendix code fragment.
- The interval `[0, 1]` is short and is the exact interval used in the local project framing.
- A 1001-point grid aligns naturally with RK4 step size `1e-3`, which makes solver outputs directly comparable without hidden interpolation in the primary comparison.
- A high-accuracy SciPy solve is used as the numerical reference for validation, but it is still treated as a numerical approximation rather than analytical truth.
"""
        ),
        code_cell(
            """import numpy as np
import pandas as pd

from lorenz1960_baseline import (
    DEFAULT_CONFIG,
    final_state_table,
    lorenz1960_coefficients,
    make_uniform_grid,
)
"""
        ),
        markdown_cell(
            """## Benchmark constants

The next cell makes every constant explicit so the downstream notebooks can be checked against the same setup."""
        ),
        code_cell(
            """config = DEFAULT_CONFIG
coefficients = lorenz1960_coefficients(config.k, config.l)
time_grid = make_uniform_grid(config.t_span, step=config.rk4_step)

problem_summary = pd.DataFrame(
    {
        "item": [
            "k",
            "l",
            "initial_state",
            "time_interval",
            "rk4_step",
            "n_eval_points",
            "scipy_method",
            "scipy_rtol",
            "scipy_atol",
            "coeff_dx",
            "coeff_dy",
            "coeff_dz",
        ],
        "value": [
            config.k,
            config.l,
            config.initial_state,
            config.t_span,
            config.rk4_step,
            config.n_eval,
            config.scipy_method,
            config.scipy_rtol,
            config.scipy_atol,
            coefficients[0],
            coefficients[1],
            coefficients[2],
        ],
    }
)
problem_summary
"""
        ),
        markdown_cell(
            """## Discretization check

This confirms that the chosen RK4 step size covers the interval exactly and produces the shared comparison grid."""
        ),
        code_cell(
            """pd.DataFrame(
    {
        "quantity": ["t_start", "t_end", "grid_points", "time_step"],
        "value": [time_grid[0], time_grid[-1], len(time_grid), time_grid[1] - time_grid[0]],
    }
)
"""
        ),
        markdown_cell(
            """## Libraries used

- `numpy` for vectorized numerical work
- `pandas` for small summary tables
- `matplotlib` for plotting in later notebooks
- `scipy.integrate.solve_ivp` for the scientific-library baseline

## Assumptions separated from verified facts

Verified from local project files:

- The Lorenz-1960 equation form
- Parameter values `k = 2`, `l = 1`
- Initial conditions `(0.5, 0.75, 1.0)`
- Time interval `[0, 1]`

Chosen numerical assumptions for this baseline:

- Fixed RK4 step `h = 1e-3`
- High-accuracy SciPy baseline using `DOP853`
- Output comparison on the same 1001-point grid

These choices are standard and simple, but they are still numerical design decisions rather than facts from the paper."""
        ),
    ]


def build_rk4_notebook() -> list[dict]:
    return [
        markdown_cell(
            """# 02 Custom RK4 Implementation

This notebook implements the classical fourth-order Runge-Kutta method from scratch and applies it to the fixed Lorenz-1960 benchmark defined in `01_problem_setup.ipynb`.

## Why RK4 is included

The custom RK4 solver is not included because it is more advanced than modern adaptive solvers. It is included because it is transparent. Every update is visible, reproducible, and easy to audit, which makes it a strong baseline for later ANN work."""
        ),
        code_cell(
            """import matplotlib.pyplot as plt
import pandas as pd

from lorenz1960_baseline import (
    DEFAULT_CONFIG,
    final_state_table,
    lorenz1960_rhs,
    plot_3d_trajectory,
    plot_state_time_series,
    rk4_solve,
)

plt.style.use("seaborn-v0_8-whitegrid")
"""
        ),
        markdown_cell(
            """## RK4 update rule

For step size `h`, RK4 computes four slope estimates:

\\[
k_1 = f(t_n, y_n), \\quad
k_2 = f\\left(t_n + \\frac{h}{2}, y_n + \\frac{h}{2}k_1\\right),
\\]
\\[
k_3 = f\\left(t_n + \\frac{h}{2}, y_n + \\frac{h}{2}k_2\\right), \\quad
k_4 = f(t_n + h, y_n + hk_3).
\\]

The state update is

\\[
y_{n+1} = y_n + \\frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).
\\]
"""
        ),
        code_cell(
            """config = DEFAULT_CONFIG
ts_rk4, ys_rk4 = rk4_solve(
    lorenz1960_rhs,
    config.t_span,
    config.initial_state,
    config.rk4_step,
)

final_state_table(ts=ts_rk4, ys=ys_rk4, solver_name="RK4")
"""
        ),
        markdown_cell(
            """## Time-series plots"""
        ),
        code_cell(
            """plot_state_time_series(
    ts_rk4,
    ys_rk4,
    title="Lorenz-1960 states from the custom RK4 solver",
)
plt.show()
"""
        ),
        markdown_cell(
            """## 3D trajectory plot"""
        ),
        code_cell(
            """plot_3d_trajectory(
    ys_rk4,
    title="Lorenz-1960 trajectory from the custom RK4 solver",
)
plt.show()
"""
        ),
        markdown_cell(
            """## Interpretation

- This notebook only demonstrates the custom fixed-step RK4 simulation.
- Agreement with a scientific-library solver is not claimed here.
- That validation is performed explicitly in `04_validation_and_comparison.ipynb`."""
        ),
    ]


def build_scipy_notebook() -> list[dict]:
    return [
        markdown_cell(
            """# 03 Python Solver Baseline

This notebook solves the same Lorenz-1960 initial value problem using SciPy's `solve_ivp`. The equations, parameters, initial conditions, and comparison grid are identical to the RK4 notebook.

## Solver choice

`solve_ivp` is a standard scientific-library interface for initial value problems. In this baseline, `method="DOP853"` is used because the local project notes emphasize high-accuracy numerical reference solutions, and this short benchmark interval is well suited to a high-order explicit method."""
        ),
        code_cell(
            """import matplotlib.pyplot as plt
import pandas as pd

from lorenz1960_baseline import (
    DEFAULT_CONFIG,
    final_state_table,
    make_uniform_grid,
    plot_3d_trajectory,
    plot_state_time_series,
    solve_lorenz1960_scipy,
)

plt.style.use("seaborn-v0_8-whitegrid")
"""
        ),
        markdown_cell(
            """## Run the SciPy baseline"""
        ),
        code_cell(
            """config = DEFAULT_CONFIG
t_eval = make_uniform_grid(config.t_span, step=config.rk4_step)
ts_scipy, ys_scipy, scipy_solution = solve_lorenz1960_scipy(config=config, t_eval=t_eval)
scipy_method = config.scipy_method

solver_summary = pd.DataFrame(
    {
        "field": ["method", "nfev", "status", "message"],
        "value": [
            scipy_method,
            scipy_solution.nfev,
            scipy_solution.status,
            scipy_solution.message,
        ],
    }
)
solver_summary
"""
        ),
        code_cell(
            """final_state_table(ts=ts_scipy, ys=ys_scipy, solver_name=f"SciPy {scipy_method}")
"""
        ),
        markdown_cell(
            """## Time-series plots"""
        ),
        code_cell(
            """plot_state_time_series(
    ts_scipy,
    ys_scipy,
    title=f"Lorenz-1960 states from SciPy {scipy_method}",
)
plt.show()
"""
        ),
        markdown_cell(
            """## 3D trajectory plot"""
        ),
        code_cell(
            """plot_3d_trajectory(
    ys_scipy,
    title=f"Lorenz-1960 trajectory from SciPy {scipy_method}",
)
plt.show()
"""
        ),
        markdown_cell(
            """## Interpretation

- The library solver uses the same initial value problem and the same output grid as the RK4 notebook.
- Numerical agreement with RK4 is evaluated separately in the comparison notebook so that the baseline claim is backed by explicit evidence rather than by inspection alone."""
        ),
    ]


def build_validation_notebook() -> list[dict]:
    return [
        markdown_cell(
            """# 04 Validation And Comparison

This notebook compares the custom RK4 solver against the SciPy baseline on the same Lorenz-1960 benchmark.

## Validation goals

- Compare trajectories visually
- Compare each state variable over time
- Quantify the discrepancy using error metrics
- Check whether the fixed-step RK4 result improves when the step size is reduced
- Judge whether the numerical baseline is reliable enough to support later ANN experiments
"""
        ),
        code_cell(
            """import matplotlib.pyplot as plt
import pandas as pd

from lorenz1960_baseline import (
    DEFAULT_CONFIG,
    compute_error_metrics,
    lorenz1960_rhs,
    make_uniform_grid,
    plot_3d_trajectory,
    plot_error_curves,
    plot_state_time_series,
    rk4_solve,
    solve_lorenz1960_scipy,
    subsample_uniform_solution,
)

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.float_format", lambda value: f"{value:.12e}")
"""
        ),
        markdown_cell(
            """## Run both solvers on the shared comparison grid"""
        ),
        code_cell(
            """config = DEFAULT_CONFIG
comparison_grid = make_uniform_grid(config.t_span, step=config.rk4_step)
scipy_method = config.scipy_method

ts_rk4, ys_rk4 = rk4_solve(
    lorenz1960_rhs,
    config.t_span,
    config.initial_state,
    config.rk4_step,
)
ts_scipy, ys_scipy, scipy_solution = solve_lorenz1960_scipy(config=config, t_eval=comparison_grid)

assert len(ts_rk4) == len(ts_scipy) == len(comparison_grid)
"""
        ),
        markdown_cell(
            """## Quantitative error metrics

The SciPy solution is used here as the numerical reference because it uses tighter adaptive error control. This does **not** make it analytical truth, but it is a reasonable benchmark for checking whether the custom RK4 implementation is behaving consistently."""
        ),
        code_cell(
            """metrics_primary = compute_error_metrics(reference=ys_scipy, candidate=ys_rk4)
metrics_primary
"""
        ),
        markdown_cell(
            """## Final-state difference"""
        ),
        code_cell(
            """final_state_comparison = pd.DataFrame(
    {
        "state": ["x", "y", "z"],
        "rk4_final": ys_rk4[-1],
        "scipy_final": ys_scipy[-1],
        "absolute_difference": abs(ys_rk4[-1] - ys_scipy[-1]),
    }
)
final_state_comparison
"""
        ),
        markdown_cell(
            """## Time-series overlay"""
        ),
        code_cell(
            """plot_state_time_series(
    ts_rk4,
    ys_rk4,
    title="RK4 and SciPy trajectories on the same grid",
    overlay=ys_scipy,
    overlay_name=f"SciPy {scipy_method}",
)
plt.show()
"""
        ),
        markdown_cell(
            """## 3D trajectory overlay"""
        ),
        code_cell(
            """plot_3d_trajectory(
    ys_rk4,
    title="3D trajectory comparison: RK4 vs SciPy",
    overlay=ys_scipy,
)
plt.show()
"""
        ),
        markdown_cell(
            """## Absolute error curves"""
        ),
        code_cell(
            """plot_error_curves(
    ts_rk4,
    reference=ys_scipy,
    candidate=ys_rk4,
    title="Absolute error of RK4 against the SciPy baseline",
)
plt.show()
"""
        ),
        markdown_cell(
            """## RK4 step-halving check

If the RK4 implementation is working correctly, reducing the fixed step size should reduce the discrepancy against the SciPy reference."""
        ),
        code_cell(
            """ts_rk4_fine, ys_rk4_fine = rk4_solve(
    lorenz1960_rhs,
    config.t_span,
    config.initial_state,
    config.rk4_step / 2.0,
)
if len(ts_rk4_fine[::2]) == len(ts_rk4) and (abs(ts_rk4_fine[::2] - ts_rk4) < 1e-15).all():
    ys_rk4_fine_on_primary_grid = ys_rk4_fine[::2]
else:
    ys_rk4_fine_on_primary_grid = subsample_uniform_solution(ts_rk4_fine, ys_rk4_fine, ts_rk4)

metrics_fine = compute_error_metrics(reference=ys_scipy, candidate=ys_rk4_fine_on_primary_grid)
metrics_fine
"""
        ),
        markdown_cell(
            """## Reliability discussion"""
        ),
        code_cell(
            """primary_rmse = float(metrics_primary.loc[metrics_primary["state"] == "combined_l2", "rmse"].iloc[0])
fine_rmse = float(metrics_fine.loc[metrics_fine["state"] == "combined_l2", "rmse"].iloc[0])
relative_gap = abs(fine_rmse - primary_rmse) / max(primary_rmse, fine_rmse, 1e-30)

if fine_rmse < primary_rmse * (1.0 - 1e-3):
    verdict = "Reducing the RK4 step size decreases the discrepancy, which supports the expected convergence behavior."
elif relative_gap <= 1e-3:
    verdict = "The step-halving check is essentially unchanged at this scale, which suggests the RK4-vs-SciPy discrepancy has already reached a numerical floor for this comparison."
else:
    verdict = "The step-halving check changed in an unexpected way, so the baseline should be investigated further before using it for ANN targets."

print("Primary combined RMSE:", primary_rmse)
print("Fine-step combined RMSE:", fine_rmse)
print("Relative gap:", relative_gap)
print("Reliability note:", verdict)
"""
        ),
        markdown_cell(
            """## Interpretation

Use the metrics and plots above for the baseline decision:

- If RK4 and SciPy agree closely on `[0, 1]`, the pipeline is trustworthy enough to generate ANN training targets for this benchmark.
- If the discrepancy is large or does not shrink under step halving, the RK4 implementation or the numerical settings should be revised before any ANN experiment begins.
- The SciPy solution is still a numerical approximation, so future work can tighten tolerances further or compare against additional reference settings if a supervisor asks for stronger validation."""
        ),
    ]


def main() -> None:
    BASE_DIR.mkdir(exist_ok=True)
    write_notebook("01_problem_setup.ipynb", build_problem_setup())
    write_notebook("02_rk4_implementation.ipynb", build_rk4_notebook())
    write_notebook("03_python_solver_baseline.ipynb", build_scipy_notebook())
    write_notebook("04_validation_and_comparison.ipynb", build_validation_notebook())


if __name__ == "__main__":
    main()
