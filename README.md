# Lorenz-1960 PINN

A Physics-Informed Neural Network (PINN) that solves the Lorenz-1960 ODE system,
validated against an RK4 / SciPy DOP853 reference solver. This is the code for
the FYDP-2 phase of a final-year design project at UIU — implementation only, no
manuscript or reference material.

The system, with `k = 2`, `l = 1`:

```
dx/dt = -0.10 · y·z
dy/dt =  1.60 · x·z
dz/dt = -0.75 · x·y
```

with `u(0) = (0.5, 0.75, 1.0)` on `t ∈ [0, 1]`. The coefficients are computed in
`src/baseline/lorenz1960_baseline.py` and imported by the PINN, so the equations
live in exactly one place.

## Layout

```
src/
  fydp2/                   the PINN
    config.py              one Config dataclass: equations, IC, architecture, training
    pinn.py                MLP + hard/soft IC + autograd residual + loss
    train.py               training loop, evaluation vs the baseline, artifact collection
    history.py             per-iteration optimisation telemetry (gradients, LR, ref error)
    figures.py             the figure suite: per-run panels and architecture-sweep plots
    test_pinn.py           sanity, training, and figure-suite checks
    results/               tracked outputs: metrics.csv, results.png, figures/
  baseline/                the numerical reference solver
    lorenz1960_baseline.py RK4 + SciPy solvers, coefficients, error metrics (imported by fydp2)
    lorenz1960_solver.py   standalone RK4-vs-SciPy validation script
    generate_lorenz1960_baseline_notebooks.py   notebook generator for the baseline study
notebooks/lorenz_pinn.ipynb  runnable notebook (Kaggle / Colab)
run_pinn.py                entry point: train, evaluate, save results
docs/CODE_EXPLAINED.md     line-by-line plain-language walkthrough of the code
docs/adr/                  architecture decision records
```

`src/baseline/` is the source of truth for all numerical computation. The PINN
imports it and never modifies it; the reference trajectory is used only to
evaluate the trained network, never in the loss.

## Method

The network output is wrapped in a trial solution so the initial condition holds
exactly:

```
u_T(t) = u0 + g(t)·N(t),    g(t) = (t − t0)/(t_f − t0)
```

The loss is residual-only, `mean(r²)` with `r_j = du_j/dt − f_j(u)`, where the
derivative comes from autograd and `f` from the baseline coefficients.
Collocation points are a Latin hypercube sample over `[0, 1]`. A soft-IC mode
(`ic="soft"`, Raissi-style penalty weighted by `gamma`) is available for
comparison.

Defaults follow the forward-PINN protocol of Matthews & Bihlo (PinnDE), Section
4.1: 4 hidden layers × 60 units, tanh, 3000 collocation points, 20000 Adam steps
with polynomial LR decay from 1e-3 to 1e-4.

L-BFGS polishing after Adam is implemented but **off by default** —
`lbfgs_iters` defaults to `0`, for an Adam-only ablation. Set
`Config(lbfgs_iters=5000)` for the paper's full recipe.

## Install and run

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest
```

Train, evaluate, and write results:

```bash
python run_pinn.py
```

Output paths are anchored to the repository root, not the working directory, so
this works from anywhere. Tracked results go to `src/fydp2/results/`; bulk telemetry
(per-iteration loss, sampled diagnostics) and the checkpoint go to `data/fydp2/`,
which is gitignored because it is regenerable and large.

`fydp2.train.rebuild_figures()` redraws every figure from a finished run's
checkpoint and saved CSVs, with no retraining.

## Kaggle / Colab

Open `notebooks/lorenz_pinn.ipynb`. The first cell locates the repository root,
changes into it, and puts `src/` on the import path; on Colab, uncomment the two
`git clone` lines. A GPU runtime is used automatically when available.

## Configuration

Every knob is a field on `Config` — no code edits needed:

```python
from fydp2.config import Config
Config(depth=3, width=60, activation="gelu", ic="soft", gamma=10.0, lbfgs_iters=5000)
```

Activations: `tanh`, `relu`, `sigmoid`, `gelu`, `swish`. IC modes: `hard`, `soft`.
`log_every` and `eval_every` control how often gradient diagnostics and
reference-error evaluations are recorded.

## Results

`src/fydp2/results/metrics.csv` reports MAE, RMSE, and max absolute error per state
variable plus a combined L2 row. `results.png` is the summary figure, and
`figures/` holds the full suite:

Each figure is written as PNG (for slides) and PDF (for LaTeX).

| Figure | Shows |
|---|---|
| `training_dynamics` | loss convergence, loss decomposition, LR schedule, physics loss vs true error |
| `gradient_diagnostics` | global and per-layer gradient norms, effective step size, gradient-loss coupling |
| `collocation_points` | LHS density, uniformity, spacing regularity, trained residual per point |
| `solution_vs_reference` | per-state solution and signed error against the locked baseline |
| `error_analysis` | error growth, distribution, parity plot with R², relative error |
| `phase_portraits` | x–y, x–z, y–z projections and the 3-D orbit |
| `metrics_summary` | grouped MAE/RMSE/max-error bars plus the metric table |
| `physics_residual` | ODE residual over the domain and its distribution |
| `invariant_drift` | drift in the two conserved quadratic forms, PINN vs reference |

Bulk telemetry for the run lands in `data/fydp2/` (gitignored): `pinn.pt`,
`loss_history.csv`, `training_diagnostics.csv`, `reference_error.csv`.

`figures.generate_sweep(df, outdir)` additionally renders the depth × width
heatmap, activation comparison, and seed-robustness plots from a tidy table with
one row per run (`depth`, `width`, `activation`, `seed`, and a metric column).
These are ready for the plain-ANN architecture search; nothing calls them yet.
