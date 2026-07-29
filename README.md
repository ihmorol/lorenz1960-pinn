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
  baseline/                the numerical reference solver
    lorenz1960_baseline.py RK4 + SciPy solvers, coefficients, error metrics (imported by fydp2)
    lorenz1960_solver.py   standalone RK4-vs-SciPy validation script
    generate_lorenz1960_baseline_notebooks.py   notebook generator for the baseline study
tests/test_pinn.py         sanity, training, and figure-suite checks
notebooks/lorenz_pinn.ipynb  runnable notebook (Kaggle / Colab)
results/fydp2/             tracked outputs: metrics.csv, results.png, figures/
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
this works from anywhere. Tracked results go to `results/fydp2/`; bulk telemetry
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

`results/fydp2/metrics.csv` reports MAE, RMSE, and max absolute error per state
variable plus a combined L2 row. `results.png` is the summary figure, and
`figures/` holds the full suite:

| Figure | Shows |
|---|---|
| `training_dynamics` | loss and reference error against epoch |
| `gradient_diagnostics` | global and per-layer gradient norms, update norm, LR |
| `collocation_points` | where the residual is enforced, and its magnitude there |
| `solution_vs_reference` | network solution against the trusted solver |
| `error_analysis` | per-component error over time |
| `phase_portraits` | trajectory projections, predicted vs reference |
| `metrics_summary` | the metric table as a figure |
| `invariant_drift` | drift in the system's conserved quantities |
| `physics_residual` | residual evaluated densely across the interval |

`figures.generate_sweep()` additionally renders architecture-heatmap,
activation-comparison, and seed-robustness plots from a tidy multi-run table.

> The `results/fydp2/` committed here is from the last full training run and
> predates the current figure suite, so it holds only `metrics.csv` and the
> earlier three-panel `results.png`. Run `python run_pinn.py` to regenerate the
> complete set.
