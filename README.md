# A Physics-Informed Neural Network for the Lorenz-1960 System

Implementation and experimental record for the FYDP-2 phase of a final-year
design project at United International University. A physics-informed neural
network (PINN) solves the Lorenz-1960 maximum-simplification equations from the
governing equations alone, and is validated against a locked high-accuracy
numerical reference. This repository holds code, data, and figures only; no
manuscript or reference material.

## 1. Problem

Lorenz's 1960 maximum simplification of the barotropic vorticity equation reduces
the flow to three interacting spectral modes. With wavenumbers `k = 2`, `l = 1`:

```
dx/dt = -0.10 · y·z
dy/dt =  1.60 · x·z
dz/dt = -0.75 · x·y
```

integrated from `u(0) = (0.5, 0.75, 1.0)` over `t ∈ [0, 1]`. The coefficients are
derived once, in `src/baseline/lorenz1960_baseline.py`, and imported everywhere
else, so the equations exist in exactly one place in the repository.

The system is quadratically conservative: every form
`I = αx² + βy² + γz²` whose weight vector is orthogonal to the coefficient
vector `a` is conserved, since `İ = 2xyz(αa₁ + βa₂ + γa₃)`. Two independent
invariants exist, and their drift is used below as a physics-quality measure
that is completely independent of the reference solution.

## 2. Method

### 2.1 Trial solution

The network output is wrapped so the initial condition holds **exactly**, by
construction rather than by penalty:

```
u_T(t) = u₀ + g(t)·N(t),        g(t) = (t − t₀)/(t_f − t₀)
```

`N(t)` is the raw multilayer perceptron output. At `t = t₀`, `g = 0` and
`u_T = u₀` identically, to machine precision, for any parameter values. A
soft-penalty variant (`ic="soft"`, Raissi-style, weighted by `gamma`) is
implemented for comparison but is not the default.

### 2.2 Loss

The loss is the mean squared ODE residual and nothing else:

```
r(t) = du_T/dt − f(u_T),        L = (1/(3·N_c)) · Σᵢ Σⱼ r_j(tᵢ)²
```

The time derivative comes from automatic differentiation, not finite
differences. Note the `1/(3·N_c)` normalisation: the mean runs over all
`3 × N_c` residual entries — three state components at each of the `N_c`
collocation points — not over the `N_c` points alone. The `loss_contribution`
column of the per-point breakdown (Section 5.2) makes this explicit: it sums
across the `N_c` rows of a snapshot to exactly the reported loss.

**The reference solution never enters the loss.** It is used only to evaluate a
trained network, and every column derived from it is marked *evaluation-only* in
the schemas below.

### 2.3 Collocation

`N_c = 3000` points are drawn once by Latin hypercube sampling over `[t₀, t_f]`
and reused every epoch — training is full-batch, one gradient step per epoch over
all points. Because `Config.seed` drives both the weight initialisation and the
LHS draw, two runs at the same seed see the identical collocation design, which
is what makes the architecture comparison in Section 6 a controlled one.

### 2.4 Defaults

Following the forward-PINN protocol of Matthews & Bihlo (PinnDE, §4.1): 4 hidden
layers × 60 tanh units (11,283 trainable parameters), 3000 collocation points,
20,000 Adam iterations with the learning rate decaying linearly from 1e-3 to
1e-4. L-BFGS polishing is implemented but **off by default** (`lbfgs_iters=0`),
so the headline result is an Adam-only ablation. Set `Config(lbfgs_iters=5000)`
for the full recipe.

## 3. Repository layout

```
src/
  pinn/                     the PINN
    config.py                Config dataclass; reference trajectory accessors
    pinn.py                  MLP, trial solution, autograd residual, loss
    train.py                 training loop, evaluation, run summary, artifacts
    history.py               optimisation telemetry + per-point snapshot writer
    figures.py               figure suite (pure arrays in, figures out; no torch)
    sweep.py                 depth × width architecture sweep driver
    test_pinn.py             18 checks: sanity, training, telemetry, figures, sweep
    report.md                implementation report with the headline results
    results/                 run of record: metrics, figures, summary
    history/                 run of record: checkpoint and bulk telemetry
  baseline/                  the numerical reference (source of truth)
    lorenz1960_baseline.py   RK4 + SciPy solvers, coefficients, error metrics
    lorenz1960_solver.py     standalone RK4-vs-DOP853 validation script
runs/                        architecture sweep output (see Section 6)
notebooks/lorenz_pinn.ipynb  runnable notebook (Kaggle / Colab)
run_pinn.py                  entry point: the single run of record
run_sweep.py                 entry point: all nine architectures (resumable)
run_single.py                entry point: one architecture
run_epoch.py                 entry point: one epoch, every collocation point
run_eval.py                  entry point: score a finished model, no retraining
docs/CODE_EXPLAINED.md       line-by-line plain-language walkthrough
docs/the-short-version.md    short plain-words summary
```

`src/baseline/` is the source of truth for all numerical computation. The PINN
imports it and never modifies it.

## 4. Installation and use

```bash
pip install -r requirements.txt
```

Run the test suite:

```bash
python -m pytest
```

### 4.1 Entry points

Five scripts at the repository root, in decreasing order of how much they run.
Each is thin glue over the package; the logic they call is covered by the test
suite.

| Script | Scope | Cost | Writes |
|---|---|---|---|
| `run_sweep.py` | all nine architectures | ≈ 60–90 min, ≈ 3.6 GB | `runs/` |
| `run_single.py [depth width]` | one architecture | ≈ 5–8 min, ≈ 420 MB | `runs/<arch>/` |
| `run_pinn.py` | the run of record | ≈ 5.5 min | `src/pinn/results/`, `src/pinn/history/` |
| `run_epoch.py [n]` | one epoch, all `N_c` points | seconds | `runs/epoch_probe/` |
| `run_eval.py [run_dir]` | evaluate a finished model | seconds | nothing |

```bash
python run_sweep.py             # the full grid; skips runs already finished
python run_single.py 5 70       # just depth 5, width 70
python run_pinn.py              # the 4x60 run of record
python run_epoch.py             # print one epoch's 3000-point table
python run_eval.py runs/5x70    # reload a checkpoint and score it
```

**`run_sweep.py` resumes.** A run whose `run_summary.csv` exists is reused rather
than retrained — the runs are deterministic, so retraining one would only
reproduce it — while a run interrupted mid-training has no summary and is redone.
An hour-long sweep can therefore be stopped and restarted freely.

**`run_epoch.py`** is the one to reach for when the question is *what actually
happens in a single step*. It trains for `n` epochs (default 1) taking a snapshot
every epoch, then prints the whole `N_c`-row table — time, raw network output,
trial solution, autograd derivative, physics right-hand side, residual — followed
by the check that `loss_contribution` summed down the table reproduces the
reported loss. No figures: one epoch is too few points to plot.

**`run_eval.py`** never retrains. It reads the run's own `run_summary.csv` to
recover the network shape, loads the checkpoint, and reports the error table, the
mean squared residual on a dense 2001-point grid the network never trained on,
and the full 63-column summary. For the run of record it falls back to the
package defaults, which are that run's configuration.

Output paths are anchored to the repository root, not the working directory, so
all five work from anywhere. `pinn.train.rebuild_figures()` redraws every figure
of a finished run from its checkpoint and CSVs, and `pinn.train.load_run()`
returns its model and telemetry, both with no retraining.

## 5. Data products

Every run writes the same set of tables. Floating-point columns are formatted
`%.9g`, which round-trips float32 exactly — the precision the network actually
trains in — at roughly 40% fewer bytes than the 17-significant-digit default.

### 5.1 Per-run tables

| File | Rows | Contents |
|---|---|---|
| `metrics.csv` | 4 | MAE, RMSE, max abs. error per state plus a combined-L2 row |
| `run_summary.csv` | 1 | 63 columns: configuration, accuracy, physics, cost (Section 5.3) |
| `point_summary.csv` | `N_c` | one row per collocation point, aggregated over training |
| `history/loss_history.csv` | `epochs` | loss at every iteration |
| `history/training_diagnostics.csv` | `epochs / log_every` | loss decomposition, gradient norms per layer, update norm, learning rate |
| `history/reference_error.csv` | `epochs / eval_every` | MSE against the reference trajectory |
| `history/pinn.pt` | — | trained weights |

### 5.2 Per-epoch collocation breakdown

`breakdown/epoch_<n>.csv`, one file per snapshot epoch, `N_c` rows each. This is
the loop over collocation points made explicit: at snapshot epochs, the residual
of every individual point is written out before the optimiser step, so the
recorded values are exactly those the epoch's loss and gradient were built from.

Capture is free. `pinn.residual_parts()` returns every intermediate of the
residual evaluation the training step already performs; the writer detaches and
serialises them rather than recomputing anything. The only added cost is the CSV
write itself (~12 ms per snapshot).

| Group | Columns | Meaning |
|---|---|---|
| Index | `epoch`, `i`, `t` | snapshot epoch, point index `0..N_c−1`, its time |
| Raw network | `n_x`, `n_y`, `n_z` | `N(t)`, the MLP output before the trial wrapper |
| Trial solution | `x`, `y`, `z` | `u_T = u₀ + g(t)·N(t)`; the model's actual prediction |
| Autograd derivative | `dx_dt`, `dy_dt`, `dz_dt` | `du_T/dt` by automatic differentiation |
| Physics RHS | `f_x`, `f_y`, `f_z` | `f(u_T) = (a₁yz, a₂xz, a₃xy)` |
| Residual | `r_x`, `r_y`, `r_z` | `r = du_T/dt − f(u_T)` |
| Residual magnitude | `r_sq` | `r_x² + r_y² + r_z²` |
| Loss share | `loss_contribution` | `r_sq / (3·N_c)`; sums over rows to the epoch's loss |
| *Evaluation only* | `ref_x`, `ref_y`, `ref_z` | DOP853 reference, integrated at `t` |
| *Evaluation only* | `err_x`, `err_y`, `err_z`, `err_norm` | prediction minus reference |

The `ref_*` and `err_*` columns are computed after the fact and never enter the
loss. They are integrated directly at the collocation times rather than
interpolated from the uniform reference grid, whose ~1e-6 interpolation error
would be the same order as the error being measured.

Because the residual is computed in float32, reconstructing `r_x` as
`dx_dt − f_x` in float64 agrees only to float32 precision (~1e-7). The columns
are mutually consistent, not independently exact to their printed digits.

`breakdown/` is **gitignored** — at `snapshot_every=50` over 20,000 epochs it is
401 files and roughly 420 MB per run, 3.6 GB across the sweep. It regenerates
deterministically, and the tracked `point_summary.csv` and figures carry the
analysis drawn from it.

Three helpers read it back without loading the whole cube:

```python
from pinn.history import point_history, residual_grid, load_snapshots

point_history("runs/4x60/breakdown", i=1734)   # one point's 401-row trajectory
residual_grid("runs/4x60/breakdown")           # (epochs, t bins, |r|) field for plotting
load_snapshots("runs/4x60/breakdown")          # the entire cube, if memory allows
```

### 5.3 `point_summary.csv`

One row per collocation point, accumulated as the snapshots stream past, so no
second pass over the breakdown is needed.

| Column | Meaning |
|---|---|
| `i`, `t` | point index and its time |
| `r_init`, `r_final` | `‖r‖₂` at the first and last snapshot |
| `r_max`, `epoch_of_max` | worst residual reached during training, and when |
| `r_mean`, `r_std` | mean and standard deviation of `‖r‖₂` across snapshots |
| `loss_contribution_final` | this point's share of the final loss |
| `err_norm_final` | final distance from the reference (evaluation only) |
| `epoch_below_1e-4` | first snapshot epoch with `‖r‖₂ < 1e-4`; blank if never |
| `n_snapshots`, `rank_by_final_residual` | snapshots taken; 1 = worst final residual |

### 5.4 `run_summary.csv`

A single wide row per run — the row the sweep concatenates into
`runs/comparison.csv`.

| Group | Columns |
|---|---|
| Configuration | `arch`, `depth`, `width`, `n_params`, `activation`, `ic`, `gamma`, `epochs`, `lbfgs_iters`, `seed`, `n_collocation`, `lr_start`, `lr_end`, `t_start`, `t_end` |
| Accuracy | `mae_*`, `rmse_*`, `max_abs_error_*` for `x`, `y`, `z`, `combined_l2` |
| Normalised accuracy | `rel_mae_{x,y,z}` (MAE over the state's range), `r2_{x,y,z}` |
| Endpoint | `final_{x,y,z}`, `final_ref_{x,y,z}`, `final_err_{x,y,z}` |
| Error distribution | `err_norm_p50`, `err_norm_p90`, `err_norm_p99`, `err_norm_max` |
| Physics | `final_loss`, `best_loss`, `residual_mse_collocation`, `residual_mse_dense`, `residual_generalisation_gap` |
| Conservation | `invariant_{1,2}_max_drift`, `invariant_{1,2}_max_drift_reference` |
| Cost | `wall_clock_s`, `ms_per_epoch`, `epochs_to_1e-04`, `epochs_to_1e-06`, `epochs_to_1e-08`, `device`, `peak_mem_mb`, `n_snapshots` |

Two of these deserve comment.

`residual_generalisation_gap` is the mean squared residual on a dense uniform
grid divided by the same quantity on the training collocation points. It is the
only column that separates *"the network satisfies the ODE at the 3000 points it
was trained on"* from *"the network satisfies the ODE everywhere"*. A value near
1 means the physics constraint generalised off the collocation set.

`invariant_*_max_drift_reference` is a control: the DOP853 reference conserves
both quadratic invariants to solver precision, so this column should be
essentially zero. When it is, any drift in the PINN column is the network's, not
an artefact of the measurement.

`peak_mem_mb` is populated from the CUDA allocator and is `NaN` on CPU, where
torch exposes no equivalent counter.

## 6. Architecture sweep

`python run_sweep.py` trains nine networks — depth ∈ {3, 4, 5} crossed with width
∈ {50, 60, 70} — under otherwise identical settings, writing each to
`runs/<depth>x<width>/` with the same structure as the run of record:

```
runs/
  comparison.csv             one row per run, the 63 columns of Section 5.4
  figures/                   architecture_heatmap, architecture_scatter
  3x50/ … 5x70/
    metrics.csv  run_summary.csv  point_summary.csv  results.png
    figures/                 full per-run suite, including the breakdown figures
    history/                 checkpoint + bulk telemetry
    breakdown/               epoch_000000.csv … epoch_019999.csv  (gitignored)
```

`runs/<arch>/` is overwritten on rerun; the runs are deterministic at a fixed
seed, so a rerun reproduces the same bytes. **The sweep never writes to
`src/pinn/results/` or `src/pinn/history/`** — the run of record is left
exactly as committed, and a test enforces this.

### 6.1 A caveat on a single seed

The sweep uses one fixed seed per cell. The seed determines both the Xavier
initialisation and the LHS draw, so a different seed gives a different but
equally valid run of the same architecture. That run-to-run spread is not
measured here.

Consequently the comparison table supports statements of the form *"5×70 reached
an RMSE of X on seed 0"*, and **not** *"5×70 is more accurate than 4×60"* — a gap
between two cells cannot be distinguished from seed noise without repeated runs.
`sweep(seeds=(0, 1, 2))` runs three seeds per cell (27 runs, ≈ 4.5 h) and
enables the `seed_robustness` figure, which reports 95% intervals; the
single-activation and single-seed figures are skipped automatically when there
is nothing to compare.

The one thing the fixed seed buys is control: all nine architectures see the
identical 3000 collocation points, so point `i` means the same `t` in every run
and `point_summary.csv` is directly comparable across architectures.

## 7. Figures

Each figure is written as PNG (slides) and PDF (LaTeX), at 300 dpi in a
colourblind-safe serif style.

### Per-run

| Figure | Shows |
|---|---|
| `training_dynamics` | loss convergence, loss decomposition, LR schedule, physics loss vs true error |
| `gradient_diagnostics` | global and per-layer gradient norms, effective step size, gradient–loss coupling |
| `collocation_points` | LHS density, uniformity, spacing regularity, trained residual per point |
| `solution_vs_reference` | per-state solution and signed error against the locked baseline |
| `error_analysis` | error growth, distribution, parity plot with R², relative error |
| `phase_portraits` | x–y, x–z, y–z projections and the 3-D orbit |
| `metrics_summary` | grouped MAE/RMSE/max-error bars plus the metric table |
| `physics_residual` | ODE residual over the domain and its distribution |
| `invariant_drift` | drift in the two conserved quadratic forms, PINN vs reference |

### From the per-epoch breakdown

| Figure | Shows |
|---|---|
| `residual_evolution` | the whole run as one field: epoch × t, coloured by log₁₀‖r‖, beside the median and worst-point traces |
| `residual_profiles` | ‖r(t)‖ against t at six epochs, overlaid — the error front moving across the domain |
| `point_convergence` | histogram of the epoch each point crossed ‖r‖ < 1e-4, that epoch against t, and final vs worst residual per point |
| `residual_surface.html` | the same field as a rotatable 3-D surface (requires `plotly`; skipped silently if absent) |

### Sweep-level

| Figure | Shows |
|---|---|
| `architecture_heatmap` | depth × width grid of the chosen metric |
| `architecture_scatter` | accuracy against parameter count and against wall-clock cost, labelled per cell |
| `activation_comparison` | metric spread per activation — emitted only for multi-activation sweeps |
| `seed_robustness` | 95% intervals across seeds — emitted only for multi-seed sweeps |

## 8. Results (run of record)

Errors against the DOP853 reference (rtol 1e-10, atol 1e-12) over 1001
evaluation points, 4 hidden layers × 60 tanh units, Adam only:

| state | MAE | RMSE | max abs. error |
|---|---|---|---|
| x | 5.20e-06 | 5.77e-06 | 9.24e-06 |
| y | 1.12e-05 | 1.36e-05 | 2.56e-05 |
| z | 8.27e-06 | 9.04e-06 | 1.25e-05 |
| combined L2 | 1.66e-05 | 1.73e-05 | 2.60e-05 |

The combined-L2 row is **not** an average of the three states. It is computed
from the Euclidean error magnitude at each time point, `sqrt(eₓ² + e_y² + e_z²)`,
so it is necessarily larger than any single component; the mean of the three
per-state RMSEs is 9.48e-06.

The training loss falls from 2.17e-01 at the first iteration to 6.82e-09 at the
last — 7.5 orders of magnitude. That value is logged before the final optimiser
step; recomputing the residual from the saved checkpoint gives 6.79e-09. The
state variables span roughly 0.5 to 1.6, so the solution is accurate to about
five decimal places, obtained from the differential equations alone.

## 9. Reproducibility

The pipeline is deterministic. Retraining the default configuration from scratch
at seed 0 reproduces `src/pinn/results/metrics.csv` to the last digit, and this
was re-verified after the telemetry refactor of Section 5.2: `Config` defaults to
`snapshot_every=0`, so the run of record's code path is unchanged.

The test suite (18 checks) covers the exact initial condition, the residual
definition against the reference trajectory, loss reduction in both IC modes,
telemetry recording and its CSV round trip, the snapshot schema and its
reconstruction of the loss, the point summary and residual grid, the run-summary
columns, the sweep's output layout, resume behaviour, checkpoint reloading,
and generation of every figure and table.

## 10. Limitations

- Training runs in float32, which plausibly sets the error floor near 1e-5 to
  1e-6. No float64 comparison has been run.
- Results are for a single initial value problem on one time window. A new
  initial condition requires retraining, and on this problem a classical solver
  remains far cheaper.
- The architecture sweep uses one seed per cell; see Section 6.1 for what that
  does and does not support.
- The activation function is not swept. `Config.activation` accepts `tanh`,
  `relu`, `sigmoid`, `gelu`, and `swish`, and the sweep driver accepts them, but
  the reported grid varies depth and width only.
- `peak_mem_mb` is unavailable on CPU.

## 11. Configuration reference

Every knob is a field on `Config`, so no code edits are needed:

```python
from pinn.config import Config

Config(depth=3, width=70, activation="gelu", ic="soft", gamma=10.0,
       lbfgs_iters=5000, snapshot_every=50)
```

| Field | Default | Effect |
|---|---|---|
| `depth`, `width`, `activation` | 4, 60, `tanh` | network shape and nonlinearity |
| `ic`, `gamma` | `hard`, 1.0 | exact trial solution, or soft IC penalty with this weight |
| `epochs`, `lr_start`, `lr_end` | 20000, 1e-3, 1e-4 | Adam schedule (linear decay) |
| `lbfgs_iters` | 0 | L-BFGS polishing iterations after Adam |
| `n_collocation`, `seed` | 3000, 0 | LHS sample size; seeds both weights and the draw |
| `log_every`, `eval_every`, `print_every` | 10, 100, 250 | telemetry, reference-error, and console cadences |
| `snapshot_every` | 0 | epochs between full per-point snapshots; 0 disables |
| `results_dir`, `ckpt_dir`, `runs_dir` | see `config.py` | output roots, anchored to the repository root |

## 12. Kaggle / Colab

Open `notebooks/lorenz_pinn.ipynb`. The first cell locates the repository root,
changes into it, and puts `src/` on the import path; on Colab, uncomment the two
`git clone` lines. A GPU runtime is used automatically when available.

## Reference

Matthews, J. and Bihlo, A. *PinnDE: Physics-Informed Neural Networks for
Differential Equations*, §4.1 (forward-PINN protocol).

Lorenz, E. N. (1960). Maximum simplification of the barotropic vorticity
equation. *Tellus*, 12(3), 243–254.
