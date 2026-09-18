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

**Evaluating all `N_c` points in one batched pass is exact, not an
approximation.** The residual system is diagonal in the collocation points: the
right-hand side `f(u(tᵢ))` uses only `u(tᵢ)`, and `du_T/dt(tᵢ)` is obtained by
automatic differentiation of a network that takes the single scalar `tᵢ` as
input. No collocation point can influence another one's residual, so if
`ℓᵢ = ‖r(tᵢ)‖²/3` then

```
L = (1/N_c) · Σᵢ ℓᵢ,            ∇L = (1/N_c) · Σᵢ ∇ℓᵢ
```

holds as an identity. Looping over the points one at a time and accumulating
their gradients therefore produces the same gradient vector as one batched
backward pass, differing only by float32 summation round-off. `run_trajectory.py
--experiment` measures this: on the 4×60 network with 3000 points the two losses
agree to 4e-8 relative and the two gradients to 8e-7 relative with a cosine
similarity of 1.0000001, against a float32 round-off scale of ~1e-5
(see `runs/<arch>/trajectory/tables/batching_equivalence.csv`).

What the batched pass *does* fix is the optimiser schedule: one step per epoch
is a full-batch gradient step. Iterating instead means choosing between taking
one step per point — 3000 rank-one gradient estimates per epoch, each of which
costs ~1.9 ms of fixed per-step overhead on this problem, roughly 400× the
wall-clock of the single 14 ms full-batch step that sees all 3000 points — or
accumulating over the loop, which reproduces the batched gradient exactly and so
buys nothing. Batching is therefore exact, and iterating is not a more correct
algorithm but a *different* one: stochastic rather than full-batch gradient
descent. Section 7 (*Batching experiments*) measures what that difference
actually costs and buys. What genuinely limits convergence here is not the batch
structure but that the loss weights every `t` equally, so error at late `t` is as
expensive as error at the initial condition; Section 10 lists the
causal-weighting remedies.

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
    trajectory.py            trajectory capture + the batching experiments
    trajectory_figures.py    trajectory-across-epochs figures (pure plotting)
    marching.py              time-marching PINN: staged IC chain + seam diagnostic
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
run_trajectory.py            entry point: the trajectory across epochs, + batching
run_marching.py              entry point: time marching, one stage at a time
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

Seven scripts at the repository root, in decreasing order of how much they run.
Each is thin glue over the package; the logic they call is covered by the test
suite.

| Script | Scope | Cost | Writes |
|---|---|---|---|
| `run_sweep.py` | all nine architectures | ≈ 60–90 min, ≈ 3.6 GB | `runs/` |
| `run_single.py [depth width]` | one architecture | ≈ 5–8 min, ≈ 420 MB | `runs/<arch>/` |
| `run_pinn.py` | the run of record | ≈ 5.5 min | `src/pinn/results/`, `src/pinn/history/` |
| `run_trajectory.py` | the trajectory across epochs, from a run or a fresh one | seconds to minutes | `runs/<arch>/trajectory/` |
| `run_marching.py` | time marching: `n_segments` stages × epochs | ≈ 2 min per 20 000 epochs | `runs/marching_<n>x<arch>/` |
| `run_epoch.py [n]` | one epoch, all `N_c` points | seconds | `runs/epoch_probe/` |
| `run_eval.py [run_dir]` | evaluate a finished model | seconds | nothing |

```bash
python run_sweep.py             # the full grid; skips runs already finished
python run_single.py 5 70       # just depth 5, width 70
python run_pinn.py              # the 4x60 run of record
python run_epoch.py             # print one epoch's 3000-point table
python run_eval.py runs/5x70    # reload a checkpoint and score it
```

```bash
# the trajectory of all 3000 collocation points across training, from a
# finished run's breakdown CSVs — no retraining, nothing to recompute
python run_trajectory.py --run runs/3x50 --animate

# a denser cadence than the finished runs have, from a fresh short run
python run_trajectory.py --train --epochs 2000 --every 1

# plus the batched-versus-iterated gradient measurement and batch-size curves
python run_trajectory.py --run runs/3x50 --experiment
```

```bash
# time marching: 8 stages x 2500 epochs = the same 20000-epoch budget as the
# sweep, diffed against the single-domain 4x60 run
python run_marching.py --segments 8 --epochs-per-stage 2500 --compare runs/4x60

# plus a joint fine-tune of all stages, which frees the interior anchors
python run_marching.py --segments 8 --epochs-per-stage 2500 --finetune-epochs 5000 --compare runs/4x60
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

**`run_trajectory.py`** answers *what does the whole collocation batch look like
at epoch k, and how does it move?* In `--run` mode it reads a finished run's
`breakdown/epoch_*.csv` and needs no retraining — the nine runs under `runs/`
already carry 401 snapshots each over the full 20 000 epochs, which is where the
figures in Section 7 come from. In `--train` mode it runs a shorter schedule at a
denser cadence: it sets `Config.snapshot_every` (default every epoch) and calls
the stock `train()` unchanged, then reads the CSVs the existing snapshot writer
produced. The cost of a dense cadence is disk, not compute — one ~1.3 MB CSV per
snapshot, so every epoch over 2000 epochs is ~2.6 GB.

Both modes write `trajectory.npz`, a per-epoch scalar CSV, a figure suite and
optionally a GIF (Section 7). `--experiment` adds the batching measurements of
Section 2.3. It writes only inside `<run dir>/trajectory/`, and the training loop
itself is never modified, so the run-of-record pipeline cannot be disturbed.

Output paths are anchored to the repository root, not the working directory, so
all six work from anywhere. `pinn.train.rebuild_figures()` redraws every figure
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

### Time marching (`run_marching.py`)

An alternative scheme, built to be tested rather than assumed. The window is split
into stages; stage 0 gets the problem's initial condition, and every later stage
gets the *previous stage's own right endpoint* as its initial condition, frozen.
The network is shared, so the piecewise solution is stitched from one set of
weights, and each stage's trial transform is only valid inside its own stage —
hence `predict_marching`, which routes every point to its own stage.

Three details decide whether this works, and the module is explicit about all
three: the anchor is the network's own output, never the reference (marching on
the true endpoint state would be circular); the anchor is frozen, so the piecewise
solution develops a **jump** at each boundary; and that jump is measured, because
it is error the stitching introduced and no average over `t` will reveal it.

Measured on the 4×60 network, 3000 collocation points, same seed as the sweep:

| scheme | stages | epochs | wall-clock | RMSE vs reference | max abs. error | dense residual | max seam |
|---|---|---|---|---|---|---|---|
| single-domain | 1 | 20 000 | 351 s | **1.70e-05** | 2.60e-05 | **6.83e-09** | — |
| marching, sequential | 8 | 20 000 | 114 s | 0.388 | 0.643 | 3.78e-03 | 0.116 |
| marching, 8× data | 8 | 160 000 | 845 s | 0.388 | 0.643 | 4.12e-03 | 0.117 |
| marching + fine-tune | 8 | 25 000 | 270 s | 0.222 | 0.355 | 1.46e-06 | 0.060 |

**Marching is not competitive here, by four orders of magnitude.** Two things rule
out the obvious excuses. Giving each stage eight times more data (160 000 epochs,
2.4× the single-domain wall-clock) changed the RMSE from 0.3882 to 0.3877 —
nothing — so this is not under-training. And the per-stage RMSE grows
monotonically down the chain (0.035 → 0.091 → 0.147 → 0.201 → 0.253 → 0.302 →
0.349 → 0.363): each stage inherits a worse anchor and adds its own error. That is
classical error accumulation, rebuilt from scratch.

The diagnostics make the mechanism visible in a way the headline number does not.
Stage residual losses fell to 9.6e-17 and 3.5e-13 in the matched-data run — each
stage drove its *own* residual to machine-flat while the global solution was 0.39
wrong. A stage-local loss is not a proxy for global accuracy, because it cannot
see the anchor error it inherited.

The fine-tune is the mitigation: it frees the interior anchors and optimises all
stages jointly so they can close their own gaps. It works in the direction
predicted — seams halve (0.117 → 0.060), RMSE improves 1.75× — and confirms the
diagnosis, but it cannot close a four-order-of-magnitude gap.

The one bug worth knowing about, because it fails *quietly*: freeing the anchors
must exclude anchor 0. If it does not, the optimiser moves the initial condition,
and since the loss is the ODE residual and nothing else it will happily do so — the
run then reports a residual of 1.67e-07 alongside a solution error of 0.334,
because it converged to a different initial value problem. `_finetune` pins
anchor 0 with a gradient hook, and a test holds that.

This does not make time marching a bad technique. It makes it the wrong tool for a
smooth ODE on a single unit interval, which is what Section 2.3 and Section 10
already argue from the error profile. It becomes the right tool when the horizon
grows, or when the dynamics get stiff enough that one network cannot represent
them — neither of which is true here.

### Sweep-level

| Figure | Shows |
|---|---|
| `architecture_heatmap` | depth × width grid of the chosen metric |
| `architecture_scatter` | accuracy against parameter count and against wall-clock cost, labelled per cell |
| `activation_comparison` | metric spread per activation — emitted only for multi-activation sweeps |
| `seed_robustness` | 95% intervals across seeds — emitted only for multi-seed sweeps |

### Trajectory across epochs (`run_trajectory.py`)

The whole 3000-point collocation batch, at every snapshot epoch, in the problem's
own units. Nothing in this set is normalised or rescaled: the question these
figures answer is how the trajectory moves on the scale the physics lives on.

| Figure | Shows |
|---|---|
| `trajectory_3d_panels` | the orbit in (x, y, z) at six geometrically spread epochs, on one fixed axes box, each panel labelled with its RMSE |
| `trajectory_3d_final` | the same, zoomed to the last 20% of training, where the box closes in on the final orbit shape |
| `trajectory_3d_points` | the 3000 individual collocation points as a scatter rather than a joined curve |
| `trajectory_projections` | every epoch's orbit on the x–y, x–z, y–z planes, coloured by epoch; the fan collapsing onto the reference *is* the convergence |
| `trajectory_projections_true_aspect` | the same panels at equal aspect, showing the true geometry of a trajectory that is far longer in y than in x |
| `trajectory_time_slices` | x(t), y(t), z(t), one curve per epoch — where in `t` the deformation happens and what lagged behind |
| `trajectory_extent` | the batch's min/max envelope per state against the reference band, 3000 points collapsed into two curves |
| `trajectory_animation.gif` | the orbit and its three projections animating over training, axes fixed so only the prediction moves |

Beside them, `--experiment` writes `batching_equivalence` and `batching_cost`
under `figures/`, with their tables under `tables/`: the first shows the batched
and per-point gradients agreeing to float32 round-off, the second puts full-batch
and mini-batch training on a shared point-evaluation axis. `trajectory_epochs.csv`
holds the per-epoch scalars (RMSE, per-state min/max/spread) behind the figures.

#### Batching experiments

Both figures come from `python run_trajectory.py --run runs/3x50 --experiment`
on the 4×60 network with 3000 collocation points. The arms differ only in batch
size, all starting from the same initialisation at a constant `lr = 1e-3` (the
project's own decaying schedule is not used here, so the absolute losses are not
the protocol's — the cost ratios are unaffected).

**The batched gradient is the per-point gradient.** One backward pass over all
3000 points against 3000 single-point backward passes accumulated by hand:

| quantity | value |
|---|---|
| relative difference in the loss | 3.9e-08 |
| relative difference in the gradient | 8.3e-07 |
| cosine similarity of the two gradients | 1.00000012 |
| float32 ε | 1.2e-07 |
| round-off scale √(3N_c)·ε | 1.1e-05 |

Both differences sit inside the round-off scale, so the two are the same
computation. Batching is not an approximation of anything.

**Iterating is nevertheless a different algorithm, and the difference is real.**
Cost per collocation-point evaluation, and where each arm got to:

| batch | steps | point-evals | wall-clock | µs per point-eval | best full-batch loss |
|---|---|---|---|---|---|
| full (3000) | 1 000 | 3 000 000 | 15.2 s | 5.1 | 2.97e-06 |
| 256 | 11 718 | 2 999 808 | 51.9 s | 17.3 | 2.84e-07 |
| 32 | 68 680 | 2 197 760 | 240.0 s | 109.2 | 3.07e-08 |
| 1 | 83 852 | 83 852 | 240.0 s | 2 862.2 | 1.20e-07 |

The two effects point in opposite directions, and both are real:

- **Per unit of data, mini-batching wins.** The single-point arm reached 1.2e-07
  after 84 k point evaluations; at that same data exposure the full-batch arm had
  taken 28 steps and was still at ~9e-02. A per-point gradient is a poor estimate
  of the full gradient, but there are 3000× more of them per epoch's worth of
  data, and on this problem that buys six orders of magnitude.
- **Per unit of wall-clock, full-batch wins, by a lot.** A step costs 5.1 µs per
  point at full batch and 2 862 µs per point at batch size 1 — 566× — because the
  fixed per-step overhead (backward setup plus the optimiser update on 11 283
  parameters) dominates once the batch is small. At 15.2 s the full-batch arm was
  at 2.97e-06; the single-point arm, at 5 300-odd steps, was near 2e-04.

For this project the full batch is the right trade: it reaches the float32 noise
floor (RMSE ~1e-5) in 1 000 epochs and 15 s, and the protocol's 20 000 epochs
cost ~5 minutes. Exposing those same 20 000 epochs at one point per step would
take ~60 M steps, around 47 hours. So the honest reading of the supervisor's
concern is: batching is **exact**, not wrong; iterating is a *different* optimiser
(SGD rather than full-batch GD) that is more data-efficient but far more
expensive here, and it is not a cheaper route to the accuracy already achieved.

Two things worth knowing when reading `trajectory_3d_panels`: the epoch-0
prediction overshoots in x (it reaches ≈ 0.74 where the reference never exceeds
0.5), so the shared box has to be wide and the converged orbit sits narrow inside
it — that is why `trajectory_3d_final` exists. And because the batch is a finite
sample of `t`, joining its points traces the orbit at 3000-point resolution, not
the 1001-point resolution of the reference.

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
- **The loss weights every `t` equally.** Information about the solution enters
  only through the initial condition, but the residual at `t = 1` is weighted the
  same as at `t = 0`, so the network can drive the late-time residual down while
  the early-time solution is still wrong — a documented failure mode of forward
  PINNs. This, not the batch structure (Section 2.3), is where the "each point
  should be trained in turn" intuition points. The standard remedies — causal
  loss weighting that ramps later windows in as earlier ones converge, adaptive
  per-point weights, or residual-based adaptive resampling of the collocation
  points — are not implemented here, and the per-epoch breakdown in
  `runs/<arch>/trajectory/` is the diagnostic that would show whether they are
  needed, since it exposes which `t` regions stay hard and for how long.

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

Cited in Section 10 as the remedies for uniform time weighting, none of them
implemented here:

- Wang, S., Sankaran, S. and Perdikaris, P. (2022). *Respecting causality is all
  you need for training physics-informed neural networks.* arXiv:2203.07404.
- McClenny, L. and Braga-Neto, U. (2023). *Self-adaptive physics-informed neural
  networks.* Journal of Computational Physics, 474, 111722. (arXiv:2009.04544)
- Wu, C., Zhu, M., Tan, Q., Kartha, Y. and Lu, L. (2023). *A comprehensive study
  of non-adaptive and residual-based adaptive sampling for physics-informed
  neural networks.* Computer Methods in Applied Mechanics and Engineering, 403,
  115671. (arXiv:2207.10289)
