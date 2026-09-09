# Implementation report: a PINN for the Lorenz-1960 system

FYDP-2 phase. This note records what was built, how it was trained, and what it
achieved. Plain-language background is in `docs/the-short-version.md`; the full
code walkthrough is in `docs/CODE_EXPLAINED.md`; the schema of every table and
figure the pipeline writes is in the project README.

## What was implemented

A forward physics-informed neural network that solves the Lorenz-1960 maximum
simplification equations (k = 2, l = 1) on t in [0, 1], starting from
u(0) = (0.5, 0.75, 1.0). The network is a fully connected MLP, 4 hidden layers
of 60 tanh units, built in PyTorch. Three design points matter:

1. The initial condition is enforced exactly through a trial solution,
   u(t) = u0 + g(t) N(t) with g(t) = (t - t0)/(tf - t0), so the loss contains
   only the ODE residual. A soft-penalty variant is implemented for comparison.
2. The time derivative in the residual comes from automatic differentiation,
   not finite differences.
3. The governing equations, coefficients, reference solver, and error metrics
   are all imported from the locked baseline module. The reference trajectory
   is used only for evaluation after training, never in the loss.

Training follows the forward-PINN protocol of Matthews & Bihlo (PinnDE, §4.1):
3000 Latin-hypercube collocation points, 20000 Adam iterations with the
learning rate decaying linearly from 1e-3 to 1e-4. L-BFGS polishing is
implemented and available through `Config(lbfgs_iters=...)`, but the reported
run is Adam-only.

Two capabilities were added after the first reported run:

- **Per-point telemetry.** Every `snapshot_every` epochs the full state of the
  collocation batch is written to `breakdown/epoch_<n>.csv`: for each of the
  3000 points, its time, the raw network output, the trial solution, the
  autograd derivative, the physics right-hand side, the residual, and that
  point's share of the loss. The tensors come from the training step's own
  residual evaluation, so recording them adds no forward or backward pass.
  Disabled by default, which leaves the run of record's code path unchanged.
- **An architecture sweep.** Nine networks, depth in {3, 4, 5} crossed with
  width in {50, 60, 70}, trained under otherwise identical settings.

## Results: the run of record

Errors against the DOP853 reference (rtol 1e-10) over 1001 evaluation points:

| state | MAE | RMSE | max abs. error |
|---|---|---|---|
| x | 5.20e-06 | 5.77e-06 | 9.24e-06 |
| y | 1.12e-05 | 1.36e-05 | 2.56e-05 |
| z | 8.27e-06 | 9.04e-06 | 1.25e-05 |
| combined L2 | 1.66e-05 | 1.73e-05 | 2.60e-05 |

The combined L2 row is not an average of the three states. It is computed from
the Euclidean error magnitude at each time point, sqrt(ex^2 + ey^2 + ez^2), so
it is larger than any single component. The average of the three per-state
RMSEs would be 9.48e-06.

The training loss (mean squared ODE residual over the 3000 collocation points)
falls from 2.17e-01 at the first iteration to 6.82e-09 at the last, a drop of
7.5 orders of magnitude. That figure is logged before the final optimiser step;
recomputing the residual from the saved checkpoint gives 6.79e-09. The state
variables span roughly 0.5 to 1.6, so the solution is accurate to about five
decimal places, obtained from the differential equations alone.

The run is deterministic: retraining from scratch with the default
configuration (seed 0) reproduces the metric table above to the last digit.
This was re-verified after the telemetry refactor.

### The loss normalisation

`residual.pow(2).mean()` averages over an (Nc, 3) tensor, so it divides by
3 x Nc — three residual components at each of the 3000 points — not by Nc. The
breakdown's `loss_contribution` column is defined as `r_sq / (3 Nc)` and summed
down a snapshot's 3000 rows it reproduces the reported loss exactly:
2.167403916e-01 against 2.167403996e-01 at the first epoch, agreeing to eight
significant figures (the residual to the difference is float32 accumulation).
`python run_epoch.py` prints this check.

## Results: the architecture sweep

Nine runs, one fixed seed (0), 20000 Adam epochs each, roughly 60 minutes of
training in total and 3.6 GB of per-point breakdown data.

| arch | params | RMSE (combined L2) | max abs. error | final loss | gen. gap | wall-clock |
|---|---|---|---|---|---|---|
| 3x50 | 5,353 | 1.80e-05 | 2.70e-05 | 9.76e-09 | 1.007 | 281 s |
| 3x60 | 7,623 | 1.29e-04 | 2.71e-04 | 6.06e-08 | 1.000 | 292 s |
| 3x70 | 10,293 | 3.30e-05 | 5.10e-05 | 2.95e-08 | 1.006 | 392 s |
| **4x50** | 7,903 | **9.62e-06** | 1.40e-05 | 2.98e-09 | 1.007 | 333 s |
| 4x60 | 11,283 | 1.73e-05 | 2.60e-05 | 6.82e-09 | 1.006 | 351 s |
| 4x70 | 15,263 | 1.70e-05 | 2.20e-05 | 7.90e-09 | 1.007 | 481 s |
| 5x50 | 10,453 | 9.60e-05 | 1.96e-04 | 2.31e-08 | 1.002 | 454 s |
| 5x60 | 14,943 | 1.40e-05 | 1.90e-05 | 6.71e-09 | 1.007 | 424 s |
| 5x70 | 20,233 | 1.50e-05 | 1.90e-05 | 1.04e-08 | 1.009 | 586 s |

**The result is not monotonic in either depth or width.** The spread from best
to worst is 13.4x, but it is not ordered: 3x60 is seven times worse than 3x50
and four times worse than 3x70; 5x50 is seven times worse than 5x60. The
smallest network in the grid (3x50, 5,353 parameters) beats three larger ones,
and the best cell is 4x50, not the largest.

Rank correlation between accuracy and model size is weak and does not reach
significance at n = 9 (Spearman rho = -0.45 against parameter count, -0.47
against depth, -0.05 against width; |rho| would need to exceed about 0.68).

**What this table does and does not support.** Each cell is a single run at one
seed. The seed fixes both the Xavier initialisation and the Latin-hypercube
draw, so a different seed would give a different but equally valid run of the
same architecture, and that run-to-run spread was not measured. The table
therefore supports statements of the form "4x50 reached an RMSE of 9.6e-06 on
seed 0". It does not support "4x50 is more accurate than 4x60": the
non-monotonic pattern above is the signature one expects when variation is
dominated by optimisation luck rather than capacity. Establishing an
architecture effect needs repeated seeds — `sweep(seeds=(0, 1, 2))` runs the
grid three times, about 4.5 hours, and enables the seed-robustness figure.

The one thing the fixed seed does buy is control: all nine architectures saw the
identical 3000 collocation points, so point i means the same t in every run.

### Physics quality, independent of the reference

Two measures do not depend on the reference solution at all, and both are
consistent across the whole grid:

- **Generalisation off the collocation set.** The mean squared residual on a
  dense uniform grid, divided by the same quantity on the 3000 training points,
  lies between 1.0001 and 1.0085 in all nine runs. The ODE holds essentially as
  well at times the network never trained on as at the ones it did, so the
  networks learned the equation rather than fitting 3000 points.
- **Conserved quantities.** Drift in the first quadratic invariant ranges from
  2.1e-05 to 1.3e-04 across the nine, tracking the error metrics. The DOP853
  reference drifts by at most 5.6e-11 on the same measure, which confirms the
  drift being reported is the network's and not an artefact of the measurement.

### What the per-point breakdown shows

The scalar loss curve cannot say *where* in the domain the physics is violated;
the 401 snapshots per run can.

**The hardest collocation points are at the domain boundaries.** In seven of the
nine runs the twenty worst final residuals sit at t between 0.0003 and 0.0065 —
the very start of the interval — and in the other two (3x60, 4x60) they sit at
t between 0.9967 and 0.9998, the very end. The median t of the twenty worst
points, across all nine runs, is 0.0034.

The t = 0 clustering follows from the trial solution. u_T = u0 + g(t) N(t) pins
the *value* at t = 0 exactly, for any parameters, but the *derivative* there is
du_T/dt = N(0), which is not constrained by the construction at all. The network
has to learn N(0) = f(u0) from the residual alone, and near the origin the
factor g(t) -> 0 suppresses the gradient signal that would teach it. This is a
structural property of the hard-IC formulation, not a training failure, and it
is invisible in any aggregate loss.

Convergence is also slow and uneven: the median point crosses |r| < 1e-4 only
around epoch 16000 of 20000, and between 133 (4x50) and 2232 (3x70) of the 3000
points never cross it at all.

## Verification

Eighteen pytest checks cover the exact initial condition, the residual
definition against the reference trajectory, loss reduction in both IC modes,
telemetry recording and its CSV round trip, the snapshot schema and its
reconstruction of the loss, the point summary and residual grid, the run-summary
columns, the sweep's output layout and resume behaviour, checkpoint reloading,
and generation of every figure and table. All pass.

Entry points: `run_pinn.py` (the run of record), `run_sweep.py` (all nine,
resumable), `run_single.py` (one architecture), `run_epoch.py` (one epoch, every
collocation point), `run_eval.py` (score a finished checkpoint, no retraining).

## Limitations

- Training runs in float32, which plausibly sets the error floor near 1e-5 to
  1e-6; no float64 comparison was run. The recorded residual columns are
  likewise float32-precise.
- Results are for a single initial value problem on one time window. A new
  initial condition requires retraining, and on this problem the classical
  solver remains much cheaper.
- The sweep is one seed per cell, and as argued above cannot separate an
  architecture effect from initialisation luck.
- The activation function was not swept; all nine runs use tanh.
- `peak_mem_mb` is recorded but is NaN throughout, as training ran on CPU and
  torch exposes no CPU equivalent of the CUDA allocator counter.
