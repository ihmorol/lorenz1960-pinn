# Three causal-window runs: result and code audit

This compares the completed historical `runs/4x60_f64_unit_win27_causal_warm/` run
with the 512-point and 1500-point candidates under `runs/causal-window/`.
It uses the saved CSVs and the checked-in training code; no model was retrained.
The 512-point artifacts are on `codex/causal-512pt`; this branch contains only
the 1500-point candidate. Full per-step breakdown CSVs and resumable progress
state remain local because they are large generated artifacts; the final model
checkpoint, config, compact histories, point summary, and figures are included.
The local Python environment lacks PyTorch, so checkpoints could not be
independently evaluated here.

## What is comparable

All three use 27 sequential 4-hidden-layer, width-60 tanh networks (304,641
trainable parameters), float64, hard initial conditions, unit time scaling,
warm starts, uniform sampling, the same initial state and Lorenz-1960
coefficients, the same time interval `[0, 13.26446]`, and 13,265 evaluation
points. The reference is SciPy DOP853 with `rtol=1e-10`, `atol=1e-12`.
Each window's endpoint state anchors the next one, enforcing value continuity.

| Setting | Historical | 512-point | 1500-point |
|---|---:|---:|---:|
| Total collocation points | 39,793 (~1,474/window) | 13,825 (~512/window) | 40,474 (~1,499/window) |
| Epsilon schedule / threshold | 0.01, 0.1, 1, 10 / 0.99 | same | same |
| Adam maximum per epsilon stage | 4,000 | 4,000 | 4,000 |
| L-BFGS maximum iterations per window | 1,500 | 1,500 | 500 |
| Adam StepLR | 0.9 every 5,000 steps | 0.9 every 1,000 | 0.9 every 1,000 |
| Recorded device | CUDA | CUDA | CPU |

The historical run used older code. The newer code adds the outgoing endpoint
to each nonfinal window's training set and checks causal threshold attainment
*after* the last Adam update. It also repairs progress logging and final-loss
reporting. The 512-versus-1500 comparison changes point count **and** L-BFGS
budget and device. None of the pairwise comparisons isolates one treatment.

`epochs=20000` is a common but unused generic setting for these causal runs.
Their actual Adam work is governed by four epsilon stages, a 4,000-step cap per
stage, and the causal stopping test. `lr_end=1e-4` is also inactive because
StepLR is selected. The historical per-window Adam count never reached 5,000,
so its scheduled LR never decayed; the new schedule did decay in 5 windows for
the 512-point run and 12 windows for the 1500-point run.

## Final accuracy and physics

All error metrics below use the same saved 13,265-point reference grid. The
combined RMSE is `sqrt(mean(||prediction-reference||_2^2))`, rather than the
mean of the three component RMSEs. Residual MSE averages the three squared ODE
residual components. Smaller is better in this table.

| Metric | Historical | 512-point | 1500-point |
|---|---:|---:|---:|
| Combined state MAE | 7.094e-5 | 1.037e-4 | 8.273e-5 |
| **Combined state RMSE** | **8.435e-5** | 1.569e-4 | 1.078e-4 |
| Maximum state-error norm | **2.355e-4** | 5.878e-4 | 3.388e-4 |
| Median state-error norm | 6.489e-5 | 8.562e-5 | **5.426e-5** |
| 99th-percentile state-error norm | **2.220e-4** | 5.491e-4 | 2.786e-4 |
| `x` RMSE | **2.840e-5** | 2.896e-5 | 3.515e-5 |
| `y` RMSE | **5.918e-5** | 1.376e-4 | 9.013e-5 |
| `z` RMSE | 5.297e-5 | 6.980e-5 | **4.766e-5** |
| Dense-grid residual MSE | **1.470e-7** | 3.700e-7 | 2.626e-7 |
| Collocation residual MSE | **1.468e-7** | 3.712e-7 | 2.625e-7 |
| Maximum invariant-1 drift | **2.510e-4** | 2.974e-4 | 2.747e-4 |
| Maximum invariant-2 drift | 1.919e-4 | 3.459e-4 | **1.852e-4** |
| Final endpoint-error norm | **3.462e-5** | 3.098e-4 | 2.278e-4 |

The historical run wins on the primary whole-trajectory RMSE, maximum error,
dense residual, and endpoint accuracy. Relative to it, the 512-point run has
1.86x RMSE and 2.50x maximum error. The 1500-point run narrows these to 1.28x
RMSE and 1.44x maximum error. Relative to 512 points, 1500 points has 31.3%
lower RMSE, 42.4% lower maximum error, and 29.0% lower dense residual MSE.
Its `x` RMSE is 21.4% worse, so the gain is mainly in `y` and `z`.

The 1500-point run has the best median error and second invariant drift. This
does not overturn the aggregate ranking: its worse upper tail and endpoint
make its total RMSE higher than the historical run. The reference invariant
drifts are about `3e-10`, far below the PINN drifts. Each run's dense and
collocation residual MSE differ by under 0.4%; the gaps are small, but low
residual on its own does not guarantee a small accumulated state error.

## Where the errors occur

The table uses `point_summary.csv` for each run. These are *different uniform
collocation grids*, so the window comparisons are localization evidence rather
than a controlled paired-point test. Window 10 covers about `t=4.913–5.404`;
windows 24–26 cover about `t=11.791–13.264`.

| Window or region | Historical residual MSE / state RMSE | 512-point | 1500-point |
|---|---:|---:|---:|
| Window 10 | 2.43e-8 / 8.24e-5 | 2.15e-6 / 2.26e-4 | 2.06e-6 / 2.17e-4 |
| Window 24 | 5.05e-8 / 8.36e-5 | 1.96e-6 / 3.46e-4 | 4.02e-8 / 1.69e-4 |
| Window 25 | 4.75e-9 / 5.97e-5 | 2.06e-6 / 4.03e-4 | 1.56e-8 / 2.09e-4 |
| Window 26 | 1.50e-8 / 4.25e-5 | 1.87e-6 / 4.01e-4 | 3.79e-8 / 2.30e-4 |
| Windows 24–26 share of whole-domain collocation residual | 1.8% | **58.7%** | 1.3% |

The 512-point run's four worst residual windows, 10 and 24–26, contribute
about 80% of its final total residual. Its late local fits and its propagated
state both deteriorate. Raising density to 1500 points largely removes the
late *local residual* problem, yet its late state error remains 2–5 times the
historical level. That pattern is consistent with an inaccurate incoming
anchor or error accumulated earlier; it does not prove a particular joint is
the cause. In both new runs, window 10 remains a local difficulty. The saved
joint-continuity plots show value jumps near floating-point roundoff and slope
or one-sided residual mismatches around `1e-3`; exact value continuity alone
does not certify a good handoff state.

## Optimization and resource use

| Measure | Historical | 512-point | 1500-point |
|---|---:|---:|---:|
| Adam updates | 29,277 | 21,183 | 39,666 |
| L-BFGS closure evaluations | 49,400 | 49,428 | 16,410 |
| Epsilon-10 stages reaching 4,000-step cap | 6/27 | 2/27 | 4/27 |
| Recorded wall time | 929 s | 932 s | 2,105 s |
| Recorded peak CUDA allocation | 549 MB | 212 MB | unavailable (CPU) |

The 512-point run uses 65.3% fewer collocation points and records 61.3% lower
peak CUDA allocation than historical, but essentially the same wall time and
worse accuracy. It completes fewer Adam updates, while L-BFGS evaluation count
is nearly unchanged. Closure evaluations are not optimizer iterations. The
1500-point run uses 2.93x as many points as 512,
1.87x as many Adam updates, and about one third as many L-BFGS evaluations.
Its CPU wall time cannot be used to compare GPU throughput. Even the two CUDA
times do not establish hardware-normalized speed because GPU model and runtime
environment were not recorded. The candidate folders are much larger due to
full snapshot and resume artifacts; folder size is not a model-memory metric.

The causal gate is point-count dependent: `w_min = exp(-eps * sum(previous
per-point losses))`. At epsilon 10 and threshold 0.99, the preceding average
loss must be roughly below `1.96e-6` with 512 points, versus `6.7e-7` with
1500 points. The 512-point run can therefore pass the same nominal threshold
with about 2.9x larger mean preceding residual. Its two capped stages are
windows 10 and 25 (`min_w=0.9888, 0.9894`). The 1500-point caps are windows
10, 11, 23, 24 (`min_w=0.9695, 0.9685, 0.9175, 0.9360`). Historical caps
were windows 9–11 and 23–25, inferred from its marks. Historical threshold
status was checked before, not after, the update; these counts are descriptive,
not directly equivalent certifications.

## Code audit and interpretation limits

1. `src/pinn/pinn.py` builds 27 independent networks and uses the previous
   endpoint as the next hard initial state. The networks do not share learned
   weights after warm starting. This reduces the time span each network must
   fit, at the cost of 27x parameters and sequential handoff sensitivity.
   There is no architecture difference among these three runs and no evidence
   here to rank depth, width, causal weighting, warm start, or float64 against
   alternatives.
2. `src/pinn/train.py` trains each stage with detached causal weights, then
   plain residual L-BFGS. A per-window training-loss curve changes meaning at
   every epsilon and optimizer transition. The historical `final_loss` is only
   the final window's objective (`1.495e-8`), while the new `final_loss` is the
   whole-domain collocation residual. Compare the explicit
   `residual_mse_collocation` and `residual_mse_dense` fields instead.
3. The newer code includes the outgoing endpoint in the nonfinal window loss
   and checks the causal gate on the post-update parameters. These changes are
   sensible correctness improvements, but their effect on accuracy cannot be
   extracted from the three runs because other settings changed. Capping a
   stage means it proceeded without meeting the requested threshold.
4. The 1500-point candidate also cuts L-BFGS from 1500 to 500 iterations.
   Because L-BFGS remains important in every window and the run has a different
   number of Adam updates, its outcome cannot be attributed solely to higher
   collocation density. Its closeness to the historical point count does not
   make the two runs a controlled code comparison.
5. Final point summaries agree numerically with each saved full-domain
   collocation MSE. The old run lacks the new full breakdown and resume files,
   limiting exact historical per-joint reconstruction. Its summary records 16
   snapshots while its point summary records 17 observations per point; the
   final numerical comparisons above do not depend on that count. The candidate config
   manifests record settings, but not a Git commit or device model. All three
   have one seed, so no uncertainty interval or reliability ranking is known.

## Decision and next controlled tests

For the accuracy goal on this exact orbit, retain the historical checkpoint as
the best demonstrated result. For a lower recorded CUDA allocation, the
512-point candidate is the tradeoff, with a large accuracy penalty and no
observed wall-time gain. The 1500-point candidate is a better *accuracy*
candidate than 512 points, particularly in the upper error tail, but it does
not beat the historical run on the primary metrics. It does have the best
median error, `z` RMSE, and second invariant drift.

The most informative next run is a matched 1500-point experiment on the same
GPU and repaired code, with 1500 L-BFGS iterations and the same LR schedule as
the 512-point candidate, changing only the point count. A second matched run
should hold the point count fixed and vary only L-BFGS budget. Evaluate every
candidate on the existing common reference grid, report per-window endpoint
state error and both one-sided joint residuals, record GPU model and library
versions, and repeat promising settings with multiple seeds. If the causal
schedule itself is changed, scale or redefine its cumulative loss and retune
epsilon and threshold together rather than treating the old threshold as
dimension independent.
