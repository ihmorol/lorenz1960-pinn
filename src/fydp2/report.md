# Implementation report: a PINN for the Lorenz-1960 system

FYDP-2 phase. This note records what was built, how it was trained, and what it
achieved. Plain-language background is in `docs/the-short-version.md`; the full
code walkthrough is in `docs/CODE_EXPLAINED.md`.

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

## Results

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
The figure suite in `results/figures/` documents convergence, gradient health,
collocation quality, phase portraits, physics residuals, and drift in the two
conserved quadratic forms.

## Verification

Nine pytest checks cover the exact initial condition, the residual definition
against the reference trajectory, loss reduction in both IC modes, telemetry
recording, the CSV round trip, and generation of every figure and table. All
pass. `python run_pinn.py` reruns the full pipeline; `rebuild_figures()`
redraws every figure from the saved checkpoint and CSVs without retraining.

## Limitations

Training runs in float32, which plausibly sets the error floor near 1e-5 to
1e-6; no float64 comparison was run. Results are for a single initial value
problem on one time window and a single seed. A new initial condition requires
retraining, and on this problem the classical solver remains much cheaper. The
architecture and activation sweep (depth x width x activation x seed) is
scaffolded in `figures.generate_sweep` and is the subject of the next phase.
