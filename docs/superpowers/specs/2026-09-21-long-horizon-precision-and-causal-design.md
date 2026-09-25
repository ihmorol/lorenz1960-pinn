# Long-horizon Lorenz-1960 PINN: precision-first batch run and causal windowed run

Date: 2026-09-21. Status: historical design; the selected implementation and
measured results are documented in the repository README.

## 1. Goal

Reach the lowest achievable trajectory error for the Lorenz-1960 PINN over one full closed loop of the orbit, t in [0, 13.26446] (the state returns to u0 to 6e-9
at T = 13.26446; x has period 6.63, y and z have 13.26), with the existing 4x60 tanh network, under two training schemes that the literature
supports, and make every phase of training and evaluation visible in figures that
expose problems rather than decorate results.

Target: RMSE <= 1e-6 against the DOP853 reference on 13 265 evaluation points (1000 per time unit), seed 0.
Honest expectation from the evidence (Rathore et al. 2024; Xu et al. 2025; Wang and
Lai 2024): float64 with an L-BFGS finisher moves a single network from the current
1.6e-3 (measured on [0, 10]) into the 1e-5 to 1e-6 range; 1e-6 is not guaranteed. The number reached is
reported as is. The multi-stage residual network (Wang and Lai 2024) is the agreed
fallback if both runs land short, and is out of scope for this spec.

Constraints agreed with the user: GPU via Colab is available; more epochs are fine;
seeds stay at one; method changes are marginal (the network, activation and hard
initial condition stay; knobs are added with today's behaviour as default).

## 2. The two runs

Shared by both: 4x60 tanh, raw t input, float64, hard initial condition in
Lagaris form u = u0 + (t - t0) N(t), and Adam followed by L-BFGS. Run A uses
39,793 full-domain collocation points. The selected Run B uses 41,446 global
grid points, which gives 1,536 effective training points in every window after
adding each nonfinal outgoing endpoint.

Run A, batch (branch `feat/batch-precision-run`): one network on the full loop; 39 793
Latin-hypercube points; loss = mean squared residual; Adam 40 000 epochs at lr 1e-3
with exponential decay 0.9 every 5000 epochs; then L-BFGS up to 5000 iterations,
strong-Wolfe line search, in float64.

Run B, causal windows (finalized on `feat/batch-training-0-1`): the
Lorenz recipe of Wang, Sankaran and Perdikaris (CMAME 2024, Algorithm 1 and
Appendix E), adapted to our network. the loop is split into 27 windows of 0.491, the closest
to the paper's 0.5; each
window trains a fresh 4x60 whose initial state is the previous window's prediction at
the joint; each window receives 1,536 effective uniform points; per-point causal weights
w_i = exp(-eps * sum_{k<i} L(t_k)) with stop-gradient; eps runs through
[1e-2, 1e-1, 1, 10], advancing when min_i w_i > 0.99, capped at 4,000 Adam
iterations per eps; then up to 1,000 L-BFGS iterations per window (our one addition to the paper, for the
float64 floor). The chain is one `WindowedPINN` that predicts at any t.

Everything else (architecture width and depth, activation, seed, reference solver)
is identical between the runs so that the comparison isolates the training scheme.

## 3. Repository layout

Hard rules from the user: visualisation code lives in its own package; each important
step of the method is a named function in its own file under `functions/`; the main
files compose those functions and stay short; comments only where the logic is not
obvious; no existing figure is dropped.

```
src/pinn/
  config.py                 Config; every new field defaults to the run-of-record value
  problems.py               Problem dataclass + registry; lorenz1960 entry
  pinn.py                   PINN, WindowedPINN: compose functions
  train.py                  the loop: compose functions; ends with viz.generate_all
  history.py                TrainHistory, SnapshotWriter, residual_grid, param trail
  sweep.py                  run_one, sweep, config_for
  test_pinn.py
  functions/
    collocation.py          latin_hypercube_points, uniform_points
    trial.py                hard_initial_condition (span | unit), end_state_penalty
    derivative.py           time_derivative
    physics.py              residual
    losses.py               mean_squared_residual, causal_weights, causal_loss
    optimizers.py           adam_with_decay, run_lbfgs
    windows.py              split_windows, carry_end_state
    reference.py            solve_reference, reference_at
  viz/
    __init__.py             RunArtifacts, generate_all(run_dir)
    style.py                palette, set_style, save_figure
    training.py             loss with phase bands, temporal residual surface, causal
                            weight surface and min-w, per-window grid, gradient norms
    evaluation.py           solution vs reference, error and residual vs t, phase
                            portraits, invariant drift, error-growth fit, joint continuity
    trajectory3d.py         animated 3-D trajectory, error and residual surfaces
    landscape.py            PCA loss surface with optimiser path, weight path
    compare.py              multi-run comparison page
    sweep.py                architecture heatmap, scatter, seed robustness
run_pinn.py, run_single.py, run_sweep.py      unchanged entry points
run_compare.py, run_viz3d.py, run_landscape.py  thin wrappers over viz/
colab.ipynb                                   clone branch, run one script, zip runs/
```

`figures.py` (960 lines) is split into `viz/` by phase with every figure and output
filename preserved. `train.py` calls `viz.generate_all(run_dir)` once at the end.
`RunArtifacts` (arrays only, no model objects) is the single contract between core
and viz, so figures can be regenerated from a saved run.

## 4. Core changes

### 4.1 Config knobs (branch A unless noted)

| field | default (= today) | Run A | Run B |
|---|---|---|---|
| `dtype` | "float32" | "float64" | "float64" |
| `ic_scale` | "span" (g = (t-t0)/(tf-t0)) | "unit" (g = t-t0) | "unit" |
| `n_collocation` | 3000 | 39,793 | 41,446 (1,536 effective/window) |
| `collocation` | "lhs" | "lhs" | "uniform" |
| `eval_per_unit` | 1000 (n_eval = 1001 on [0, 1]) | 1000 | 1000 |
| `epochs` | 20000 | 40000 | (per eps cap 4,000) |
| `lr_decay`, `lr_decay_every` | none (linear 1e-3 to 1e-4 today) | 0.9, 5000 | 0.9, 1000 |
| `lbfgs_iters` | 0 | 5000 | 1,000 per window |
| `t_span` | (0, 1) | (0, 13.26446) | (0, 13.26446) |
| `problem` | "lorenz1960" | same | same |
| `n_windows` (B) | 1 | 1 | 27 |
| `causal_eps_schedule` (B) | () | () | (1e-2, 1e-1, 1, 10) |
| `causal_delta` (B) | 0.99 | - | 0.99 |
| `causal_max_iters` (B) | 0 | - | 4000 |

`lr_decay=None` keeps today's linear schedule so `Config()` reproduces the run of
record. The run tag gains suffixes for non-default knobs (`_f64`, `_unit`, `_win27`,
`_causal`) so runs never overwrite each other.

### 4.2 `Problem` (branch A)

```python
@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    rhs: Callable[[Tensor], Tensor]        # u (N, dim) -> du/dt (N, dim)
    initial_state: tuple[float, ...]
    t_span: tuple[float, float]
    invariants: tuple[Callable, ...] = ()
    end_state: tuple[float, ...] | None = None
    reference: Callable | None = None      # None -> generic solve_ivp DOP853
```

`lorenz1960` is registered with today's coefficients and the locked baseline solver so
its numbers do not change. `end_state` is declared for the future two-point BVP: when
set, `loss_terms` adds `gamma * |u(t_end) - end_state|^2`; nothing else about BVPs is
built now.

### 4.3 Model (branch A: `PINN`; branch B: `WindowedPINN`)

`PINN.forward(t) = hard_initial_condition(t, net(t), form=ic_scale)`. `residual` is
`time_derivative(u, t) - problem.rhs(u)`.

`WindowedPINN` holds one `PINN` per window, each with its own local `t0` and
`initial_state`; `forward` and `residual_parts` route each t to its window and scatter
results back in the caller's order. Predictions on the joints use the right-hand window.

### 4.4 Training loop (branch A: single window; branch B: adds the eps loop)

```
points = collocation(...)
model  = build(problem, cfg)                     # PINN or WindowedPINN
for window in split_windows(cfg):                # one window on branch A
    for eps in cfg.causal_eps_schedule or [None]:
        for it in range(cap):
            r = residual(model, points_w)
            loss = causal_loss(r, t, eps) if eps else mean_squared_residual(r)
            step(adam)
            log; snapshot every k
            if eps and min_w > delta: break
    run_lbfgs(model_w, points_w, cfg.lbfgs_iters)
    carry_end_state(model_w -> next window)
save; viz.generate_all(run_dir)
```

Logged per snapshot: epoch, loss, eps, min_w, w(t) profile (branch B), per-layer
gradient norms, weights (param trail), flattened gradient (grad trail), per-point
residual and error (breakdown CSV).

### 4.5 Precision

`dtype` sets the default torch dtype for the model, points and reference tensors.
L-BFGS uses `tolerance_grad=1e-12`, `tolerance_change=1e-16` in float64
(today's 1e-14 is below float32 resolution and is what makes it stop instantly).

## 5. Visualisation suite

Generated by `viz.generate_all(run_dir)` from saved artifacts. Every figure carries a
caption stating what it exposes. Existing figures keep their names. New:

Training: (1) loss vs iteration with Adam / eps stages / L-BFGS bands and eps-advance
markers; (2) temporal residual surface L(t, iteration), log z, 3-D HTML and PNG;
(3) causal weight surface w(t, iteration) and min-w vs iteration with the delta line
(B only); (4) per-window loss grid, one panel per window (B only); (5) PCA loss
landscape with optimiser path and 3-PC weight path (moved from run_landscape.py);
(6) per-layer gradient norms, raw and rolling median; (7) trajectory animation over
snapshots, opening at the final epoch (moved from run_viz3d.py).

Evaluation: (8) solution vs reference with signed error (existing); (9) |error| and
|residual| vs t, log scale, window joints marked; (10) 3-D trajectory vs reference
coloured by log error; (11) phase portraits and invariant drift from
`Problem.invariants`; (12) error growth vs t on log-log with fitted power law;
(13) float32-vs-float64 loss-floor panel when both histories exist; (14) joint
continuity: value and slope jumps at each window boundary (B only); (15) the
multi-run six-panel comparison (existing `run_compare`).

Training dynamics, the views PINN papers use for "how the gradient changes":
(16) gradient direction stability: cosine similarity between gradients at consecutive
snapshots and the gradient norm (needs the flattened gradient saved per snapshot, like
the weight trail); (17) per-layer gradient histograms at chosen snapshots; (18) NTK
eigenvalue spectrum at snapshots on 256 subsampled points (Wang, Yu and Perdikaris
2022), showing which frequencies the network can learn at that moment.

Film: (19) a Manim video per run (`viz/film/`, rendered from saved artifacts with the
`manim-video` skill, never during training) in three 3-D acts: the trajectory drawn per
snapshot epoch against the reference loop, coloured by per-point error; the PCA loss
surface with the Adam path and a gradient arrow per epoch; the residual front L(t, epoch)
and, for the causal run, the weight curve w(t) sweeping forward as eps advances.

## 6. Tests

One check per non-trivial piece, in `test_pinn.py` (existing tests move with the
functions they cover):

- `Config()` with no arguments reproduces the run-of-record summary row (arch,
  n_params, rmse to 1e-9 on the tiny test config).
- `hard_initial_condition`: both forms equal `u0` at `t0`; the unit form has
  `du/dt(t0) = N(t0)`.
- `causal_weights`: `w[0] == 1`; non-increasing in cumulative loss; all ones when
  the residual is zero.
- eps advances exactly when `min_w > delta` and not before (tiny synthetic loop).
- `WindowedPINN.forward` equals the sub-model at interior points; value at each joint
  equals the next window's `initial_state`.
- `Problem` registry and every config knob round-trip through `run_summary.csv`
  via `config_for`.
- float64 path trains end to end on a tiny config and produces float64 tensors.

## 7. Compute plan

Run A on a Colab T4: 40 000 points x 40 000 epochs float64 about 30-45 min plus
L-BFGS about 10 min. Run B uses 27 windows and up to four epsilon stages; most
stages stop early. The selected saved run took 881 seconds on CUDA. Local
training can checkpoint per window and resume. `colab.ipynb` clones `main`, installs
`requirements.txt`, runs one script, zips `runs/`.

## 8. Out of scope

Multi-stage residual networks, Fourier features, modified MLP, KANs, seed sweeps,
the BVP solver itself, and any change to the network shape. Each is a separate spec.

## 9. Outcome (selected final run, 2026-09-24)

| | Run A `4x60_f64_unit` | Run B `4x60_f64_unit_win27_causal_warm` |
|---|---|---|
| RMSE (x, y, z) | 0.38, 2.08, 0.74 | 2.74e-5, 4.96e-5, 4.05e-5 |
| RMSE combined | 2.24 | 6.97e-5 |
| max abs error | 3.35 | 3.09e-4 |
| final loss | 2.8e-4, flat over 6251 L-BFGS evals | 1.87e-7 full-domain residual MSE |
| iterations | 40 000 Adam + 5000 L-BFGS | 34,765 Adam + 33,143 L-BFGS evaluations |
| wall | 75 min | 881 s |

Run A never left the fixed-point plateau: the prediction decays to a constant
(x -> 0.01, y -> 1.7, z -> -0.05) while the reference oscillates. Run B follows
the whole closed orbit with error around 1e-5 to 3e-4; the error
does not grow window to window. Hand-off value jumps are ~1e-16 (hard IC), the
initial-slope mismatch at each joint is ~1e-3 (learned, not enforced), which is
the residual bump seen at every window start.

The selected budget is eps `(1e-2, 1e-1, 1, 10)`, a 4,000-iteration cap per
stage, 1,000 L-BFGS iterations per window, and warm-started weights. Windows 10
and 11 reached the epsilon-10 cap with minimum weights `0.9718467` and
`0.9031337`. L-BFGS completed afterward, so the run is complete but those two
causal stages did not satisfy the `0.99` gate.

Two figure bugs found from these runs and fixed on branch B: windowed runs now
end with a snapshot of the finished model (the comparison page had drawn a
mid-training window), and the joint figure evaluates both windows at the joint
instead of at t +/- h.
