# Error Propagation & Strategy: Batch Monolithic PINN vs. Causal Windowed PINN

**Date**: September 22, 2026
**Branches analyzed**: `feat/batch-precision-run` (Run A) and `feat/causal-window-training` (Run B)
**Problem**: Lorenz-1960 maximum-simplification oscillator, `dx/dt = -0.10 y z`, `dy/dt = 1.60 x z`, `dz/dt = -0.75 x y`, from `u(0) = (0.5, 0.75, 1.0)` over one closed orbit `t in [0, 13.26446]`.
**Companion doc**: `comprehensive_causal_vs_batch_analysis.md` (benchmarks, loss landscapes, NTK, SOTA roadmap). This document focuses on **how error propagates** in each scheme, the **strategy** each branch encodes, and a **verified bug audit** of both.

---

## 1. What each branch actually runs

### Run A — `feat/batch-precision-run` (batch monolithic)

`run_batch.py`: one 4x60 tanh MLP (11,283 params) over the whole domain.

- **Grid**: 39,793 LHS points over `[0, 13.26446]`, drawn once, full-batch.
- **Trial**: `u_T(t) = u0 + (t - t0) * N(t)` (`ic_scale="unit"`, Lagaris form) — IC exact by construction.
- **Loss**: plain mean squared ODE residual over all `3 x 39,793` residual entries. No time structure at all.
- **Optimizer**: Adam 40,000 epochs, `StepLR(5000, gamma=0.9)` (fires 8 times: lr 1e-3 -> ~4.3e-4), then 5,000 L-BFGS iterations. Checkpoints every 5,000 epochs (resumable).
- **Code path**: `train()` single-domain loop in `src/pinn/train.py` (lines 78-143). The branch diff vs. the causal branch is exactly the absence of `train_windows` / `WindowedPINN` / `causal_loss`; the single-domain loop is identical.

### Run B — `feat/causal-window-training` (causal windows)

`run_causal.py`: 27 windows of `dt ~ 0.491`, one 4x60 net per window (27 x 11,283 = 304,641 params).

- **Grid**: 39,793 uniform points (~1,474/window). Uniform, not LHS, so causal ordering is exact.
- **Chaining**: window `k`'s hard IC anchor is window `k-1`'s **predicted** endpoint, detached (`set_window_start(k+1, sub(end)[0])`, `train.py:222-225`). The reference solution never enters training.
- **Warm start**: window `k` inherits window `k-1`'s trained weights before its own Adam (`train.py:173-176`).
- **Causal weighting** (Wang, Sankaran & Perdikaris 2024, eq. 3.5): per-point loss `L_i` sorted by `t`, weight `w_i = exp(-eps * sum_{k<i} L_k)`, detached. eps ramps `0.01 -> 0.1 -> 1 -> 10`; a stage advances when `min_i w_i > 0.99` or after 4,000 iterations (`causal_max_iters`).
- **Optimizer**: fresh Adam per window (`train.py:177`), then 1,500 L-BFGS iterations per window, then freeze and hand off.
- **Code path**: `train_windows()` (`train.py:146-229`), `WindowedPINN` (`pinn.py:76-105`), `causal_loss` (`functions/losses.py:15-19`).

### Results (verified against `runs/*/run_summary.csv` and `metrics.csv`)

| Metric | Run A (batch) | Run B (causal windows) |
| :--- | :--- | :--- |
| Final loss | 2.80e-4 (plateau) | 1.50e-8 |
| Combined-L2 RMSE | **2.237** | **8.43e-5** |
| Combined-L2 max error | 3.346 | 2.35e-4 |
| R2 (x, y, z) | -30.89, -1.88, 0.027 | 0.9999998, 0.999999998, 0.999999995 |
| Invariant 1 / 2 max drift | 0.100 / 0.114 (10-11%) | 2.51e-4 / 1.92e-4 (0.02%) |
| Final x (ref 0.5000001) | 0.00075 | 0.49998 |
| Wall clock | 4,528.6 s | 928.6 s |
| Adam iterations | 40,000 | 29,277 causal-Adam (total over windows) + L-BFGS |

---

## 2. How error propagates — the technical mechanics

### 2.1 Run A: no propagation channel, so the failure is global mode collapse

A batch PINN has **no mechanism that couples early and late times** except the shared weights. Every collocation point pulls the network toward satisfying the residual *locally and independently*. That sounds safe, and on `[0, 1]` it works (run of record: RMSE 1.7e-5). On a 13.26-time-unit orbit it fails catastrophically, for a structural reason:

**The ODE has a line of equilibria.** For Lorenz-1960, `x = z = 0` (any `y`) gives `f = (0, 0, 0)` identically. A trajectory that sits on that line has *exactly zero physics residual*. The hard IC pins `u(0) = u0` exactly, but the trial form lets the solution relax away from `u0` over a thin transition layer; the loss only pays for the layer, not for being on the wrong trajectory. Averaged over 39,793 points, the wrong-but-trivial solution beats the right-but-hard one. The network is not "wrong somewhere and propagating the error forward" — it has found a **degenerate global minimizer** of the residual functional.

**The data proves residual and solution error are anti-correlated.** Per-window means from `runs/4x60_f64_unit/point_summary.csv` (27 equal bins of `t`, bin width 0.491, for direct comparability with Run B):

| t-bin | mean solution error | mean residual |
| :--- | :--- | :--- |
| 0 (t~0.25) | 2.3e-2 | 8.9e-2 (worst) |
| 4 (t~2.2) | 3.2e-1 | 3.8e-2 |
| 9 (t~4.7) | 1.30 | 1.2e-2 |
| 14 (t~7.1) | 2.94 | **1.4e-3 (best)** |
| 18 (t~9.1) | **3.34 (worst)** | 2.7e-3 |
| 22 (t~11.1) | 2.97 | 2.5e-3 |
| 26 (t~13.0) | 1.67 | 8.2e-3 |

Where the solution error is largest (bins 13-21, error ~3), the residual is *smallest* (~2e-3): the network satisfies `du/dt = f(u)` almost perfectly there — because both sides are near zero. `du/dt ~ f(u) ~ 0` is exactly the equilibrium-line signature. The loss cannot distinguish "correct orbit" from "trivially satisfied dynamics on a wrong manifold", and a mean over all points cannot see that the cheap 97% is buying the wrong answer.

**Why Adam cannot escape.** The basin is flat: thousands of epochs with gradient cosine similarity ~+1 and decaying gradient norm (sliding along the valley). Adam's second moment `v_t` decays to ~1e-8; any small curvature change then divides the step by `sqrt(v_t) ~ 1e-4`, amplifying it ~1e3 — the three parameter blow-ups at iterations ~9k, ~23k, ~36k, each followed by re-convergence to the same 2.8e-4 plateau. L-BFGS after 40k Adam epochs cannot leave the basin either: it converges to the nearest local minimizer, which is the same one.

**What the failure is NOT.** The NTK eigenspectrum is essentially identical to Run B's (15 orders of magnitude, fixed through training) — the network *can* represent the orbit. And the horizon is one closed loop of a *conservative* oscillator, so this is not Lyapunov/chaos blow-up either. It is purely the optimization geometry of a monolithic space-time loss on a long horizon.

### 2.2 Run B: error propagates forward through exactly one channel — the endpoint

The windowed scheme deletes the degenerate minimizer by construction: on a `dt ~ 0.49` slab, the transition-layer trick costs more than it saves, and the causal weights force the left end of the slab to converge before the right end gets any loss weight. Error then propagates through the only remaining channel:

```
window k's endpoint error  --(detached hand-off)-->  window k+1's initial-condition error
```

- **Forward only.** Once window `k` is frozen, nothing later can contaminate it. The `residual_evolution` figure shows the causal staircase: convergence sweeps left to right and never returns.
- **One number crosses each joint.** Only the *endpoint* state error propagates (same structural property Mattey & Ghosh prove for bc-PINN: interior errors of a window die with it). Derivative error does **not** cross, because the trial form only enforces `C^0` value continuity (see Bug B2).
- **Accumulation is sub-linear, not exponential.** This system is conservative; a small IC error produces a nearby orbit, not a Lyapunov divergence. Per-window means from `runs/4x60_f64_unit_win27_causal_warm/point_summary.csv`:

| window | mean err | max err | mean residual | note |
| :--- | :--- | :--- | :--- | :--- |
| 0 | **1.07e-4** | 1.6e-4 | 1.2e-3 | cold start (no warm start) — worst-trained window |
| 1 | 2.0e-5 | 3.0e-5 | 2.5e-4 | warm start pays off immediately |
| 2 | 1.2e-5 | 1.7e-5 | 9.2e-5 | best window |
| 5-8 | 2.9e-5 -> 5.4e-5 | | ~2e-4 | steady accumulation |
| 9-11 | 8.3e-5 -> 9.1e-5 | | 2.5-4.7e-4 | high-velocity stretch, Adam capped (B1) |
| 12 | 1.15e-4 | 2.1e-4 | **1.2e-3** | residual spike (capped window) |
| 13-17 | 7.6e-5 -> 5.6e-5 | | ~2-4e-4 | partial recovery (warm start from a good joint) |
| 18-23 | 6.8e-5 -> **1.6e-4** | 2.4e-4 | up to **1.3e-3** | second high-velocity stretch, capped again |
| 24-26 | 8.3e-5 -> **4.2e-5** | | ~2e-4 | error *shrinks* as the orbit closes back to u0 |

Three things to read off this table:

1. **Growth is consistent with roughly additive per-joint injection**: endpoint errors of a few 1e-5 per window accumulate to ~1.6e-4 by window 23 — faster than a pure random walk (`sqrt(20) x 1.2e-5 ~ 5e-5`), so the per-window errors are partially systematic (same-sign), not independent. That is expected: the same physics defeats each window in the same places.
2. **The two error surges (windows 9-12, 18-23) co-locate with residual spikes** — these are the windows where Bug B1 (unnormalized causal cumsum) stalled Adam at the 4,000-iteration cap with `min w` stuck at 0.75-0.90, leaving L-BFGS to finish the job without causal protection. The error surge is a *bug consequence*, not a scheme consequence.
3. **Error decays after window 23 and again at the head (window 0 vs 1-2).** The orbit is closed: as `t -> T`, the true trajectory returns to `u0`, and phase-type errors partially cancel — final combined RMSE 8.4e-5 despite the mid-orbit peak of 1.6e-4. Window 0 is the worst single window because it is the only one trained cold (Xavier init); every later window inherits a trained neighbour.

**The failure mode this scheme has not eliminated**: if a window's endpoint is wrong, every later window solves the wrong IVP. Over many orbits (`T >> 13`) this must eventually destroy phase accuracy even without chaos — which is why the SOTA roadmap recommends overlapping Schwarz/XPINN coupling for multi-orbit horizons.

---

## 3. Verified bug audit

Six defects were previously reported in `comprehensive_causal_vs_batch_analysis.md` (section 2). Each was re-verified against the code on 2026-09-22; seven **new** findings were added (B7-B13). None of the bugs change the headline result (Run B's 8.4e-5 stands; Run A's collapse is algorithmic, not a code bug — Run A's single-domain loop is itself correct).

### Critical

**B1 — Causal-weight cumsum is not normalized** (`functions/losses.py:9-19`). *Verified.*
`earlier = cumsum(L) - L` sums ~1,474 per-point losses raw. Wang et al. partition time into `M` chunks and sum `M` chunk-losses; here every point is its own chunk, so the exponent scales with `N_c`. Advancing the eps=10 stage requires `sum L_k < 1.005e-3`, i.e. mean per-point loss `< 6.8e-7` — at or below Adam's noise floor at lr 1e-3. Consequence observed in `history/causal.csv` + `point_summary.csv`: Adam stalls at the 4,000-iteration cap in the high-velocity windows (residual spikes 1.2e-3 at windows 12, 22, 23 vs. ~2e-4 typical; `min w` stuck at 0.75-0.90).
*Fix*: `earlier = (cumsum(L) - L) / N_c` (or multiply by `dt`).

**B2 — No C1 continuity at window joints** (`functions/trial.py`, `pinn.py:59-62`). *Verified.*
With `g = t - t0`, `du/dt|_{t0} = N(t0)` — arbitrary network output, not `f(u0)`. So `r(t_left) = N(t_left) - f(u_anchor)` is structurally nonzero at every joint; each window must spend capacity learning its left-edge derivative, and warm start does not provide it (the inherited `N(t_k)` encodes the *previous* window's average slope, not `f(anchor_k)`).
*Fix*: `u_T = u0 + (t - t0) f(u0) + (t - t0)^2 N(t)` — then `du/dt|_{t0} = f(u0)` identically and `r(t_left) = 0` by construction.

### Major

**B3 — L-BFGS polishing abandons the causal weighting** (`train.py:211-216`). *Verified.*
The closure calls `pinn_loss` (unweighted mean residual). In windows that ended Adam with `min w < delta` (the capped ones), the switch to L-BFGS silently drops causality exactly where it was not achieved.
*Fix*: only run L-BFGS once `min w > delta`, or keep the final-eps weighted loss inside the closure.

**B4 — LR scheduler is recreated per window; decay almost never fires in Run B** (`train.py:177`, `functions/optimizers.py:13`). *Verified, with a refinement to the previous audit.*
`StepLR(5000, 0.9)` resets every window. In the ~21 fast windows (a few hundred Adam steps each) it never fires; it fires only in capped windows exceeding 5,000 steps (e.g. window 9). So "decay was never applied" holds for the majority of Run B, not literally all of it. Run A is unaffected (40,000 epochs; fires 8 times as intended).
*Fix*: either pass a window-aware scheduler, or use the linear `lr_start -> lr_end` schedule over `causal_max_iters`.

### Minor

**B5 — `u0`/`coeffs` buffers hardcoded to float32** (`pinn.py:52-53`). *Verified.* Only bites standalone `PINN(Config(dtype="float64"))` instantiation; `train()` and `load_run()` both call `.to(dtype=cfg.torch_dtype)`, which converts buffers. Keep as robustness fix: register with `dtype=cfg.torch_dtype`.

**B6 — `time_derivative` runs 3 separate autograd traversals** (`functions/derivative.py`). *Verified.* Performance only (3x graph memory/time); a single `torch.func.jvp`/vmap pass would do.

### New findings (this audit)

**B7 — Mid-training telemetry is polluted by un-handed-off anchors** (`pinn.py:83`, `train.py:158, 201`).
Every window's `u0` is initialized to the *global* IC `(0.5, 0.75, 1.0)` and only corrected at its hand-off. During window `k`'s training, `record_reference` and the breakdown snapshots evaluate the **full** `WindowedPINN` over the **full** grid — so for `t > edges[k+1]` they measure untrained nets anchored at the wrong state. `history/reference_error.csv` and mid-run breakdown CSVs are therefore meaningless outside the trained region. Final results are unaffected (all hand-offs complete before evaluation).
*Fix*: mask telemetry to `t <= edges[k+1]`, or initialize later anchors lazily.

**B8 — Validation gap: causal schedule with `causal_max_iters = 0` runs zero Adam steps** (`config.py:37-39`, `train.py:152`).
`cap = cfg.causal_max_iters if cfg.causal_eps_schedule else cfg.epochs`. The default `causal_max_iters` is 0, so setting only `causal_eps_schedule` makes every eps stage run 0 iterations and sends an untrained (or merely warm-started) net straight to L-BFGS — silently.
*Fix*: `__post_init__` should require `causal_max_iters > 0` when the schedule is non-empty.

**B9 — `marks.csv` and `causal.csv` use different iteration units** (`train.py:208-210, 221`).
Window/eps marks record `len(history.loss)`, which counts L-BFGS evaluations; `causal.csv` has one row per causal-Adam iteration only. Slicing the `min_w` trace by the marks is off by the cumulative L-BFGS eval count (this exact misalignment bit the first pass of this audit).
*Fix*: record marks in causal-Adam units, or store a parallel `loss`-index column.

**B10 — `run_summary.csv` is misleading for windowed runs** (`train.py:296-306, 350`).
Run B's row reports `epochs = 20000` (an unused config default — the actual budget is `causal_max_iters` per stage) and `ms_per_epoch = wall / 20000`. The real counts: 29,277 causal-Adam iterations + per-window L-BFGS evals, 928.6 s.
*Fix*: report `len(history.loss)` (and Adam-only count) for windowed runs.

**B11 — `record_step` logs the causal-weighted loss as `residual`** (`train.py:197-199`).
After `res, w = causal_loss(...)`, the weighted scalar is passed as `residual=res`; in the single-domain path the same column is the raw mean squared residual. Cross-scheme comparisons of `training_diagnostics.csv` read a weighted quantity as if raw.
*Fix*: log both (`residual_raw`, `residual_weighted`).

**B12 — Stale docstring in `run_causal.py`.** Says it writes `runs/4x60_f64_unit_win27_causal/`; the arch tag appends `_warm`, so the actual directory is `runs/4x60_f64_unit_win27_causal_warm/`.

**B13 — `torch.empty` in `WindowedPINN.forward` / `_windowed_parts`** (`pinn.py:100, 114`).
If any window owned zero collocation points, its slice would stay uninitialized memory. Not triggerable with the current uniform grid (>=1,473 points/window), but a guard (`out.zero_()` or an assertion on `window_of`) is one line.

---

## 4. One-paragraph summary

Run A fails because a monolithic residual loss over a long horizon has a degenerate minimizer — the equilibrium line `x = z = 0` — that satisfies the ODE trivially over most of the domain while being O(1) wrong as a trajectory; with no mechanism coupling early and late times, Adam slides into that basin and its own `v_t` decay makes escape violent and futile. Run B replaces the global loss with 27 causally-ordered slabs, so error can only flow forward, one endpoint at a time, accumulating sub-linearly to 8.4e-5 over one closed orbit; its remaining error surges are not intrinsic to the scheme but trace to one bug (B1's unnormalized causal cumsum) stalling Adam in the two high-velocity stretches, plus a structural joint-derivative spike (B2) that a second-order trial form would eliminate.
