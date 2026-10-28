# Simulation Upgrade Plan — How Reference Papers Tell the PINN Story, and How We Close the Gap

Written 2026-10-06. Companion to `DEFENSE_QA.md`. Read this before touching the demo or the poster.

---

## 1. The honest diagnosis

We feel like we are "failing extremely" at the simulation part. The repo survey says something
different: **we already have almost every ingredient a good PINN paper has.** What we lack is
assembly — the ingredients exist in `runs/` but are not organized into the story reviewers and
examiners expect, and the interactive HTML is a *replay* of saved numbers instead of a *live
physics simulation*.

What already exists (verified in the repo):

| Ingredient | Where it lives |
|---|---|
| Per-iteration loss components, global + per-layer grad norms, update norms, lr | `history/training_diagnostics.csv` (`src/pinn/history.py:58-120`) |
| Per-epoch per-point snapshots (t, residual, error, ref, derivatives) | `breakdown/epoch_*.csv` |
| Full parameter + gradient trail at every snapshot | `history/param_trail.npz` |
| Causal weight profiles per epoch | `history/causal.csv`, `param_trail.npz` |
| Gradient histograms, layer grad norms, NTK spectrum, loss landscape, optimizer path in PCA space | `figures/gradient_*.png`, `ntk_spectrum.png`, `loss_landscape.*`, `weight_path_pca3.html` — but **only in the single-window run**, not the candidate-1536pt run |
| Error growth curve, error vs t, phase portraits, solution vs reference | `figures/error_growth.png`, `phase_portraits.*`, `solution_vs_reference.*` |
| Epoch-animated 3D trajectory replay (Plotly) + Manim film | `figures/trajectory.html`, `film/final.mp4` |
| Reference solution DOP853 rtol=1e-10 | `src/pinn/functions/reference.py:6` |

The real gaps, in order of how badly they hurt us at the defense:

1. **The 3D HTML is a replay, not a simulation.** No RK4 in the browser, no sliders, no
   butterfly-effect twin experiment, no live PINN inference. An examiner who plays with it will
   correctly conclude it is "just workable."
2. **The optimization story is invisible for our headline run.** `layer_grad_norms`,
   `ntk_spectrum`, `loss_landscape` are only generated for the single-window run. The
   candidate-1536pt causal run — the one we defend — has no optimization diagnostics at all.
3. **The reference part is asserted, not demonstrated.** We never show *why* DOP853 at
   rtol=1e-10 is trustworthy (no self-convergence check, no quoted reference uncertainty).
4. **"Different phases" is not simulated at all.** We have exactly one regime (rho=28) and one
   initial condition. No stable regime, no regime transition, no rho sweep.
5. **Our accuracy is quoted in MSE, not in the field's units.** The standard is relative L2 error
   and prediction horizon measured in Lyapunov times.

---

## 2. How the reference papers do it

### 2.1 Wang, Teng & Perdikaris — "Understanding and Mitigating Gradient Flow Pathologies in PINNs" (SIAM J. Sci. Comput. 43(5), 2021; [arXiv:2001.04536](https://arxiv.org/abs/2001.04536))

This is the canonical paper for the *gradient descent / weights-and-biases* story. Its figures:

- **Gradient-norm ratio curves**: log-scale plots of `||∇L_data|| / ||∇L_res||` (per loss term,
  per layer) over training iterations. This is how they *show* the pathology — one loss term's
  gradients are orders of magnitude larger, so the residual is never learned.
- **Per-layer gradient distributions** (histograms) at selected iterations — before vs after
  their fix.
- **Learning-rate annealing ablation**: same training with and without per-term lr annealing,
  shown as loss curves side by side.

**What this teaches us:** we must not just *say* "we use gradient diagnostics" — we must show one
figure where a raw/imbalanced state and our causal-warm state are compared on the same axes:
per-term and per-layer gradient norms over iterations, log scale. Our
`training_diagnostics.csv` already contains every column needed (`residual_loss`, `ic_loss`,
`layer_1..5_grad_norm`, `grad_norm`, `lr`). Zero retraining needed — just plot.

### 2.2 Wang, Sankaran & Perdikaris — "Respecting Causality for Training PINNs" (CMAME, 2024; [arXiv:2303.07404](https://arxiv.org/abs/2303.07404))

This is *our method's* source paper, and its Lorenz/chaotic figures are the template we should copy:

- **Causal weight evolution**: `w_i(t)` profiles drawn at several snapshots of training — you
  literally watch the causal front sweep forward in time. We have `param_trail.npz` +
  `causal.csv` with exactly this data; `causal_weights_redesigned.pdf` is close but static —
  add a small-multiples version at 5–6 training epochs.
- **Trajectory vs reference for the chaotic case**, with prediction horizons quoted.
- **With/without causal weighting** error-vs-time comparison on one axes.

**What this teaches us:** the single most convincing figure for our defense is
**error vs t, our causal-warm candidate vs a no-causal ablation, on the same axes**. If we have
(or can cheaply rerun) a non-causal baseline at the same budget, that figure is the paper's
Figure-"money shot" reproduced on our own system.

### 2.3 Grossmann et al. — "PINNacle: A Comprehensive Benchmark" (NeurIPS 2024 Datasets & Benchmarks; [arXiv:2306.08827](https://arxiv.org/abs/2306.08827), [code](https://github.com/i207M/PINNacle))

The field's evaluation conventions come from here:

- **Relative L2 error** is the headline metric — `||u_pred − u_ref||₂ / ||u_ref||₂` — not raw MSE.
- **Error bars over random seeds** whenever a method is compared.
- A distinction between fit error (on collocation points) and **generalization error** (off-grid),
  which we already independently verified (1.00× gap, `scripts/analyze_generalization_gap.py`).

**What this teaches us:** quote relative L2 everywhere (thesis, poster, QA doc), and label the
generalization-gap number explicitly — it is a genuine strength and we currently bury it.

### 2.4 The chaos-forecasting convention (Lyapunov time)

Across hybrid/RNN/PINN forecasting papers on Lorenz ([e.g. arXiv:2507.15959](https://arxiv.org/abs/2507.15959)-style hybrid forecasting work; the convention is uniform), the honest metric for a chaotic system is **prediction horizon in units of the Lyapunov time**. For standard Lorenz (sigma=10, rho=28, beta=8/3), λ₁ ≈ 0.906, so the Lyapunov time ≈ 1.1 time units. If our candidate is accurate to some tolerance out to t = T, report `T / 1.1` Lyapunov times. This reframes trajectory divergence from "our model fails" to "chaos does what chaos does — and here is our horizon," which is exactly the defensible framing.

---

## 3. The five upgrade tracks

### Track A — Make the optimization story visible (highest value / lowest cost, no retraining)

All from existing CSVs; add one script `scripts/plot_optimization_story.py` producing a
4-panel figure + a companion for the candidate run:

1. **Panel 1 — Loss decomposition** (log-y): total, residual, IC vs iteration, with Adam→L-BFGS
   phase bands and window boundaries marked (reuse `loss_phases.png` logic, tighter design).
2. **Panel 2 — Per-layer gradient norms** (log-y): `layer_1..5_grad_norm` vs epoch — shows the
   pathology shrinking; mirrors Wang 2021's diagnostic figure.
3. **Panel 3 — Gradient-norm ratio**: `ic_grad / residual_grad` style ratio over training.
4. **Panel 4 — lr + update norm** on twin axes: shows the Adam→L-BFGS handoff.

Weight & bias evolution: from `param_trail.npz`, plot **weight-distribution histograms**
(all-layer weights) at 4–5 snapshot epochs — the network's weights visibly reorganize as the
causal front advances. Pair with the existing `weight_path_pca3.html` for the interactive version.

**Blocker to clear:** port `layer_grad_norms` / `ntk_spectrum` / `loss_landscape` generation into
the causal-windowed trainer path (`src/pinn/viz/__init__.py:38` gates them to single-window runs).
For figures we can compute post-hoc from `training_diagnostics.csv` + `param_trail.npz`, no port
is needed — prefer that (minimal diff).

### Track B — Make the reference part demonstrable (cheap, high credibility)

1. **Self-convergence check**: integrate the reference at rtol=1e-8 and rtol=1e-10, report
   max deviation over the horizon; quote that number as the reference uncertainty in the thesis
   and QA doc. (Already on the peer-V&V cheap-upgrade list.)
2. **Reference uncertainty statement** as a one-line caption on `solution_vs_reference`:
   "Reference: DOP853, rtol=1e-10; self-convergence vs rtol=1e-8 ≤ X."
3. **FFT period check** on the reference (peer-V&V list) — confirms the reference reproduces
   Lorenz's known oscillation content, i.e., the reference itself is validated.

### Track C — Simulate different phases / regimes (the actual "physics simulation" work)

This is what "simulate different phases" should mean concretely. Four runs, same architecture,
same pipeline, one variable changed:

| Regime | rho | Behavior | What it demonstrates |
|---|---|---|---|
| Stable spiral | 14 | Trajectories decay to fixed points | PINN converges trivially; accuracy ceiling |
| Near-transition | 22 | Transients long, pre-chaotic | Training gets harder as stiffness grows |
| **Chaotic (headline)** | **28** | Strange attractor | Our existing candidate run |
| Periodic window | 99.65 (or rho≈166) | Periodic orbit | PINN can track indefinitely — no chaos wall |

Deliverable: one composite figure (2×4 grid — 4 regimes × [predicted vs reference trajectory |
error vs t]) plus a one-line summary table (relative L2 + horizon per regime). This is the
figure that turns "we trained on Lorenz" into "we *characterized* PINN training across the
Lorenz bifurcation structure." It also directly feeds the ODE-agnostic thesis pivot
(problem registry with regimes, not one hardcoded system).

Cost: 3 extra training runs (rho=14 is minutes; 22 and 99.65 similar budget to headline).
Run on Colab like the candidate.

### Track D — A real physics simulation in the browser (fixes the "just workable" 3D HTML)

Three levels, in increasing effort. Even level 1 alone transforms the demo.

**Level 1 — Live RK4 integrator with controls.** Lorenz RHS is 3 lines of JS; a fixed-step
RK4 at dt=0.005 is stable and fast. Build `poster/final/sim.html` (standalone, no build step):
- sliders: sigma, rho, beta, dt, playback speed;
- draggable initial condition (or IC sliders);
- **twin-trajectory butterfly experiment**: second trajectory offset by δ=1e-5 in x₀, toggle on;
  show ||Δ(t)|| on log-y strip — live demonstration of sensitive dependence;
- color: speed |v| or z-height.

**Level 2 — Live PINN inference in the browser.** The candidate network is 4×60 tanh — a few
thousand parameters. Export `pinn.pt` weights to JSON (note: `double()` the model before
`load_state_dict` — float64; convert to float32 for JS or keep JS in float64 via typed arrays),
write a forward pass in JS, and add a "PINN" toggle: browser runs trial solution `x(t) = IC + t·NN(t)`
at the current t. Now the demo is literally *our defended solution running live next to the
ground truth*. This is the strongest possible answer to "is it a real simulation?"

**Level 3 — The money view.** Reference (RK4), PINN (live inference), and twin (perturbed RK4)
on the same axes, with error vs t readout and a "divergence time" marker where PINN-vs-ref
error crosses a tolerance. One page, everything the defense needs.

Constraint: the HTML must run offline from the poster laptop (no CDN — inline three.js or use
plain canvas/WebGL-free 3D projection like the existing Plotly replays).

### Track E — Training phases as an exhibit

We already record everything; this track is only assembly:

1. **Training atlas figure**: pick 6 epochs spanning the causal front's sweep (from `marks.csv`
   + `causal.csv`), and for each draw predicted-vs-reference + residual strip from
   `breakdown/epoch_*.csv`. One figure = the entire training story.
2. **Causal front small-multiples**: `w_i(t)` profiles at the same 6 epochs (Wang 2024 style).
3. Keep `trajectory.html` (epoch slider) and `film/final.mp4` as the *live* exhibits at the
   defense — they are already good; the poster links to them via QR.

---

## 4. Priority order (defense timeline)

| # | Item | Track | Cost | Why first |
|---|---|---|---|---|
| 1 | Optimization-story figure from existing CSVs | A | ~half a day, no retrain | Biggest credibility gain per hour; directly answers "we don't know what's happening in gradient descent" |
| 2 | Reference self-convergence + uncertainty statement | B | ~1 hour | Answer ready for the "how do you trust your reference?" attack |
| 3 | Relative L2 + Lyapunov-time horizon in all docs | metric | ~1 hour | Same numbers, field-standard units |
| 4 | Level-1 browser sim (RK4 + sliders + twin) | D | 1–2 days | Turns the demo from replay into physics |
| 5 | Causal vs non-causal error-vs-t ablation | A/C | 1 Colab rerun | Reproduces Wang 2024's money figure on our system |
| 6 | Training atlas + causal front small-multiples | E | ~half a day, no retrain | Poster/QA exhibit |
| 7 | Regime sweep (rho = 14/22/28/99.65) | C | 3 Colab runs | The "different phases" deliverable; feeds ODE-agnostic pivot |
| 8 | Level-2 live PINN inference in browser | D | 1–2 days | The showstopper, if time remains |

Items 1–3 need no training and should land before the next poster revision. Item 4 is the
minimum bar for calling the HTML "a physics simulation."

---

## 5. One-paragraph answer for the defense (if asked "what does your simulation show?")

"We validate against a DOP853 reference whose self-convergence bounds its uncertainty below
[X]. We report accuracy in relative L2 and prediction horizon in Lyapunov times — for a chaotic
system, trajectory divergence beyond [N] Lyapunov times is physics, not model failure. Our
optimization diagnostics — per-layer gradient norms, gradient-norm ratios, and causal-weight
evolution — show the training pathology the causal weighting fixes, and an ablation without
causal weighting stalls at [Y] while our candidate reaches [Z]. Finally, we trained the same
architecture across the Lorenz bifurcation structure — stable, near-transition, chaotic, and
periodic regimes — showing which regimes PINNs handle and where the chaos wall sits."
