# Two training modes, explained: the single-domain PINN and the sequential rollout

This document explains the two training schemes in this repository in plain
language, then backs every statement with code, tests, and measured run data.
It sits between `the-short-version.md` (the plainest account) and
`CODE_EXPLAINED.md` (the line-by-line file tour).

- **Mode 1 — the single-domain PINN** (`Config()` defaults): the network maps
  time → state. 3000 collocation points over `t ∈ [0, 1]` are checked in
  parallel against the physics.
- **Mode 2 — the sequential rollout** (`Config(sequential=True)`, or
  `python run_single.py 4 60 --sequential`): the same network maps time →
  *velocity*, and the state is rebuilt by walking forward from the exact
  initial condition.

Evidence keys used below: **[code]** = source file and line numbers,
**[test]** = an automated check in `src/pinn/test_pinn.py`,
**[data]** = numbers read from a completed run under `runs/`.
Line numbers refer to the current state of `feat/sequential-rollout`.

---

## 1. The problem, in plain words

Lorenz's 1960 model has three numbers, x, y, z, whose rates of change depend on
each other:

```
dx/dt = −0.10 · y·z        dy/dt = 1.60 · x·z        dz/dt = −0.75 · x·y
```

**[code]** the coefficients are computed from k = 2, l = 1 in
`src/baseline/lorenz1960_baseline.py:28-32` and imported, never re-typed
(`src/pinn/config.py:79-81`). Verified numerically: `(−0.1, 1.6, −0.75)`.

We start at `(x, y, z) = (0.5, 0.75, 1.0)` at t = 0 and want the values up to
t = 1 (`config.py:29-30`). There is no closed-form answer; the trusted
reference is SciPy's DOP853 at rtol 1e-10, atol 1e-12
(`lorenz1960_baseline.py:11-21`), and the PINN is scored against it — it is
never shown it during training.

So the training signal is only two things: the **rule** (the ODE) and the
**starting point** (the initial condition, "IC"). There are no boundary
conditions, because this is an initial-value ODE, not a PDE — the loss in both
modes contains at most a physics term and an IC term (`pinn.py:151-164`).

Both modes check the physics at the same **3000 collocation points** in
`[0, 1]` (`config.py:45`). "Collocation points" is just the jargon for the
moments in time where we ask "does the rule hold here?"

---

## 2. Mode 1 — the single-domain PINN: "guess the whole path, then check it"

### 2.1 The plain-language story

The network is a flexible formula that takes a time and directly answers:
"here is the position (x, y, z) at that moment."

The starting point is built into the output formula rather than learned, so
the answer at t = 0 is exactly right for *any* weights. Then we check the rule
at all 3000 checkpoints: does the speed this path implies match the rule at
the position it predicts? Where it doesn't, we nudge the network's weights and
check again — 20,000 times.

The analogy: a student draws the entire curve in one go, then checks 3000
spots on it with a ruler. Every spot is checked independently — fixing one
spot does not touch the others.

### 2.2 What actually happens

**Step 1 — draw the 3000 checkpoints.** Latin hypercube sampling (LHS):
partition [0, 1] into 3000 equal bins and draw one random point per bin, then
shuffle. Compared to plain random sampling this guarantees no empty stretch of
time and no clumps. The average gap between neighbouring points is 1/3000 ≈
3.3e-4.

**[code]** `train.py:27-32`; the code comment names its source: "Latin
hypercube sampling over [t0, tf], following Matthews & Bihlo (PinnDE)".
**[test]** the grid is seeded and reproducible, and unsorted (LHS order).
**[data]** measured on the actual grid: 3000 points, mean gap 0.000333,
identical across calls with the same seed.

**Step 2 — the network and the built-in start point.** A 4-hidden-layer, 60-unit
tanh MLP (11,283 parameters) maps one time to three outputs
(`pinn.py:29-41`). Its raw output `N(t)` is wrapped in the **hard trial
solution**:

```
u(t) = u₀ + g(t) · N(t),     g(t) = (t − t₀)/(t_f − t₀)      (= t here)
```

**[code]** `pinn.py:50-55`. Because `g(0) = 0`, we get `u(0) = u₀` exactly, to
the bit, whatever the network does — the IC is enforced by construction, not
by a penalty, and the loss needs no IC term and no weight γ.
**[test]** `test_pinn.py:10-14` (`test_hard_ic_exact`) asserts this with random
untrained weights.
The alternative `ic="soft"` mode returns the raw network and adds a penalty
`γ·‖N(t₀) − u₀‖²` — the classic Raissi-style formulation
(`pinn.py:137-141`, `config.py:36-37`); hard is the default and removes the
γ-balancing problem entirely.

**Step 3 — get the speed by calculus, exactly.** The physics needs `du/dt`.
PyTorch computes the exact analytic derivative of the trial solution with
respect to the input time — automatic differentiation, not a finite
difference. One subtlety is load-bearing: `create_graph=True`
(`pinn.py:119`) keeps the derivative itself differentiable, so that when the
loss is differentiated the gradient correctly flows *through* the derivative
operator. Without it the training gradient would be silently wrong. This is
why PINN training is a second-order autograd problem, and it is the main
reason Mode 1 is the slower of the two.

**[code]** `pinn.py:139-142`; the input times must be a `requires_grad` leaf,
which is why the loop passes `grid.clone().requires_grad_(True)` each epoch
(`train.py:75`).

**Step 4 — the rule-breaking score.** The residual is the mismatch between
the path's speed and the rule's speed, at every point and for all three
variables at once:

```
r_i = du/dt(t_i) − f(u(t_i))     with  f = (−0.1·yz, 1.6·xz, −0.75·xy)
```

**[code]** `pinn.py:117-125` (the physics, written as pure testable math),
`pinn.py:128-144` (one residual evaluation), `pinn.py:158` (the loss is
`mean(r²)` over all 3000 × 3 = 9000 entries).
**[test]** `test_pinn.py:17-25` feeds the *true* solution into the residual
and requires it to vanish; `test_pinn.py:152-155` proves the per-point bookkeeping
(each point's `|r|²/(3·N_c)` sums exactly to the reported loss).

**The key structural fact: the checkpoints are independent.** Entry (i, j) of
the residual depends only on the weights and on t_i — point i's residual
cannot see point j. The collocation system is *diagonal*.

**[test]** `test_pinn.py:449-470` proves a point's residual is unchanged when
its batchmates are removed; `test_pinn.py:473-518` proves the stronger
statement that one batched backward pass over all 3000 points trains
*identically* (to float32 round-off) to a literal point-by-point loop. The
rendered proof figure is `runs/parallel_proof/parallel_vs_sequential.png`.

**Step 5 — nudge and repeat.** Full-batch Adam (no minibatching), 20,000
epochs, learning rate decaying linearly 1e-3 → 1e-4, with optional L-BFGS
polishing (off by default).

**[code]** `train.py:64-68` (optimizer and schedule), `train.py:70-104` (the
loop: forward → residual → backward → one Adam step), `config.py:39-42`.

**What Mode 1 is, mathematically.** Zero residual at all 3000 points means the
*exact analytic* derivative of the ansatz satisfies the ODE there. Since f is
polynomial (hence Lipschitz on bounded sets), uniqueness of IVP solutions
means a function satisfying this everywhere with `u(0) = u₀` **is** the
solution. All remaining error is optimization and network expressivity —
there is no discretization error at all.

### 2.3 What the inputs and outputs are

Input: one column of times, shape `(3000, 1)`. Output: one state per row,
shape `(3000, 3)`. The times are fixed for the entire run — the same tensor
object is reused every epoch (`train.py:41`, cloned fresh at `train.py:75`).
**[test]** `test_pinn.py:166` asserts the collocation times never change
across snapshot epochs.

---

## 3. Mode 2 — the sequential rollout: "predict the speed, then walk"

Enabled by `Config(sequential=True)` (`config.py:53`); the run lands in its
own directory tag `4x60_seq` so it can never overwrite the single-domain run
(`config.py:97-104`, `run_single.py:5`). Same network, same 3000 LHS points,
same Adam loop, same loss formula — but the **meaning of the network's output
changes completely**.

### 3.1 The plain-language story

The network no longer answers "where am I?" — it answers "how fast am I
moving right now?" The position is built by walking: start at the known
start, step to checkpoint 1 using the speed there, then to checkpoint 2 using
the speed there, and so on — like adding up small steps.

The analogy: instead of drawing the whole curve, the student gives a direction
at each step and you literally walk the path. Each step depends on all the
steps before it — early mistakes drag everything after them. The reward:
walking in steps needs no calculus, and the starting point is guaranteed
exact because you *begin* there. The cost: the walk *accumulates* — a wrong
direction at one checkpoint shifts every later position — so at an equally
small physics loss, the walked trajectory is less accurate than the
independently checked one (measured in §3.3).

### 3.2 What actually happens

**The walk.** The 3000 times are sorted ascending, and the state is built as
a cumulative sum of small steps:

```
u(t₀)  = u₀                                       ← exact, by construction
u(t_i) = u(t_{i−1}) + (t_i − t_{i−1}) · s_i       for i = 1 … 3000
s_1    = N(t_1),    s_i = (N(t_{i−1}) + N(t_i))/2 for i > 1
```

**[code]** `pinn.py:63-114`. `torch.argsort` sorts the unsorted LHS times, the
inverse permutation scatters the walked values back to the caller's order, and
`torch.cumsum` performs the walk in one kernel. The first interval steps with
its own endpoint output — the walk has no left neighbour to average with — and
every later interval with the **trapezoid rule**, which makes the walk
second-order accurate in the step size. The docstring is explicit about the
design (`pinn.py:74-77`): "Anchoring each point to the walk rather than to a
global constraint means information enters only through the initial condition
and travels forward along the chain — and the IC is then exact at t_start by
construction, not by penalty."
**[test]** `test_pinn.py:306-336` verifies the cumulative-sum implementation
equals a literal point-by-point walk to round-off, and that it is "genuinely a
walk, not an independent per-point evaluation"; `test_pinn.py:339-357` proves
`u(t₀)` equals `u₀` with `torch.equal` (bitwise) and that shuffling the input
points cannot change the walked trajectory.

Three things to understand deeply:

1. **The network predicts the slope, not the state.** The state is a discrete
   integral (a trapezoidal quadrature) of the network's output along the
   sorted times. The role of the network has fundamentally changed.
2. **The IC is exact by construction** — the walk literally starts at u₀ — so
   no IC penalty could ever do anything. This is why
   `sequential=True, ic="soft"` is rejected outright with a "silent no-op"
   error (`config.py:73-78`). **[test]** `test_pinn.py:438-442`.
3. **The mesh is the collocation grid.** In Mode 1 the 3000 points were merely
   places to sample a loss; here their spacing (≈ 3.3e-4) is the step size of
   an integrator. The same numbers now play a structural role.

**The derivative trick: no calculus at all.** By the walk's own construction,
`u_i − u_{i−1} = dt_i · s_i`, so the discrete difference quotient of the
walked trajectory **is** the interval's quadrature slope `s_i`, identically.
The residual needs no autograd:

```
r_i = s_i − f(u_i)      with  s_i = (N(t_{i−1}) + N(t_i))/2,  s_1 = N(t_1)
```

**[code]** `pinn.py:131-138` — the sequential branch sets `dudt` to the walk's
own slopes; the `create_graph` machinery from Mode 1 is never invoked.
**[test]** `test_pinn.py:360-383` verifies `dudt == s` exactly and
`r == s − f`, in float64 — with a docstring explaining why the *reconstructed*
difference quotient would fail in float32 (subtracting nearly-equal states and
dividing by a small dt amplifies round-off by ~eps/dt; the code goes the safe
direction by defining dudt as the walk's own slope, so no cancellation ever
happens).

### 3.3 What the walk enforces, and the error floor

Set the residual to zero and see what equation the trajectory satisfies:

```
r_i = 0  ⟺  s_i = f(u_i)  ⟺  u_i = u_{i−1} + dt_i · (f(u_{i−1}) + f(u_i))/2
```

That last line is the **implicit trapezoid** — a second-order integrator — on
the collocation grid. (The first interval reduces to backward Euler, whose
local error is also O(dt²), so the order is preserved.) So Mode 2 does not
train a network to satisfy the continuous ODE; it trains a network whose
cumulative integral is a trapezoidal solution of the ODE. This is the single
most important difference between the modes:

| | Mode 1 (single-domain) | Mode 2 (sequential walk) |
|---|---|---|
| residual enforced | continuous ODE, exact `du/dt` via autograd | discrete ODE: trapezoid consistency at the nodes |
| zero-loss limit | the true IVP solution | the trapezoid trajectory on whatever grid it is walked on |

The trapezoid is second-order accurate, so even a *perfect* network carries an
O(dt²) truncation error — a far smaller one than the O(dt) floor the original
backward-Euler walk imposed.

**[test]** `test_pinn.py:386-417` proves this in isolation: it feeds the *true*
right-hand side in as if the network were perfect, walks, and measures the
error dropping fourfold as the grid is refined (error ratio 3.9–4.1 at every
halving — exactly second order). At the 1001-point evaluation grid the floor
is ≈ 1.7e-7 — orders of magnitude below any accuracy this project reports, so
the read-out grid no longer caps the scheme. (The original first-order walk
measured ≈ 7e-5 there; its runs are preserved under `runs/4x60_seq_euler*`.)

**The consequence, visible in the runs:** the walk "re-integrates on whatever
grid it is handed" (`test_pinn.py:393-394`), so the **evaluation grid still
sets a floor** — but a second-order one. Under the original backward-Euler
walk, tripling the collocation count changed nothing because the floor sat at
the read-out grid; under the trapezoid walk that ceiling is gone:

**[data]** from the `run_summary.csv` files (the `4x60_seq` row is the
trapezoid walk; `4x60_seq_euler` is the original first-order walk, preserved
for comparison):

| run | walk | collocation pts | RMSE (combined) | ms/epoch | wall clock | loss, first → last |
|---|---|---|---|---|---|---|
| `4x60` | — | 3000 | **1.7e-5** | 17.6 | 351 s | 0.217 → 6.8e-9 |
| `4x60_seq` | trapezoid (current) | 3000 | **8.2e-5** | 8.3 | 166 s | 0.225 → 4.4e-9 |
| `4x60_seq_euler` | backward Euler (original, preserved) | 3000 | 1.2e-4 | 8.1 | 161 s | 0.225 → 4.1e-9 |
| `4x60_seq_n6000` | backward Euler (original) | 6000 | 1.2e-4 | 9.7 | 193 s | — |
| `4x60_seq_n12000` | backward Euler (original) | 12000 | 1.2e-4 | 16.1 | 322 s | — |

Two findings, one from each generation of the walk:

- **The first-order walk was read-out-limited.** Under backward Euler,
  tripling the collocation count changed the RMSE not at all (1.2e-4 at 3000,
  6000 and 12000 points): the floor belonged to the 1001-point read-out grid,
  not to training. Mode 2 looked ~7× worse than Mode 1 for exactly that
  reason.
- **The trapezoid walk removes that floor but does not close the gap.** With
  the discretization floor now at 1.7e-7 (proven above), the RMSE improves to
  8.2e-5 — still ~5× above Mode 1's 1.7e-5. The remaining gap is *not*
  discretization: both runs drive the physics residual to ~1e-9 (Mode 2's best
  loss is actually the lower of the two). The difference is structural. In
  Mode 1 a residual error at t_i only degrades the fit near t_i; in Mode 2 the
  walk *integrates* its errors — a slope error of size ε at time t becomes a
  position error of roughly ε·(T−t) at every later point — so at matched
  residual levels the walked trajectory is inherently less accurate. This is
  the proof this branch needed: the batched single-domain scheme wins on this
  benchmark for a structural reason, not because the alternative was left
  untuned.

### 3.4 The gradient: a 3000-step chain

In Mode 1 the loss was diagonal — 3000 independent contributions. Here it is
not. Differentiating the residual shows that **the residual at point i depends
on the network outputs at all earlier points**, because they were all summed
into `u_i`:

```
∂r_i/∂N(t_k) ≠ 0   for every k ≤ i
```

A change in the network's early output shifts every later state. When
`loss.backward()` runs, PyTorch executes the adjoint of the cumulative sum (a
reverse cumulative sum), so the gradient accumulates backward through the
chain: each `N(t_k)` receives the sum of all later residuals' gradients,
weighted by the step sizes and by the Jacobian of f along the walked
trajectory. Structurally this is backpropagation through time of a 3000-step
unrolled integrator — in one `backward()` call.

This is what "information enters only through the initial condition and
travels forward along the chain" (`pinn.py:74-77`) means: the forward pass is
causal *by construction*, and the backward pass is where late errors reach
back and adjust early slopes.

**The payoff:** Mode 2 is ~2.1× faster per epoch (8.3 vs 17.6 ms) because it
trades Mode 1's second-order autograd graph for one cumulative sum and its
cheap reverse-cumsum adjoint. **[data]** `runs/*/run_summary.csv`; it also
crosses loss 1e-4 sooner (epoch 55 vs 71).

---

## 4. The two modes side by side

| aspect | Mode 1 — single-domain | Mode 2 — sequential rollout |
|---|---|---|
| network output means | the state `u(t)` | the slope `N(t)`; state = walked integral |
| initial condition | exact via the trial ansatz (`pinn.py:50-55`) | exact via the walk anchor (`pinn.py:113`) |
| derivative | exact, second-order autograd (`pinn.py:141`) | the walk's own step, no autograd (`pinn.py:138`) |
| zero-loss limit | continuous ODE | implicit trapezoid on the collocation grid |
| gradient structure | diagonal, points independent | 3000-step coupled chain (BPTT-like) |
| what you can evaluate | a true function of t, valid at any t | a trajectory on a given mesh — grid-tied |
| error floor | optimization only | O(dt²) of the read-out grid (1.7e-7 at 1001 pts) |
| measured RMSE | 1.7e-5 | 8.2e-5 (gap = error accumulation, not discretization) |
| speed per epoch | 17.6 ms | 8.3 ms (~2.1× faster) |

---

## 5. Where this sits in the literature

### 5.1 Classical time-marching / sequential PINN training

The established "sequential in time" schemes split [0, T] into windows and
train them one after another, each window anchored to the previous one's
endpoint. Krishnapriyan et al. showed that single-domain PINNs fail on longer
horizons and proposed training consecutive time domains sequentially,
reusing the previous segment's solution as the next initial condition
[5]. Mattey & Ghosh solve time-dependent systems "sequentially over successive
time segments using a single neural network" [6]. Fabiani et al. extend
physics-informed random-projection networks over successive intervals for
IVPs of ODEs specifically, with a variable-step scheme [7]. Penwarden et al.
generalize the idea into causal sweeping strategies with temporal
decompositions [8], and Wang et al. enforce the same forward-in-time
information flow softly, through causality-respecting loss weights [9].

Our rollout shares the philosophy — anchor everything to the past, let
information flow forward — but is mechanistically different: there are no
windows, no retraining stages, no frozen handoff. The march lives *inside the
trial solution*, executed on every forward pass, and the whole walk is one
differentiable computation, so late-time residuals can still adjust early-time
slopes through the chain gradient. Causality is structural, not reweighted.

### 5.2 The closest match: network-as-derivative, integrated from the IC

The construction in Mode 2 — network outputs the derivative, state obtained by
integrating the network from the exact IC, physics enforced as consistency
between the two — is the **integral-form trial solution**. It is used for
exactly this kind of problem (systems of ODEs) by Mattheakis et al., whose
Hamiltonian neural networks parametrize solutions as `q(t) = q(0) + ∫ N` and
train by enforcing the equations of motion on the integrated trajectory [10].
The general framing — a network-parametrized vector field advanced by a
numerical integrator — is neural ODEs [11]. Our walk is a fixed-quadrature
(implicit trapezoid) instance: the network learns the vector field, the trial
solution integrates it, and the loss is the integrator's own consistency
residual.

So when writing this up: cite [5, 6, 7, 8] for *sequential / time-marching
PINN training* in general, [10] for the *integral-form construction* our
rollout implements, [11] for the *network-as-derivative* view, and [9] for the
*causality* argument.

### 5.3 What to cite for each claim in this document

| Claim | Cite |
|---|---|
| Hard trial solution with exact IC/BC by construction | Lagaris et al. 1998 [1]; McFall & Mahan 2009 [2] |
| PINN formulation with soft-constraint losses | Raissi et al. 2019 [3] |
| LHS collocation, 4×60 tanh, 20k Adam + lr decay recipe | PinnDE, Matthews & Bihlo 2024 [4] |
| PINN failure on long horizons; sequential time training | Krishnapriyan 2021 [5]; Mattey & Ghosh 2022 [6]; Penwarden 2023 [8] |
| Marching/extending solutions over successive intervals for ODE IVPs | Fabiani et al. 2023 [7] |
| Integral-form trial solution (network = derivative) | Mattheakis et al. 2022 [10] |
| Neural network as vector field integrated by a scheme | Chen et al. 2018 [11] |
| Causality in PINN training | Wang et al. 2024 [9] |
| Sequential schemes accumulate error across windows | Lin & Chen 2023 [12] |
| Windowed training of chaotic Lorenz uses a batched hard-IC PINN per window | Wang et al. 2024 [9], Appendix E |

### 5.4 "But papers say single-domain is not ideal — is sequential the ideal approach?"

Short answer: **no paper says that, once you read what "sequential" means in
those papers.** Three facts settle it.

**Fact 1 — the literature's "sequential" is a training schedule wrapped
around Mode 1, not a different network.** In Krishnapriyan et al. the fix is
to "pose the problem as a sequence-to-sequence learning task, rather than
learning to predict the entire space-time at once" [5] — the network still
outputs the state and is still trained with an autograd residual; it is just
trained one time window at a time. Wang et al. are explicit about the
mechanics (Appendix E, verified from the paper): they solve the chaotic
Lorenz-63 system on `t ∈ [0, 20]` by splitting it into "40 disjoint time
windows of size Δt = 0.5", and inside each window they use a network
`t → [x_θ, y_θ, z_θ]` with the initial condition imposed exactly as
`x̂_θ(t) = x_θ(t)·t + x(0)`, trained on the residual loss [9]. That is
*our Mode 1*, line for line: state output (`pinn.py:33-36`), the same hard-IC
ansatz `u0 + t·N(t)` (`pinn.py:50-55`), autograd derivative (`pinn.py:141`),
residual loss (`pinn.py:158`). Their whole method is "run Mode 1 on many short
windows, hand the end state forward". Our benchmark is one such window —
`[0, 1]` is two of their Δt = 0.5 windows laid end to end — so on a domain
this short, the literature's sequential recipe **collapses back into what we
already do.**

**Fact 2 — what the papers actually criticise is single-domain training on
long, chaotic, multi-scale problems, and they say so.** Krishnapriyan:
"existing PINN methodologies can learn good models for relatively trivial
problems, [but] can easily fail to learn relevant physical phenomena for even
slightly more complex problems" [5]. Wang: PINNs "have not been successful in
simulating dynamical systems whose solution exhibits multi-scale, chaotic or
turbulent behavior" [9]. Penwarden: the failures appear "when solving forward
time-dependent PDEs with no data" that fall into "poor local minima" [8].
Lorenz-1960 on `[0, 1]` is none of these: it is a bounded, non-chaotic
system with two conserved quantities, and Mode 1 reaches RMSE 1.7e-5 with
the invariants drifting by only ~3e-5 (`runs/4x60/run_summary.csv`). The
failure the sequential papers cure does not occur here, so their cure has
nothing to buy. Wang et al. themselves position causal/sequential training
as "not ... a replacement" for ordinary training but "a crucial enhancement"
for the hard cases [9].

**Fact 3 — sequential schemes carry their own known pathology: error
accumulation across windows.** Lin & Chen (2023) improve the bc-PINN
"based on the characteristics of error propagation", because in a temporally
sequential scheme the error of each window is inherited by the next; their
error analysis shows that slowing "error accumulation speed" is what buys
accuracy [12]. This is the same mechanism we measured in our own walk
(§3.3–3.4): a slope error at time t becomes a position error at every later
time, and it is why Mode 2 sits at 8.2e-5 while its physics loss (4.4e-9) is
*lower* than Mode 1's (6.8e-9). Sequential is not free; it is a trade.

**What this means for the thesis.** Our Mode 2 is *not* the literature's
sequential method — it is the integral-form / network-as-derivative
construction [10, 11] with a trapezoid walk. The literature's sequential
method (windows of Mode 1) is the right tool when Mode 1 breaks, i.e. on
long or chaotic horizons; on `[0, 1]` it is the identity. So the correct,
literature-consistent statement is:

> Single-domain (batched) training is the method of record on `[0, 1]`
> because that is the regime in which the literature reports it works and
> our measurements confirm it (1.7e-5 vs 8.2e-5). Windowed sequential
> training [5, 6, 8, 9] is the documented escape route for longer or chaotic
> horizons, where each window is itself a batched PINN of exactly our form;
> the walk-based rollout (Mode 2) is kept as a tested, second-order
> alternative whose remaining gap is the error accumulation that the
> sequential literature itself identifies [12].

---

## 6. Evidence appendix: every claim, checked

| # | Claim | Evidence |
|---|---|---|
| 1 | 3000 collocation points over [0,1] | `config.py:45`; measured: 3000 points, t range (0.0003, 0.9998) |
| 2 | LHS sampling, reproducible, unsorted | `train.py:27-32` (comment names PinnDE); measured: same seed → identical grid; order not monotone |
| 3 | mean gap ≈ 1/3000 ≈ 3.3e-4 | measured: 0.000333 (min 4.9e-6, max 6.6e-4) |
| 4 | grid fixed for the whole run | created once `train.py:41`, reused `train.py:75`; `test_pinn.py:166` |
| 5 | network 4×60 tanh, 11,283 params | `pinn.py:29-41`; measured parameter count 11283 |
| 6 | hard IC: `u = u₀ + g·N`, g(0)=0 | `pinn.py:50-55`; `test_pinn.py:10-14` |
| 7 | soft IC alternative + γ | `pinn.py:137-141`, `config.py:36-37` |
| 8 | `du/dt` by autograd, `create_graph=True` needed | `pinn.py:139-142`; fresh leaf `train.py:75` |
| 9 | residual `r = du/dt − f(u)`, f = (−0.1yz, 1.6xz, −0.75xy) | `pinn.py:117-125`; coefficients verified numerically from `lorenz1960_baseline.py:28-32` |
| 10 | loss = mean over 3000×3 entries | `pinn.py:158`; `test_pinn.py:152-155` |
| 11 | points independent (diagonal); batched ≡ looped | `test_pinn.py:456-477`, `test_pinn.py:480-525`; `runs/parallel_proof/` |
| 12 | full-batch Adam, 20k epochs, lr 1e-3→1e-4 | `train.py:64-88`, `config.py:39-42` |
| 13 | reference = DOP853, rtol 1e-10, atol 1e-12, 1001 pts | `lorenz1960_baseline.py:11-21`, `config.py:127-131` |
| 14 | walk: sort → anchor at u₀ → trapezoid cumsum → scatter | `pinn.py:63-114`; `test_pinn.py:306-336` (equals literal loop) |
| 15 | IC exact to the bit; soft IC rejected as no-op | `test_pinn.py:339-357` (`torch.equal`); `config.py:73-78`; `test_pinn.py:438-442` |
| 16 | permutation invariance of the walk | `test_pinn.py:353-356` |
| 17 | derivative trick: `dudt` = walk slope, residual `= s − f(u)` | `pinn.py:131-138`; `test_pinn.py:360-383` (`torch.equal`) |
| 18 | zero-residual limit = implicit trapezoid | derivation in §3.3; `test_pinn.py:386-417` (second-order floor: 4.0× per halving, 1.7e-7 at 1001 pts) |
| 19 | old walk was read-out-limited; trapezoid removes the ceiling | `runs/4x60_seq_euler*/run_summary.csv`: RMSE 1.2e-4 at 3000/6000/12000 pts (backward Euler); trapezoid floor 1.7e-7 far below training error |
| 20 | Mode 2 ~2.1× faster, reaches 1e-4 sooner | 17.56 vs 8.31 ms/epoch; epochs-to-1e-4: 71 vs 55 (`run_summary.csv`) |
| 21 | gradient flows through the 3000-step chain | cumsum is differentiable; walk ≠ independent evaluation: `test_pinn.py:335-336`; design note `pinn.py:74-77` |
| 22 | Mode 1 RMSE 1.7e-5; Mode 2 8.2e-5 trapezoid / 1.2e-4 backward Euler | `runs/4x60/run_summary.csv`, `runs/4x60_seq/run_summary.csv`, `runs/4x60_seq_euler/run_summary.csv` |
| 23 | losses fall 0.217 → 6.8e-9 and 0.225 → 4.4e-9 | `runs/4x60/history/loss_history.csv`, `runs/4x60_seq/history/loss_history.csv` |

Checks 1-3, 5, 9 were re-run numerically while writing this document; the rest
come from the committed run artifacts and the test suite (`python -m pytest`).

---

## 7. References

All entries were verified programmatically against the arXiv and Crossref
registries; each lists its registry identifiers.

**Foundations of Mode 1**

1. I. E. Lagaris, A. Likas, D. I. Fotiadis, "Artificial neural networks for
   solving ordinary and partial differential equations," *IEEE Transactions on
   Neural Networks* 9(5), 987–1000 (1998). doi:10.1109/72.712178 — the trial
   solution with the initial/boundary condition built in exactly.
2. K. S. McFall, J. R. Mahan, "Artificial Neural Network Method for Solution
   of Boundary Value Problems With Exact Satisfaction of Arbitrary Input
   Boundary Conditions," *IEEE Transactions on Neural Networks* 20(7),
   1221–1233 (2009). doi:10.1109/TNN.2009.2020735 — hard-constraint
   satisfaction, generalized.
3. M. Raissi, P. Perdikaris, G. E. Karniadakis, "Physics-informed neural
   networks: A deep learning framework for solving forward and inverse
   problems involving nonlinear partial differential equations," *Journal of
   Computational Physics* 378, 686–707 (2019). doi:10.1016/j.jcp.2018.10.045 —
   the PINN formulation with soft-constraint losses (our `ic="soft"` mode).
4. J. Matthews, A. Bihlo, "PinnDE: Physics-Informed Neural Networks for
   Solving Differential Equations," arXiv:2408.10011 (2024) — the recipe this
   repository follows for Mode 1 (LHS collocation points, network shape,
   Adam schedule, hard-IC recommendation).

**Sequential / time-marching PINN training**

5. A. S. Krishnapriyan, A. Gholami, S. Zhe, R. Kirby, M. W. Mahoney,
   "Characterizing possible failure modes in physics-informed neural
   networks," *Advances in Neural Information Processing Systems* 34 (NeurIPS
   2021). arXiv:2109.01050 — sequential training over consecutive time
   domains; the canonical "time-marching curriculum" reference.
6. R. Mattey, S. Ghosh, "A Physics Informed Neural Network for Time-Dependent
   Nonlinear and Higher Order Partial Differential Equations," *Computer
   Methods in Applied Mechanics and Engineering* 390, 114474 (2022).
   doi:10.1016/j.cma.2021.114474, arXiv:2106.07606 — solves the system
   "sequentially over successive time segments using a single neural network."
7. G. Fabiani, E. Galaris, L. Russo, C. Siettos, "Parsimonious
   Physics-Informed Random Projection Neural Networks for Initial-Value
   Problems of ODEs and index-1 DAEs," *Chaos* 33, 043128 (2023).
   doi:10.1063/5.0135903, arXiv:2203.05337 — ODE/DAE IVPs solved with
   interval-extension / continuation schemes.
8. M. Penwarden, A. D. Jagtap, S. Zhe, G. E. Karniadakis, R. M. Kirby, "A
   unified scalable framework for causal sweeping strategies for
   Physics-Informed Neural Networks (PINNs) and their temporal
   decompositions," *Journal of Computational Physics* 493, 112464 (2023).
   doi:10.1016/j.jcp.2023.112464, arXiv:2302.14227.
9. S. Wang, S. Sankaran, P. Perdikaris, "Respecting causality for training
   physics-informed neural networks," *Computer Methods in Applied Mechanics
   and Engineering* 421, 116813 (2024). doi:10.1016/j.cma.2024.116813,
   arXiv:2203.07404 — causality via loss weighting; our walk achieves the
   same information flow structurally.

**The integral-form construction (closest match to our rollout)**

10. M. Mattheakis, D. Sondak, A. S. Dogra, P. Protopapas, "Hamiltonian neural
    networks for solving equations of motion," *Physical Review E* 105, 065305
    (2022). doi:10.1103/PhysRevE.105.065305, arXiv:2001.11107 — trial
    solutions in integral form: the network outputs the derivative, the
    trajectory is the quadrature of that output from the exact IC.
11. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud, "Neural Ordinary
    Differential Equations," *Advances in Neural Information Processing
    Systems* 31 (NeurIPS 2018). arXiv:1806.07366 — the general
    network-as-vector-field view; our walk is a fixed-quadrature (backward
    Euler / right-rectangle) instance.

**The cost side of sequential schemes**

12. S. Lin, Y. Chen, "The improved backward compatible physics-informed
    neural networks for reducing error accumulation and applications in
    data-driven higher-order rogue waves," arXiv:2312.06715 (2023) — error
    analysis of the temporally sequential bc-PINN; shows that accuracy is
    governed by the *error accumulation speed* across successive time
    segments. Verified against the arXiv API on 2026-09-20 (arXiv record
    only; no journal DOI in the registry metadata).
