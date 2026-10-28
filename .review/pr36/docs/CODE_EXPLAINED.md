# pinn: the code explained (plain language, line by line)

This document explains **every file** in the `src/pinn/` PINN, in simple language,
for someone new to physics-informed neural networks. It covers *what* each line
does, *why* it is there, and *how* it helps solve the Lorenz-1960 system.

---

## 0. The big picture (read this first)

We want the three functions `x(t), y(t), z(t)` that solve a system of ODEs (the
Lorenz-1960 equations) on the time interval `t ∈ [0, 1]`.

The classic way (RK4/SciPy) *steps* through time in tiny increments. Our way is
different: we train a small **neural network** to **be** the solution. After training, you can plug in any `t` and it returns `(x, y, z)` directly. There is no stepping.

How can a network learn the solution **without being shown the answer**? Because
the ODEs themselves tell us what a correct solution must satisfy:

```
dx/dt = -0.10·y·z      dy/dt = 1.60·x·z      dz/dt = -0.75·x·y
```

If we move everything to one side, a perfect solution makes each of these zero:

```
r_x = dx/dt + 0.10·y·z   (should be 0 everywhere)
r_y = dy/dt − 1.60·x·z   (should be 0 everywhere)
r_z = dz/dt + 0.75·x·y   (should be 0 everywhere)
```

These `r`'s are called the **residual**: how badly the network breaks the
physics. We sample many time points, measure the residual, and nudge the
network's weights until the residual is ~0. That is the whole idea of a
**Physics-Informed Neural Network (PINN)**.

Two more ingredients make it work:
- The **initial condition** `x(0)=0.5, y(0)=0.75, z(0)=1.0` must hold, otherwise
  the ODE has infinitely many solutions. We build it into the network so it is
  true automatically (the "hard" trial solution).
- The **RK4/SciPy baseline** is used only at the end, to *check* how close we got.
  It never trains the network.

### How the files fit together

```
config.py   -> all the settings + access to the trusted baseline solver
pinn.py     -> the network, the physics residual, and the loss
train.py    -> the training loop (Adam, then optional L-BFGS), evaluation, and plots
test_pinn.py-> quick checks that the pieces are correct
__init__.py -> makes `import pinn` convenient
lorenz_pinn.ipynb -> a notebook to run everything on Kaggle/Colab (notebooks/)
```

Data flows left to right: `config` → build `pinn` → `train` it → save results.

---

## 1. Words you need (mini-glossary)

- **ODE / IVP**: an equation about a rate of change (`dx/dt = ...`) plus a known
  starting point. Solving it means finding the function over time.
- **Neural network (MLP)**: a flexible formula with tunable numbers ("weights").
  Here it maps one number `t` to three numbers `(x, y, z)`.
- **Residual**: how much the network violates the ODE at a point. Zero = perfect.
- **Automatic differentiation (autograd)**: PyTorch can compute exact derivatives
  of the network output with respect to its input `t`. That is how we get
  `dx/dt` without finite differences.
- **Collocation points**: the time samples where we check the residual.
- **Loss**: one number we minimize. Here it is the average squared residual.
- **Adam / L-BFGS**: two optimizers (recipes for adjusting weights). Adam is a
  fast general workhorse; L-BFGS is a slower, more precise "polisher."
- **Hard vs soft IC**: two ways to enforce the starting point. Build it in
  exactly (hard) or add a penalty for missing it (soft).

---

## 2. `src/baseline/lorenz1960_baseline.py` (imported, not part of pinn)

We do **not** modify this file. It is the locked, trusted reference. We only
import four things from it:

- `lorenz1960_coefficients(k, l)` → the numbers `[-0.10, 1.60, -0.75]` for k=2,l=1.
  Importing them means the equations live in exactly one place.
- `solve_lorenz1960_scipy(config=...)` → the high-accuracy numerical solution
  (SciPy DOP853), our "ground truth" for checking.
- `compute_error_metrics(reference, candidate)` → builds a table of MAE/RMSE/max
  error per variable.
- `Lorenz1960Config` → a settings object the baseline solver expects.

---

## 3. `src/pinn/config.py`: every setting in one place

```python
1  """Central configuration and locked-baseline access for the FYDP-2 PINN."""
2  from __future__ import annotations
```
Line 1 is a description. Line 2 lets us write modern type hints on older Python.

```python
4  import sys
5  from dataclasses import dataclass
6  from pathlib import Path
8  import numpy as np
```
Standard tools: `sys`/`Path` to find files, `dataclass` to make a tidy settings
object, `numpy` for arrays.

```python
10 _SRC_ROOT = Path(__file__).resolve().parents[1]   # .../src
11 _REPO_ROOT = _SRC_ROOT.parent                     # repository root
12 sys.path.insert(0, str(_SRC_ROOT / "baseline"))
```
`__file__` is this file's path. `parents[1]` goes up one level to `src/`, and its
parent is the repository root (used further down to anchor output paths).
Line 12 adds `src/baseline/` to Python's search path so the next import works.
**Why:** it lets `src/pinn/` reuse the locked baseline solver
without copying any code.

```python
13 from lorenz1960_baseline import (
14     Lorenz1960Config,
15     compute_error_metrics,
16     lorenz1960_coefficients,
17     solve_lorenz1960_scipy,
18 )
```
Pulls in the four baseline tools described in Section 2.

```python
20 ACTIVATIONS = ("tanh", "relu", "sigmoid", "gelu", "swish")
21 IC_MODES = ("hard", "soft")
```
The allowed choices for the activation function and the initial-condition mode.
Listing them lets us reject typos early.

```python
24 @dataclass(frozen=True)
25 class Config:
```
`@dataclass(frozen=True)` makes a small, **read-only** settings object: once you
create it, its fields cannot change. **Why frozen:** an experiment's settings
should not silently change halfway through. That keeps results reproducible.

```python
26     k: float = 2.0
27     l: float = 1.0
28     initial_state: tuple[float, float, float] = (0.5, 0.75, 1.0)
29     t_span: tuple[float, float] = (0.0, 1.0)
```
The physics: the two constants `k, l`; the start point; the time interval.
These match Section 4.2 of the PinnDE paper (Matthews & Bihlo) exactly.

```python
31     depth: int = 4
32     width: int = 60
33     activation: str = "tanh"
```
The network shape: 4 hidden layers, 60 neurons each, `tanh` activation, matching
the PinnDE paper's forward-PINN examples (Section 4.1). Change these to try other
architectures **without touching any code**.

```python
35     ic: str = "hard"
36     gamma: float = 1.0
```
`ic` picks how the start point is enforced ("hard" = built in, the default).
`gamma` is the weight of the soft-IC penalty (only used when `ic="soft"`).

```python
38     epochs: int = 20000
39     lbfgs_iters: int = 0
40     lr_start: float = 1e-3
41     lr_end: float = 1e-4
42     seed: int = 0
```
Training settings (PinnDE §4.1): 20000 Adam steps, the learning rate decaying
from 1e-3 to 1e-4, and a random seed so runs are repeatable. L-BFGS polishing is
implemented but **off by default** (`lbfgs_iters=0`); set e.g.
`Config(lbfgs_iters=5000)` for the paper's full recipe.

```python
44     n_collocation: int = 3000
```
How many time points we check the physics at (3000, drawn by Latin hypercube
sampling over `[0,1]`, following PinnDE).

```python
    results_dir: str = "src/pinn/results"
47     ckpt_dir: str = "src/pinn/history"
```
Where plots/tables go and where the saved model and telemetry go. Both are
tracked in git as the run of record.

```python
49     def __post_init__(self) -> None:
50         if self.activation not in ACTIVATIONS:
51             raise ValueError(f"activation must be one of {ACTIVATIONS}")
52         if self.ic not in IC_MODES:
53             raise ValueError(f"ic must be one of {IC_MODES}")
```
Runs right after a `Config` is built. It rejects invalid choices immediately with
a clear message, instead of failing mysteriously later. **How it helps:** catches
typos like `activation="tan"` at the source.

```python
55     @property
56     def coefficients(self) -> np.ndarray:
57         return lorenz1960_coefficients(self.k, self.l)
```
A convenient shortcut: `cfg.coefficients` returns `[-0.10, 1.60, -0.75]`, computed
by the baseline from `k, l`. **Why a property:** the equations are never re-typed
here. They always come from the locked baseline.

```python
60 def reference_trajectory(cfg: Config, n: int = 1001):
61     """Ground-truth (t, [x,y,z]) from the locked SciPy baseline solver."""
62     baseline = Lorenz1960Config(
63         k=cfg.k, l=cfg.l, initial_state=cfg.initial_state, t_span=cfg.t_span, n_eval=n
64     )
65     t, ys, _ = solve_lorenz1960_scipy(config=baseline)
66     return t, ys
```
Builds the settings the baseline solver wants, runs it, and returns the trusted
answer: `t` (times) and `ys` (an `n×3` array of x,y,z). This is what we compare
the PINN against. The `_` throws away the third return value (the raw solver
object) we don't need.

```python
69 __all__ = ["Config", "reference_trajectory", "compute_error_metrics", "ACTIVATIONS", "IC_MODES"]
```
The public names other files may import from here.

---

## 4. `src/pinn/pinn.py`: the network, the physics, the loss (the heart)

```python
4  import torch
5  from torch import Tensor, nn
7  from .config import Config
```
PyTorch (the deep-learning library) and our `Config`.

```python
9  _ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "swish": nn.SiLU}
```
A lookup table from a name to a PyTorch activation. ("swish" is PyTorch's `SiLU`.)
**Why:** lets `config.activation` (a string) select the real function.

```python
12 class PINN(nn.Module):
13     def __init__(self, cfg: Config) -> None:
14         super().__init__()
```
Defines our model. `nn.Module` is PyTorch's base class for networks; `super()`
does its required setup.

```python
15         act = _ACT[cfg.activation]
16         layers: list[nn.Module] = [nn.Linear(1, cfg.width), act()]
17         for _ in range(cfg.depth - 1):
18             layers += [nn.Linear(cfg.width, cfg.width), act()]
19         layers.append(nn.Linear(cfg.width, 3))
20         self.net = nn.Sequential(*layers)
```
Builds the network as a stack:
- Line 16: first layer maps the **1** input (`t`) to `width` numbers, then applies
  the activation.
- Lines 17-18: add `depth-1` more hidden layers of size `width`.
- Line 19: final layer maps `width` numbers to the **3** outputs (x, y, z).
- Line 20: `nn.Sequential` runs them in order.

`nn.Linear(a, b)` is just `output = weights·input + bias`, the tunable part.
The activation (e.g. `tanh`) adds the "bend" so the network can represent curved
functions. **Why tanh:** smooth and infinitely differentiable, which matters
because we take derivatives of the output.

```python
21         for m in self.net:
22             if isinstance(m, nn.Linear):
23                 nn.init.xavier_uniform_(m.weight)
24                 nn.init.zeros_(m.bias)
```
Sets sensible starting values for the weights (Xavier) and zero biases. **Why:**
good initialization makes training start smoothly instead of exploding.

```python
26         self.ic = cfg.ic
27         self.gamma = cfg.gamma
28         self.register_buffer("u0", torch.tensor([cfg.initial_state], dtype=torch.float32))
29         self.register_buffer("coeffs", torch.as_tensor(cfg.coefficients, dtype=torch.float32).reshape(1, 3))
30         self.t0, self.tf = float(cfg.t_span[0]), float(cfg.t_span[1])
```
Stores what the model needs later:
- `u0` = the start point `[0.5, 0.75, 1.0]`.
- `coeffs` = `[-0.10, 1.60, -0.75]`.
- `t0, tf` = the interval ends (0 and 1).

`register_buffer` is important: it makes `u0` and `coeffs` travel with the model
when we move it to a GPU (`.to("cuda")`). If we stored them as plain attributes,
they'd stay on the CPU and cause errors on GPU. **This is a correctness detail
for Kaggle.**

```python
32     def forward(self, t: Tensor) -> Tensor:
33         n = self.net(t)
34         if self.ic == "hard":
35             g = (t - self.t0) / (self.tf - self.t0)
36             return self.u0 + g * n
37         return n
```
`forward` is what the network computes. Given times `t` (shape `N×1`):
- Line 33: run the raw network → `n` (shape `N×3`).
- **Hard mode (default):** `g = (t - t0)/(tf - t0)`, which is 0 at the start.
  Return `u0 + g·n`. At `t = t0`, `g = 0`, so the output is exactly `u0`.
  **The initial condition is guaranteed, for any weights.** This is the
  "trial solution" trick.
- **Soft mode:** return the raw network; the start point is encouraged later by a
  penalty (see the loss).

**Why hard is nice:** we never have to tune how strongly to enforce the start
point. It is exact by construction, and the PinnDE paper says this trains
better for smooth problems like ours.

```python
def ode_rhs(u: Tensor, coeffs) -> Tensor:
    c = torch.as_tensor(coeffs, dtype=u.dtype, device=u.device).reshape(3)
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    return torch.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], dim=1)


def ode_residual(u: Tensor, dudt: Tensor, coeffs) -> Tensor:
    return dudt - ode_rhs(u, coeffs)
```
This is the physics, written as pure math (no network inside, so it is easy to test):
- Make sure the coefficients are a tensor on the same device/precision as `u`.
  (Doing this directly avoids a bug where numpy conversion fails on a GPU.)
- Split `u` into columns x, y, z.
- Build the right-hand side `f = [c_x·yz, c_y·xz, c_z·xy]`, the ODE's "what the
  derivative *should* be."
- Return `dudt − f`, the residual. Zero means the ODE is satisfied.

`ode_rhs` is split out because `f` is not only an intermediate on the way to the
residual — it is one of the columns recorded in the per-point breakdown, so it
needs a name of its own.

```python
def residual_parts(model: PINN, t: Tensor) -> ResidualParts:
    n = model.net(t)
    u = model.trial(t, n)
    cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
    dudt = torch.stack(cols, dim=1)
    f = ode_rhs(u, model.coeffs)
    return ResidualParts(n=n, u=u, dudt=dudt, f=f, r=dudt - f)


def residual(model: PINN, t: Tensor) -> Tensor:
    return residual_parts(model, t).r
```
Ties the network to the physics:
- `model.net(t)` is the raw network output `N(t)`; `model.trial(t, n)` wraps it
  into `u_T = u0 + g(t)·N(t)` so the start point is exact. (Together these are
  just `model(t)` — they are written separately here only so `N(t)` can be kept.)
- The key step. For each output column `j`, `torch.autograd.grad` computes its
  derivative with respect to `t`. This is `dx/dt, dy/dt, dz/dt`, computed
  **exactly** by PyTorch, not approximated. (`.sum()` is a standard trick:
  because each point's output depends only on its own `t`, summing then
  differentiating gives the per-point derivative. `create_graph=True` keeps the
  result differentiable so training can use it.)
- Stack the three derivatives into an `N×3` array, evaluate the physics, subtract.

**Why return all five pieces instead of just the residual?** Every one of them —
raw output, trial solution, derivative, right-hand side, residual — is a column
in the per-epoch breakdown described in the README. Because the training step has
to compute them anyway, handing them back means a snapshot costs nothing but the
CSV write: no second forward pass, no second call to autograd. `residual()` is
now a one-line wrapper for the callers that only want `r`.

```python
def loss_terms(model: PINN, t: Tensor) -> tuple[Tensor, Tensor, ResidualParts]:
    parts = residual_parts(model, t)
    res = parts.r.pow(2).mean()
    if model.ic == "soft":
        t0 = torch.full((1, 1), model.t0, dtype=t.dtype, device=t.device)
        ic = (model(t0) - model.u0).pow(2).mean()
    else:
        ic = torch.zeros((), dtype=res.dtype, device=res.device)
    return res, ic, parts


def pinn_loss(model: PINN, t: Tensor) -> Tensor:
    res, ic, _ = loss_terms(model, t)
    return res + model.gamma * ic if model.ic == "soft" else res
```
The one number we minimize:
- `parts.r.pow(2).mean()` is the average of the squared residual. Small = the
  network obeys the physics well.
- **Careful with that mean.** `parts.r` is an `N_c × 3` array, so `.mean()`
  divides by `3 × N_c`, not by `N_c`. The loss is the average over every residual
  *entry* — three per collocation point — not the average over points. This is
  why the breakdown's `loss_contribution` column is `r_sq / (3·N_c)`; summing it
  down a snapshot's `N_c` rows reproduces the reported loss exactly.
- The residual and initial-condition parts are returned separately so the loss
  can be plotted by component, and `parts` comes along for the snapshot writer.
- Lines 56-58 (soft mode only): add a penalty for missing the start point,
  `gamma · average((network(t0) − u0)²)`. In hard mode this term is unnecessary
  (the start point is already exact), so it's skipped.

---

## 5. `src/pinn/train.py`: train, evaluate, and save

```python
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from . import figures
from .config import Config, compute_error_metrics, reference_trajectory
from .history import TrainHistory, flat_params
from .pinn import PINN, loss_terms, pinn_loss, residual
```
Imports. `train.py` does the training and nothing else: the per-epoch telemetry
lives in `history.py`, and every plot lives in `figures.py`. **Why the split:**
`figures.py` never imports torch, so figures can be rebuilt from saved arrays on
a machine with no deep-learning stack.

```python
21 def get_device() -> torch.device:
22     return torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
Uses a GPU if one is available, otherwise the CPU. **How it helps:** the same
code runs fast on Kaggle's GPU and still works on a laptop.

```python
25 def set_seed(seed: int) -> None:
26     torch.manual_seed(seed)
27     np.random.seed(seed)
```
Fixes the randomness so a run can be repeated. Important for honest research.

```python
def make_grid(cfg, device):
    t0, tf = cfg.t_span
    sample = qmc.LatinHypercube(d=1, seed=cfg.seed).random(cfg.n_collocation)
    t = t0 + (tf - t0) * sample
    return torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
```
Creates the collocation points using **Latin hypercube sampling** (as in PinnDE):
`n_collocation` well-spread times in `[0,1]`, shaped as a column (`N×1`) because
the network expects one input per row. (In 1-D this behaves like a lightly
jittered even grid; it is seeded so runs repeat.)

```python
def train(cfg: Config) -> tuple[PINN, TrainHistory]:
    set_seed(cfg.seed)
    device = get_device()
    model = PINN(cfg).to(device)
    grid = make_grid(cfg, device)
    history = TrainHistory()
    t_ref, ys_ref = reference_trajectory(cfg, n=1001)
```
Set the seed, pick the device, build the model and move it to the device, make
the collocation grid, and prepare the record that training fills in. `t_ref`
holds the trusted solution. It is used only to *watch* the true error while training,
never in the loss.

```python
    adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_start)
46     decay = cfg.lr_end / cfg.lr_start
47     sched = torch.optim.lr_scheduler.LambdaLR(
48         adam, lambda e: 1.0 + (decay - 1.0) * min(e, cfg.epochs) / cfg.epochs
49     )
```
Sets up **Adam** (the first optimizer) with the starting learning rate. The
scheduler shrinks the learning rate **linearly** from `lr_start` (1e-3) to
`lr_end` (1e-4): at step 0 the multiplier is 1.0; at the last step it's
`decay = 0.1`, i.e. 1e-4. **Why decay:** big steps early to move fast, small steps
late to settle precisely. (This matches PinnDE's default schedule.)

```python
    for epoch in range(cfg.epochs):
        last = epoch == cfg.epochs - 1
        logging = epoch % cfg.log_every == 0 or last

        adam.zero_grad()
        res, ic, parts = loss_terms(model, grid.clone().requires_grad_(True))
        loss = res + model.gamma * ic if cfg.ic == "soft" else res
        loss.backward()

        if history.snapshots is not None and (epoch % cfg.snapshot_every == 0 or last):
            history.snapshots.write(epoch, parts)

        lr = adam.param_groups[0]["lr"]
        before = flat_params(model) if logging else None
        adam.step()
        sched.step()
        history.loss.append(loss.item())

        if logging:
            history.record_step(epoch, model=model, loss=loss, residual=res, ic=ic,
                                lr=lr, params_before=before)
        if epoch % cfg.eval_every == 0 or last:
            history.record_reference(epoch, float(np.mean((predict(model, t_ref) - ys_ref) ** 2)))
```
The training loop, repeated `epochs` times:
- `zero_grad`: clear old gradients.
- `loss_terms(...)`: measure how wrong the physics is now, keeping the residual
  and initial-condition parts separate so the loss can be plotted by component.
  `grid.clone().requires_grad_(True)` makes a fresh copy of the time points that
  PyTorch will track for derivatives (the residual needs `d/dt`).
- the snapshot line: every `snapshot_every` epochs, write all `N_c` points'
  state to `breakdown/epoch_<n>.csv`. It runs **before** `adam.step()`, so the
  numbers on disk are exactly the ones this epoch's gradient was built from.
  Disabled by default (`snapshot_every=0`), which is why the run of record is
  unaffected by any of this.
- `backward()`: compute how each weight affects the loss.
- `adam.step()`: nudge the weights to reduce the loss.
- `sched.step()`: shrink the learning rate a little.
- record the loss for the convergence plot.

The two `if` blocks are the instrumentation. Every `log_every` epochs
`record_step` snapshots the gradient norms (global and per layer), the learning
rate, and how far the weights actually moved. This is what the gradient-descent
figures are drawn from. `before` has to be captured *after* `backward()` but
*before* `adam.step()`, otherwise the "how far did we move" measurement would
compare a weight vector to itself. Every `eval_every` epochs the model is scored
against the trusted solution, which is what makes the "physics loss vs true
error" panel possible. **Why sample instead of recording every epoch:** cloning
the whole weight vector 20000 times would cost more than the training itself.

```python
    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=cfg.lbfgs_iters, history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad()
            loss = pinn_loss(model, grid.clone().requires_grad_(True))
            loss.backward()
            history.loss.append(loss.item())
            return loss

        lbfgs.step(closure)
```
The **second optimizer, L-BFGS**, polishes the result after Adam. L-BFGS is a
"quasi-Newton" method: it uses curvature information to take very precise steps,
which usually drives a PINN's error much lower than Adam alone. It needs a
`closure`, a function that recomputes the loss, because it may evaluate the loss
several times per step (line search). **Why this matters:** this two-step
Adam→L-BFGS recipe is exactly what the PinnDE paper uses for its forward-PINN ODE
examples, and it reaches the same accuracy as Adam in far fewer epochs. It is
**off by default** (`lbfgs_iters=0`, an Adam-only ablation); set e.g.
`lbfgs_iters=5000` to enable it.

```python
    return model, history
```
Hand back the trained model and the full optimisation record.

```python
76 def predict(model: PINN, t: np.ndarray) -> np.ndarray:
77     device = next(model.parameters()).device
78     tt = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
79     with torch.no_grad():
80         return model(tt).cpu().numpy().astype(np.float64)
```
Runs the trained model on given times and returns plain numpy arrays.
`torch.no_grad()` turns off derivative tracking (faster, we're just evaluating).
`.cpu().numpy()` brings the answer back from the GPU to normal numbers.

```python
83 def evaluate(model: PINN, cfg: Config) -> pd.DataFrame:
84     t, ys = reference_trajectory(cfg, n=1001)
85     return compute_error_metrics(ys, predict(model, t))
```
Compares the PINN's output against the trusted baseline and returns an error
table (MAE/RMSE/max per variable).

```python
def residual_at(model: PINN, t: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    tt = torch.as_tensor(t, ...).reshape(-1, 1).requires_grad_(True)
    return residual(model, tt).detach().cpu().numpy().astype(np.float64)
```
Evaluates the physics residual at any times you ask for, as plain numbers.
Unlike `predict`, this one *cannot* use `torch.no_grad()`, because the residual is
built from `d/dt`, so the derivative machinery has to stay on; `.detach()`
afterwards drops the graph.

```python
def collect_artifacts(model, history, cfg) -> figures.RunArtifacts:
    t, ref = reference_trajectory(cfg, n=1001)
    grid = make_grid(cfg, next(model.parameters()).device).cpu().numpy().reshape(-1)
    return figures.RunArtifacts(t=t, pred=predict(model, t), ref=ref, history=history, ...)
```
The bridge between the two halves of the code. It turns a trained model into a
bag of plain numpy arrays: the prediction, the truth, the collocation points, the
residual at those points, and the residual across the whole domain. Everything
torch-shaped stops here.

```python
def save_results(model, history, cfg) -> pd.DataFrame:
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.ckpt_path / "pinn.pt")
    return figures.write_run_report(collect_artifacts(model, history, cfg), cfg.results_path)
```
Saves the trained weights (`pinn.pt`) so you can reload the model without
retraining, then hands the run to `figures.write_run_report`, which writes
`metrics.csv`, the `results.png` report figure, and the whole `figures/` suite
into `src/pinn/results/`, with the bulk telemetry going to `src/pinn/history/`. Both paths
are anchored to the repository root, so it does not matter which folder you run
from. An earlier version used bare relative paths and quietly created a
nested duplicate results folder when run from inside the package.

`rebuild_figures(cfg)` is the reverse trip: load the checkpoint, load the saved
telemetry, redraw everything. Use it when tuning a plot, so a styling change
costs seconds instead of a full retrain.

```python
def main(cfg: Config | None = None):
    cfg = cfg or Config()
    model, history = train(cfg)
    metrics = save_results(model, history, cfg)
    print(metrics.to_string(index=False))
    return model, history, metrics

if __name__ == "__main__":
    main()
```
`main` runs the whole pipeline with default settings: train → save → print the
error table. The last two lines let you run the file directly with
`python -m pinn.train`.

---

## 5b. `src/pinn/history.py`: what the optimiser was doing

`TrainHistory` is a plain record with one list per quantity. `loss` gets an entry
every single iteration; everything else is sampled. Two methods fill it:

- `record_step(...)` is called just after `optimizer.step()`. It reads the
  gradient that `backward()` left on every parameter, reduces it to one global
  norm plus one norm per weight matrix, and measures the distance the weights
  actually travelled. **Why per-layer norms:** if the gradient in the first layer
  is orders of magnitude smaller than in the last, the network is suffering
  vanishing gradients and depth is being wasted. The per-layer panel makes that
  visible immediately.
- `record_reference(epoch, mse)` stores how far the prediction is from the
  trusted solution at that point in training.

`diagnostics_frame()` and `reference_frame()` dump the record to tidy DataFrames,
which the report writes as CSVs under `src/pinn/history/`, and `from_saved()` reads
them back. **Why bother:** without them the diagnostics exist only in memory, so
redrawing a gradient figure would mean repeating a 20000-epoch run. They live
beside the checkpoint rather than in `results/` because they are half a megabyte
of regenerable numbers, not a result.

That last one deserves emphasis. A PINN minimises the *physics residual*, not the
error against a known answer, so a falling loss is not by itself proof of a good
solution. Recording both lets the report show whether driving the residual down
really did drive the true error down.

---

## 5c. `src/pinn/figures.py`: every plot in one place

The module imports numpy, pandas, matplotlib, seaborn and scipy, but no torch. It
takes arrays and returns figures.

- `set_style()` applies the house style once: seaborn `whitegrid`, a
  colourblind-safe palette, serif type, 300 dpi. Every figure inherits it, so the
  report looks like one document rather than nine unrelated plots.
- `RunArtifacts` is the record `train.collect_artifacts` builds; `metrics` is a
  property, so the error table is always recomputed from the arrays it holds and
  cannot drift out of sync with them.
- `fig_*` functions each build one figure and return it. They can be called
  individually from the notebook.
- `save_figure(fig, outdir, name)` writes PNG (slides) and PDF (LaTeX) and closes
  the figure. Closing matters: without it, a sweep of many runs runs out of memory.
- `generate_all(run, outdir)` renders the nine-figure suite;
  `write_run_report(run, outdir)` adds the CSVs and the `results.png` report
  figure on top.
- `generate_sweep(df, outdir)` covers the *next* phase: give it one row per run
  with `depth`, `width`, `activation`, `seed` and a metric, and it draws the
  depth x width heatmap, the activation comparison, and the seed plot.

Two details worth knowing:

`_log_trend` smooths curves by taking a rolling mean **of the logarithm**, not of
the values. A loss that falls from 1e-1 to 1e-6 is dominated by its first few
points on a linear average, so an ordinary moving average lags thousands of
epochs behind the curve and tells a false story about convergence.

`fig_invariant_drift` is a physics check rather than an accuracy check. For
`du_i/dt = a_i u_j u_k`, any weighted sum `I = alpha x^2 + beta y^2 + gamma z^2`
is conserved whenever `(alpha, beta, gamma)` is perpendicular to `a`, because
`dI/dt = 2xyz(alpha a_1 + beta a_2 + gamma a_3)`. The code finds those weights as
the null space of `a`, so the invariants are derived, not hard-coded, and stay
correct if `k` and `l` change. Plotting the drift for the PINN next to the
reference shows how much conserved structure the network quietly threw away.

---

## 6. `src/pinn/test_pinn.py`: correctness checks

```python
8  def test_hard_ic_exact():
9      model = PINN(Config())
10     out = model(torch.zeros(1, 1))
11     expected = torch.tensor([Config().initial_state])
12     assert torch.allclose(out, expected, atol=1e-6)
```
Checks the **hard initial condition**: at `t=0`, even with random untrained
weights, the network must output exactly `(0.5, 0.75, 1.0)`. Proves the trial
solution works.

```python
15 def test_residual_zero_on_truth():
16     t, ys = reference_trajectory(Config(), n=1001)
17     dudt = np.gradient(ys, t, axis=0)
18     r = ode_residual(torch.tensor(ys, ...), torch.tensor(dudt, ...), Config().coefficients)
23     assert r[1:-1].abs().max().item() < 1e-2
```
Checks the **physics formula** itself: feed in the *true* solution and its
derivative (estimated with `np.gradient`); the residual must be ~0. This confirms
our residual and coefficient signs match the real ODE. (We skip the two endpoints
because the derivative estimate is less accurate there.)

```python
def test_training_reduces_loss():
    cfg = Config(epochs=100, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]
```
A fast end-to-end check: a short training run must cut the loss by at least 10×.
This exercises the real derivative computation, Adam, and L-BFGS together.

```python
def test_soft_ic_trains():
    cfg = Config(ic="soft", epochs=200, n_collocation=101, lbfgs_iters=50, seed=0)
    _, history = train(cfg)
    assert history.loss[-1] < 0.1 * history.loss[0]
```
Same idea for **soft mode**, confirming that code path also trains.

`test_history_records_diagnostics` checks the instrumentation itself: every
sampled series has the same length (a mismatch would silently misalign the x-axis
of the gradient plots), and no gradient norm is zero. The two figure tests render
the full suite and the sweep suite to a temporary folder and assert every file
exists and is non-empty. That is cheap insurance against a plotting call that only
breaks at the end of a long training run.

Run them all with: `python -m pytest` from the repository root.

---

## 7. `src/pinn/__init__.py` and `requirements.txt`

`__init__.py` (3 lines) just re-exports `Config` and `reference_trajectory` so you
can write `import pinn; pinn.Config()`.

`requirements.txt` lists the libraries needed: `torch, numpy, scipy, matplotlib,
seaborn, pandas, pytest`. Install with `pip install -r requirements.txt`.

---

## 8. `notebooks/lorenz_pinn.ipynb`: the runnable notebook (cell by cell)

1. **(markdown)** Title and one-paragraph description.
2. **(markdown)** Setup instructions for Kaggle/Colab.
3. **(code)** Optional `git clone` for Colab, then a few lines that add the repo
   folder to Python's path so `import pinn` works.
4. **(code)** Imports from `pinn.config` and `pinn.train`, and prints the device
   (GPU or CPU).
5. **(markdown)** "Configure."
6. **(code)** `cfg = Config()` holds the settings. Edit here to change the experiment.
7. **(markdown)** "Train."
8. **(code)** `model, history = train(cfg)` and prints the final loss.
9. **(markdown)** "Evaluate vs the locked baseline."
10. **(code)** `save_results(...)`, computes the final-state relative error, and
    shows the metrics table.
11. **(markdown)** "Figures."
12. **(code)** Displays `results.png` and the nine diagnostic figures inline.

---

## 9. How each piece helps the research goal

- **config.py** keeps every knob in one place, so trying a new architecture or
  equation is a one-line change. The upcoming architecture study depends on that.
- **pinn.py** encodes the method: the hard-IC trial solution + physics residual is
  what lets the network learn the solution from the equations alone.
- **train.py**'s training recipe (Adam, with optional L-BFGS polishing) is what
  actually drives the error down to a tiny value and matches the reference
  paper's protocol.
- **history.py** turns training from a single number into evidence: the report can
  argue *why* an architecture converged, rather than only that it did.
- **figures.py** makes results reproducible as artefacts. Every claim in the
  chapter maps to a figure that regenerates from one command, and the sweep
  functions are already in place for the depth x width x activation study.
- **test_pinn.py** gives quick confidence that the IC, the physics, and training
  are all correct before trusting any result.
- Comparing against the **locked baseline** keeps the science honest: the network
  is never shown the answer during training, only judged by it afterward.

---

## 10. Honest limits (what is and isn't verified)

- Results were produced and checked **on CPU**. The GPU (`cuda`) path is written
  to be device-agnostic and was reviewed, but it was **not executed on a real
  GPU** in this environment.
- The notebook's structure was validated and it only calls already-tested
  functions, but it was **not run end-to-end on Kaggle/Colab**. Do a first run
  there to confirm.
- The error plateaus around `1e-5`–`1e-6`. This is **consistent with** 32-bit
  float precision, but that was not proven (no float64 comparison was run); it may
  also be an optimization plateau.
- **Soft-IC mode** is implemented and passes its training test, but there is **no
  full hard-vs-soft comparison study yet**; that belongs to the next phase.
- This solves a **single** initial value problem. It does **not** learn a solution
  operator over many initial conditions (that is the PinnDE paper's DeepONet, out of scope
  here), and it has not been tested beyond `t ∈ [0, 1]`.


---

## What is not covered here

This walkthrough predates two later additions. Both are documented in full in the
project README rather than repeated here:

- **`history.SnapshotWriter` and the per-epoch breakdown** — the 27-column record
  of every collocation point at every snapshot epoch, the `point_summary.csv`
  aggregation built alongside it, and the `point_history()` / `residual_grid()`
  readers. See README §5.2–5.3.
- **`sweep.py`** — the depth × width architecture sweep, its `runs/<arch>/`
  output layout, and the `comparison.csv` table. See README §6, including why a
  single-seed sweep cannot support "architecture A beats architecture B".

Line numbers in the code excerpts above have been dropped where the source has
since shifted; the excerpts themselves match the current code.
