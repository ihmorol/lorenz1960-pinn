# fydp2 — The Code Explained (plain language, line by line)

This document explains **every file** in the `src/fydp2/` PINN, in simple language,
for someone new to physics-informed neural networks. It covers *what* each line
does, *why* it is there, and *how* it helps solve the Lorenz-1960 system.

---

## 0. The big picture (read this first)

We want the three functions `x(t), y(t), z(t)` that solve a system of ODEs (the
Lorenz-1960 equations) on the time interval `t ∈ [0, 1]`.

The classic way (RK4/SciPy) *steps* through time in tiny increments. Our way is
different: we train a small **neural network** to **be** the solution. After
training, you can plug in any `t` and it returns `(x, y, z)` directly — no
stepping.

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

These `r`'s are called the **residual** — how badly the network breaks the
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
train.py    -> the training loop (Adam then L-BFGS), evaluation, and plots
test_pinn.py-> quick checks that the pieces are correct (lives in tests/)
__init__.py -> makes `import fydp2` convenient
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
- **Loss**: one number we minimize — here, the average squared residual.
- **Adam / L-BFGS**: two optimizers (recipes for adjusting weights). Adam is a
  fast general workhorse; L-BFGS is a slower, more precise "polisher."
- **Hard vs soft IC**: two ways to enforce the starting point — build it in
  exactly (hard) or add a penalty for missing it (soft).

---

## 2. `src/baseline/lorenz1960_baseline.py` (imported, not part of fydp2)

We do **not** modify this file — it is the locked, trusted reference. We only
import four things from it:

- `lorenz1960_coefficients(k, l)` → the numbers `[-0.10, 1.60, -0.75]` for k=2,l=1.
  Importing them means the equations live in exactly one place.
- `solve_lorenz1960_scipy(config=...)` → the high-accuracy numerical solution
  (SciPy DOP853), our "ground truth" for checking.
- `compute_error_metrics(reference, candidate)` → builds a table of MAE/RMSE/max
  error per variable.
- `Lorenz1960Config` → a settings object the baseline solver expects.

---

## 3. `src/fydp2/config.py` — every setting in one place

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
10 _SRC_ROOT = Path(__file__).resolve().parents[1]
11 sys.path.insert(0, str(_SRC_ROOT / "baseline"))
```
`__file__` is this file's path. `parents[1]` goes up one level to `src/`.
Line 11 adds `src/baseline/` to Python's search path so the next import works.
**Why:** it lets `src/fydp2/` reuse the locked baseline solver
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
should not silently change halfway through — that keeps results reproducible.

```python
26     k: float = 2.0
27     l: float = 1.0
28     initial_state: tuple[float, float, float] = (0.5, 0.75, 1.0)
29     t_span: tuple[float, float] = (0.0, 1.0)
```
The physics: the two constants `k, l`; the start point; the time interval.
These match paper1 Section 4.2 exactly.

```python
31     depth: int = 4
32     width: int = 60
33     activation: str = "tanh"
```
The network shape: 4 hidden layers, 60 neurons each, `tanh` activation — matching
paper1's forward-PINN examples (Section 4.1). Change these to try other
architectures **without touching any code**.

```python
35     ic: str = "hard"
36     gamma: float = 1.0
```
`ic` picks how the start point is enforced ("hard" = built in, the default).
`gamma` is the weight of the soft-IC penalty (only used when `ic="soft"`).

```python
38     epochs: int = 20000
39     lbfgs_iters: int = 5000
40     lr_start: float = 1e-3
41     lr_end: float = 1e-4
42     seed: int = 0
```
Training settings (paper1 §4.1): 20000 Adam steps then up to 5000 L-BFGS polishing
steps, the learning rate decaying from 1e-3 to 1e-4, and a random seed so runs are
repeatable.

```python
44     n_collocation: int = 3000
```
How many time points we check the physics at (3000, drawn by Latin hypercube
sampling over `[0,1]`, following paper1).

```python
46     results_dir: str = "results/fydp2"
47     ckpt_dir: str = "data/fydp2"
```
Where plots/tables go (tracked in git) and where the saved model goes (ignored
by git, since it is regenerable).

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
here — always sourced from the locked baseline.

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

## 4. `src/fydp2/pinn.py` — the network, the physics, the loss (the heart)

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

`nn.Linear(a, b)` is just `output = weights·input + bias` — the tunable part.
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
  Return `u0 + g·n`. At `t = t0`, `g = 0`, so the output is exactly `u0` —
  **the initial condition is guaranteed, for any weights.** This is the
  "trial solution" trick.
- **Soft mode:** return the raw network; the start point is encouraged later by a
  penalty (see the loss).

**Why hard is nice:** we never have to tune how strongly to enforce the start
point — it's exact by construction, and paper1 says this trains better for smooth
problems like ours.

```python
40 def ode_residual(u: Tensor, dudt: Tensor, coeffs) -> Tensor:
41     c = torch.as_tensor(coeffs, dtype=u.dtype, device=u.device).reshape(3)
42     x, y, z = u[:, 0], u[:, 1], u[:, 2]
43     f = torch.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], dim=1)
44     return dudt - f
```
This is the physics, written as pure math (no network inside — easy to test):
- Line 41: make sure the coefficients are a tensor on the same device/precision as
  `u`. (Doing this directly avoids a bug where numpy conversion fails on a GPU.)
- Line 42: split `u` into columns x, y, z.
- Line 43: build the right-hand side `f = [c_x·yz, c_y·xz, c_z·xy]` — the ODE's
  "what the derivative *should* be."
- Line 44: return `dudt − f`, the residual. Zero means the ODE is satisfied.

```python
47 def residual(model: PINN, t: Tensor) -> Tensor:
48     u = model(t)
49     cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
50     dudt = torch.stack(cols, dim=1)
51     return ode_residual(u, dudt, model.coeffs)
```
Ties the network to the physics:
- Line 48: get the network's `(x, y, z)` at the times `t`.
- Line 49: the key step. For each output column `j`, `torch.autograd.grad`
  computes its derivative with respect to `t` — this is `dx/dt, dy/dt, dz/dt`,
  computed **exactly** by PyTorch, not approximated. (`.sum()` is a standard trick:
  because each point's output depends only on its own `t`, summing then
  differentiating gives the per-point derivative. `create_graph=True` keeps the
  result differentiable so training can use it.)
- Line 50: stack the three derivatives into an `N×3` array.
- Line 51: plug the network's values and derivatives into the physics from above.

```python
54 def pinn_loss(model: PINN, t: Tensor) -> Tensor:
55     loss = residual(model, t).pow(2).mean()
56     if model.ic == "soft":
57         t0 = torch.full((1, 1), model.t0, dtype=t.dtype, device=t.device)
58         loss = loss + model.gamma * (model(t0) - model.u0).pow(2).mean()
59     return loss
```
The one number we minimize:
- Line 55: average of the squared residual over all points. Small = the network
  obeys the physics well.
- Lines 56-58 (soft mode only): add a penalty for missing the start point,
  `gamma · average((network(t0) − u0)²)`. In hard mode this term is unnecessary
  (the start point is already exact), so it's skipped.

---

## 5. `src/fydp2/train.py` — train, evaluate, and save

```python
6  import matplotlib
8  matplotlib.use("Agg")
9  import matplotlib.pyplot as plt
```
Loads the plotting library and picks the "Agg" backend, which draws to image
files without needing a screen. **Why:** lets training save PNGs on a headless
server (like Kaggle).

```python
10 import numpy as np
11 import pandas as pd
12 import torch
13 from torch import Tensor
15 from .config import Config, compute_error_metrics, reference_trajectory
16 from .pinn import PINN, pinn_loss
18 STATE = ("x", "y", "z")
```
Imports and a small label list for the three variables.

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
Creates the collocation points using **Latin hypercube sampling** (as in paper1):
`n_collocation` well-spread times in `[0,1]`, shaped as a column (`N×1`) because
the network expects one input per row. (In 1-D this behaves like a lightly
jittered even grid; it is seeded so runs repeat.)

```python
35 def train(cfg: Config) -> tuple[PINN, list[float]]:
36     set_seed(cfg.seed)
37     device = get_device()
38     model = PINN(cfg).to(device)
39     grid = make_grid(cfg, device)
40     history: list[float] = []
```
Set the seed, pick the device, build the model and move it to the device, make
the collocation grid, and prepare an empty list to record the loss each step.

```python
42     def loss_fn() -> Tensor:
43         return pinn_loss(model, grid.clone().requires_grad_(True))
```
A tiny helper that computes the current loss. `grid.clone().requires_grad_(True)`
makes a fresh copy of the time points that PyTorch will track for derivatives
(the residual needs `d/dt`). Cloning each call keeps the graph clean.

```python
45     adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_start)
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
50     for _ in range(cfg.epochs):
51         adam.zero_grad()
52         loss = loss_fn()
53         loss.backward()
54         adam.step()
55         sched.step()
56         history.append(loss.item())
```
The training loop, repeated `epochs` times:
- `zero_grad`: clear old gradients.
- `loss_fn()`: measure how wrong the physics is now.
- `backward()`: compute how each weight affects the loss.
- `adam.step()`: nudge the weights to reduce the loss.
- `sched.step()`: shrink the learning rate a little.
- record the loss for the convergence plot.

```python
58     if cfg.lbfgs_iters > 0:
59         lbfgs = torch.optim.LBFGS(
60             model.parameters(), max_iter=cfg.lbfgs_iters, history_size=50,
61             tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe",
62         )
64         def closure() -> Tensor:
65             lbfgs.zero_grad()
66             loss = loss_fn()
67             loss.backward()
68             history.append(loss.item())
69             return loss
71         lbfgs.step(closure)
```
The **second optimizer, L-BFGS**, polishes the result after Adam. L-BFGS is a
"quasi-Newton" method: it uses curvature information to take very precise steps,
which usually drives a PINN's error much lower than Adam alone. It needs a
`closure` — a function that recomputes the loss — because it may evaluate the loss
several times per step (line search). **Why this matters:** this two-step
Adam→L-BFGS recipe is exactly what paper1 uses for its forward-PINN ODE examples,
and it reaches the same accuracy as Adam in far fewer epochs.

```python
73     return model, history
```
Hand back the trained model and the loss history.

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
88 def save_results(model, history, cfg) -> pd.DataFrame:
89     ... mkdir results_dir and ckpt_dir ...
93     torch.save(model.state_dict(), ckpt / "pinn.pt")
```
Makes the output folders and saves the trained weights (`pinn.pt`) so you can
reload the model without retraining.

```python
95     t, ys = reference_trajectory(cfg, n=1001)
96     pred = predict(model, t)
97     metrics = compute_error_metrics(ys, pred)
98     metrics.to_csv(results / "metrics.csv", index=False)
```
Get the truth, get the prediction, compute the error table, and write it to
`metrics.csv`.

One combined figure (matching paper1's style), saved as **`results.png`**, with
three side-by-side panels:
1. **left** — the PINN solution (solid) over the reference (dashed) for x, y, z;
   they should overlap.
2. **middle** — the signed error per variable, with the **MSE in the title**.
3. **right** — the loss on a log scale; a **red dashed line marks where L-BFGS**
   takes over from Adam.

```python
134 def main(cfg: Config | None = None):
135     cfg = cfg or Config()
136     model, history = train(cfg)
137     metrics = save_results(model, history, cfg)
138     print(metrics.to_string(index=False))
139     return model, history, metrics
142 if __name__ == "__main__":
143     main()
```
`main` runs the whole pipeline with default settings: train → save → print the
error table. The last two lines let you run the file directly with
`python -m fydp2.train`.

---

## 6. `tests/test_pinn.py` — four quick correctness checks

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
26 def test_training_reduces_loss():
29     cfg = Config(epochs=100, n_collocation=101, lbfgs_iters=50, seed=0)
30     _, history = train(cfg)
31     assert history[-1] < 0.1 * history[0]
```
A fast end-to-end check: a short training run must cut the loss by at least 10×.
This exercises the real derivative computation, Adam, and L-BFGS together.

```python
34 def test_soft_ic_trains():
37     cfg = Config(ic="soft", epochs=200, n_collocation=101, lbfgs_iters=50, seed=0)
38     _, history = train(cfg)
39     assert history[-1] < 0.1 * history[0]
```
Same idea for **soft mode**, confirming that code path also trains.

Run them all with: `python -m pytest` from the repository root.

---

## 7. `src/fydp2/__init__.py` and `requirements.txt`

`__init__.py` (3 lines) just re-exports `Config` and `reference_trajectory` so you
can write `import fydp2; fydp2.Config()`.

`requirements.txt` lists the libraries needed: `torch, numpy, scipy, matplotlib,
pandas, pytest`. Install with `pip install -r requirements.txt`.

---

## 8. `notebooks/lorenz_pinn.ipynb` — the runnable notebook (cell by cell)

1. **(markdown)** Title and one-paragraph description.
2. **(markdown)** Setup instructions for Kaggle/Colab.
3. **(code)** Optional `git clone` for Colab, then a few lines that add the repo
   folder to Python's path so `import fydp2` works.
4. **(code)** Imports from `fydp2.config` and `fydp2.train`, and prints the device
   (GPU or CPU).
5. **(markdown)** "Configure."
6. **(code)** `cfg = Config()` — the settings; edit here to change the experiment.
7. **(markdown)** "Train."
8. **(code)** `model, history = train(cfg)` and prints the final loss.
9. **(markdown)** "Evaluate vs the locked baseline."
10. **(code)** `save_results(...)`, computes the final-state relative error, and
    shows the metrics table.
11. **(markdown)** "Plots."
12. **(code)** Displays the saved `results.png` figure inline.

---

## 9. How each piece helps the research goal

- **config.py** keeps every knob in one place, so trying a new architecture or
  equation is a one-line change — essential for the upcoming architecture study.
- **pinn.py** encodes the method: the hard-IC trial solution + physics residual is
  what lets the network learn the solution from the equations alone.
- **train.py**'s Adam→L-BFGS recipe is what actually drives the error down to a
  tiny value and matches the reference paper's protocol.
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
  full hard-vs-soft comparison study yet** — that belongs to the next phase.
- This solves a **single** initial value problem. It does **not** learn a solution
  operator over many initial conditions (that is paper1's DeepONet, out of scope
  here), and it has not been tested beyond `t ∈ [0, 1]`.
