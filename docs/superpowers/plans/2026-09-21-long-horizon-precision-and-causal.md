# Long-horizon precision batch run and causal windowed run: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the existing 4x60 Lorenz-1960 PINN over one closed orbit (t in [0, 13.26446]) in float64 under two schemes, plain batch (Run A) and the causal windowed recipe of Wang, Sankaran and Perdikaris 2024 (Run B), with every training and evaluation phase visible in figures and a Manim film.

**Architecture:** Core stays in `src/pinn/{config,problems,pinn,train,history,sweep}.py`; each method step becomes a named function under `src/pinn/functions/`; all plotting moves to `src/pinn/viz/` and core only calls `viz.generate_all`. Every new `Config` field defaults to today's value so `Config()` still reproduces the [0, 1] run of record. Branch A holds the plumbing and Run A; branch B stacks windows + causal loss + Run B on top.

**Tech Stack:** Python 3.10+, PyTorch >= 2.0, NumPy, SciPy, pandas, matplotlib, seaborn, plotly (HTML), Manim CE 0.21 (film), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-21-long-horizon-precision-and-causal-design.md`.
- Branch A: `feat/batch-precision-run` off `main`. Branch B: `feat/causal-window-training` off branch A.
- `Config()` with no arguments must keep producing the run-of-record numbers (tests assert it).
- Core files stay short; plotting only in `src/pinn/viz/`; method steps only in `src/pinn/functions/`; comments only where the maths is not obvious from the code; no existing figure or output filename is dropped.
- Commits end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- Run tests with `python -m pytest src/pinn/test_pinn.py -q -p no:warnings` from the repo root.
- Long-horizon runs are executed on Colab (GPU); locally only the tiny test configs run.

---

## Part 1: branch `feat/batch-precision-run`

### Task 1: Move plotting into `viz/`, create `functions/`

**Files:**
- Create: `src/pinn/viz/__init__.py`, `src/pinn/functions/__init__.py`
- Move: `src/pinn/figures.py` -> `src/pinn/viz/figures.py`
- Modify: `src/pinn/train.py:12`, `src/pinn/sweep.py:23`, `src/pinn/test_pinn.py` (imports)

**Interfaces:**
- Produces: `from pinn import viz`; `viz.figures` is the untouched former module; `viz.generate_all`, `viz.write_run_report`, `viz.RunArtifacts`, `viz.invariant_series`, `viz.set_style`, `viz.save_figure`, `viz.FORMATS`, `viz.generate_sweep` re-exported.

- [ ] **Step 1: Move the module and create the packages**

```bash
git mv src/pinn/figures.py src/pinn/viz/figures.py 2>/dev/null || (mkdir -p src/pinn/viz && git mv src/pinn/figures.py src/pinn/viz/figures.py)
mkdir -p src/pinn/functions
```

- [ ] **Step 2: Write `src/pinn/viz/__init__.py`**

```python
"""All plotting for the PINN. Core code calls only what is exported here."""
from .figures import (FORMATS, RunArtifacts, generate_all, generate_sweep, invariant_series,
                      save_figure, set_style, write_run_report, write_residual_surface_html,
                      fig_residual_evolution, fig_residual_profiles, fig_point_convergence)

__all__ = ["FORMATS", "RunArtifacts", "generate_all", "generate_sweep", "invariant_series",
           "save_figure", "set_style", "write_run_report", "write_residual_surface_html",
           "fig_residual_evolution", "fig_residual_profiles", "fig_point_convergence"]
```

- [ ] **Step 3: Write `src/pinn/functions/__init__.py`**

```python
"""One file per step of the method; the main modules compose these."""
```

- [ ] **Step 4: Fix the relative import inside the moved module**

In `src/pinn/viz/figures.py` change `from .config import compute_error_metrics` to `from ..config import compute_error_metrics`.

- [ ] **Step 5: Repoint imports**

In `src/pinn/train.py` replace `from . import figures` with `from . import viz as figures`.
In `src/pinn/sweep.py` replace `from . import figures` with `from . import viz as figures`.
In `src/pinn/test_pinn.py` replace every `from pinn import figures` with `from pinn import viz as figures`.

- [ ] **Step 6: Run the suite**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: all 18 tests pass.

- [ ] **Step 7: Commit**

```bash
git add -A src/pinn
git commit -m "refactor: move plotting into pinn.viz, add pinn.functions package

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 2: Method steps as functions: collocation, trial, derivative, physics, loss

**Files:**
- Create: `src/pinn/functions/collocation.py`, `trial.py`, `derivative.py`, `physics.py`, `losses.py`
- Modify: `src/pinn/pinn.py`, `src/pinn/train.py:27-33`
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces:
  - `latin_hypercube_points(t_span, n, seed) -> np.ndarray (n, 1)`; `uniform_points(t_span, n) -> np.ndarray (n, 1)`
  - `hard_initial_condition(t, n, u0, t0, tf, form) -> Tensor` with `form in ("span", "unit")`
  - `time_derivative(u, t) -> Tensor (N, dim)`
  - `residual(u, dudt, rhs) -> Tensor`; `lorenz1960_rhs(u, coeffs) -> Tensor`
  - `mean_squared_residual(r) -> Tensor`

- [ ] **Step 1: Write the failing tests**

Append to `src/pinn/test_pinn.py`:

```python
def test_hard_initial_condition_forms():
    import torch
    from pinn.functions.trial import hard_initial_condition

    u0 = torch.tensor([[0.5, 0.75, 1.0]])
    t = torch.tensor([[0.0], [5.0], [10.0]], requires_grad=True)
    n = torch.ones(3, 3)
    for form in ("span", "unit"):
        u = hard_initial_condition(t, n, u0, 0.0, 10.0, form)
        assert torch.allclose(u[0], u0[0])
    span = hard_initial_condition(t, n, u0, 0.0, 10.0, "span")
    unit = hard_initial_condition(t, n, u0, 0.0, 10.0, "unit")
    assert torch.allclose(span[2], unit[2] * 0 + u0[0] + 1.0)      # g = 1 at tf
    assert torch.allclose(unit[2], u0[0] + 10.0)                    # g = t - t0


def test_collocation_samplers_cover_the_span():
    from pinn.functions.collocation import latin_hypercube_points, uniform_points

    lhs = latin_hypercube_points((0.0, 2.0), 50, seed=0)
    uni = uniform_points((0.0, 2.0), 50)
    assert lhs.shape == uni.shape == (50, 1)
    assert 0.0 <= lhs.min() and lhs.max() <= 2.0
    assert uni[0, 0] == 0.0 and uni[-1, 0] == 2.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k "hard_initial_condition_forms or collocation_samplers"`
Expected: FAIL with `ModuleNotFoundError: No module named 'pinn.functions.trial'`.

- [ ] **Step 3: Write the function files**

`src/pinn/functions/collocation.py`:

```python
import numpy as np
from scipy.stats import qmc


def latin_hypercube_points(t_span: tuple[float, float], n: int, seed: int) -> np.ndarray:
    t0, tf = t_span
    sample = qmc.LatinHypercube(d=1, seed=seed).random(n)
    return (t0 + (tf - t0) * sample).reshape(-1, 1)


def uniform_points(t_span: tuple[float, float], n: int) -> np.ndarray:
    return np.linspace(t_span[0], t_span[1], n).reshape(-1, 1)
```

`src/pinn/functions/trial.py`:

```python
from torch import Tensor


def hard_initial_condition(t: Tensor, n: Tensor, u0: Tensor, t0: float, tf: float, form: str) -> Tensor:
    """u = u0 + g(t) N(t). 'span': g = (t - t0)/(tf - t0). 'unit': g = t - t0 (Lagaris 1998)."""
    g = (t - t0) / (tf - t0) if form == "span" else (t - t0)
    return u0 + g * n
```

`src/pinn/functions/derivative.py`:

```python
import torch
from torch import Tensor


def time_derivative(u: Tensor, t: Tensor) -> Tensor:
    cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
    return torch.stack(cols, dim=1)
```

`src/pinn/functions/physics.py`:

```python
from typing import Callable

import torch
from torch import Tensor


def lorenz1960_rhs(u: Tensor, coeffs) -> Tensor:
    c = torch.as_tensor(coeffs, dtype=u.dtype, device=u.device).reshape(3)
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    return torch.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], dim=1)


def residual(u: Tensor, dudt: Tensor, rhs: Callable[[Tensor], Tensor]) -> Tensor:
    return dudt - rhs(u)
```

`src/pinn/functions/losses.py`:

```python
from torch import Tensor


def mean_squared_residual(r: Tensor) -> Tensor:
    return r.pow(2).mean()
```

- [ ] **Step 4: Compose them in `pinn.py`**

Replace the body of `src/pinn/pinn.py` from `class PINN` to the end with:

```python
class PINN(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        act = _ACT[cfg.activation]
        layers: list[nn.Module] = [nn.Linear(1, cfg.width), act()]
        for _ in range(cfg.depth - 1):
            layers += [nn.Linear(cfg.width, cfg.width), act()]
        layers.append(nn.Linear(cfg.width, 3))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.ic, self.ic_scale, self.gamma = cfg.ic, cfg.ic_scale, cfg.gamma
        self.register_buffer("u0", torch.tensor([cfg.initial_state], dtype=torch.float32))
        self.register_buffer("coeffs", torch.as_tensor(cfg.coefficients, dtype=torch.float32).reshape(1, 3))
        self.t0, self.tf = float(cfg.t_span[0]), float(cfg.t_span[1])

    def rhs(self, u: Tensor) -> Tensor:
        return lorenz1960_rhs(u, self.coeffs)

    def trial(self, t: Tensor, n: Tensor) -> Tensor:
        if self.ic == "hard":
            return hard_initial_condition(t, n, self.u0, self.t0, self.tf, self.ic_scale)
        return n

    def forward(self, t: Tensor) -> Tensor:
        return self.trial(t, self.net(t))


def ode_rhs(u: Tensor, coeffs) -> Tensor:
    return lorenz1960_rhs(u, coeffs)


def residual_parts(model: PINN, t: Tensor) -> ResidualParts:
    n = model.net(t)
    u = model.trial(t, n)
    dudt = time_derivative(u, t)
    f = model.rhs(u)
    return ResidualParts(n=n, u=u, dudt=dudt, f=f, r=residual(u, dudt, model.rhs))


def residual_of(model: PINN, t: Tensor) -> Tensor:
    return residual_parts(model, t).r


def loss_terms(model: PINN, t: Tensor) -> tuple[Tensor, Tensor, ResidualParts]:
    parts = residual_parts(model, t)
    res = mean_squared_residual(parts.r)
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

and add at the top of `pinn.py`, after the existing imports:

```python
from .functions.derivative import time_derivative
from .functions.losses import mean_squared_residual
from .functions.physics import lorenz1960_rhs, residual
from .functions.trial import hard_initial_condition
```

`residual` was a public name in `pinn.py`; it is now `residual_of`. Update `src/pinn/train.py`: `from .pinn import PINN, loss_terms, pinn_loss, residual_of as residual`. Keep `ode_rhs` and the `ResidualParts` NamedTuple unchanged for the tests that use them.

- [ ] **Step 5: Add `ic_scale` to Config and use the sampler in `make_grid`**

In `src/pinn/config.py` add after `gamma`:

```python
    ic_scale: str = "span"    # "span": g = (t - t0)/(tf - t0); "unit": g = t - t0
```

and in `__post_init__`:

```python
        if self.ic_scale not in ("span", "unit"):
            raise ValueError("ic_scale must be 'span' or 'unit'")
```

In `src/pinn/train.py` replace `make_grid`:

```python
def make_grid(cfg: Config, device: torch.device) -> Tensor:
    t = latin_hypercube_points(cfg.t_span, cfg.n_collocation, cfg.seed)
    return torch.as_tensor(t, dtype=torch.float32, device=device)
```

with `from .functions.collocation import latin_hypercube_points` added to the imports and `from scipy.stats import qmc` removed.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 20 passed.

- [ ] **Step 7: Commit**

```bash
git add -A src/pinn
git commit -m "refactor: method steps as named functions; ic_scale knob

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 3: Precision, density and optimiser knobs

**Files:**
- Create: `src/pinn/functions/optimizers.py`
- Modify: `src/pinn/config.py`, `src/pinn/train.py`, `src/pinn/sweep.py:107-126`
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces: `Config.dtype`, `Config.torch_dtype`, `Config.points_per_unit`, `Config.eval_per_unit`, `Config.n_eval`, `Config.collocation`, `Config.lr_decay`, `Config.lr_decay_every`; `adam_with_decay(params, cfg) -> (optimizer, scheduler)`; `run_lbfgs(params, closure, iters, dtype) -> None`; the run tag gains `_f64` / `_unit`.

- [ ] **Step 1: Write the failing tests**

```python
def test_density_knobs_scale_with_the_window():
    cfg = Config(t_span=(0.0, 4.0), points_per_unit=100, eval_per_unit=10)
    assert cfg.n_collocation == 400 and cfg.n_eval == 41
    assert Config().n_collocation == 3000 and Config().n_eval == 1001


def test_float64_trains_end_to_end():
    import torch
    from pinn.train import train

    cfg = Config(dtype="float64", depth=1, width=8, epochs=5, n_collocation=16, lbfgs_iters=3,
                 log_every=5, eval_every=5, print_every=0)
    model, history = train(cfg)
    assert next(model.parameters()).dtype == torch.float64
    assert cfg.arch == "1x8_f64" and len(history.loss) > 5


def test_run_of_record_config_is_unchanged():
    cfg = Config()
    assert (cfg.dtype, cfg.ic_scale, cfg.collocation, cfg.lr_decay, cfg.n_eval) == \
        ("float32", "span", "lhs", None, 1001)
    assert cfg.arch == "4x60"
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k "density_knobs or float64_trains or run_of_record"`
Expected: FAIL with `TypeError: Config.__init__() got an unexpected keyword argument`.

- [ ] **Step 3: Extend `Config`**

In `src/pinn/config.py` add the fields (after `n_collocation`):

```python
    n_eval: int = 1001
    points_per_unit: float | None = None    # overrides n_collocation as density x window length
    eval_per_unit: float | None = None      # overrides n_eval the same way
    collocation: str = "lhs"                # "lhs" | "uniform"
    dtype: str = "float32"                  # "float32" | "float64"
    lr_decay: float | None = None           # None: linear lr_start -> lr_end; else StepLR gamma
    lr_decay_every: int = 5000
```

and in `__post_init__` (frozen dataclass, so `object.__setattr__`):

```python
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.collocation not in ("lhs", "uniform"):
            raise ValueError("collocation must be 'lhs' or 'uniform'")
        span = self.t_span[1] - self.t_span[0]
        if self.points_per_unit is not None:
            object.__setattr__(self, "n_collocation", int(round(self.points_per_unit * span)))
        if self.eval_per_unit is not None:
            object.__setattr__(self, "n_eval", int(round(self.eval_per_unit * span)) + 1)
```

add the property:

```python
    @property
    def torch_dtype(self):
        import torch
        return torch.float64 if self.dtype == "float64" else torch.float32
```

and replace `arch`:

```python
    @property
    def arch(self) -> str:
        tag = f"{self.depth}x{self.width}"
        if self.dtype == "float64":
            tag += "_f64"
        if self.ic_scale == "unit":
            tag += "_unit"
        return tag
```

- [ ] **Step 4: Write `src/pinn/functions/optimizers.py`**

```python
from typing import Callable

import torch


def adam_with_decay(params, cfg):
    adam = torch.optim.Adam(params, lr=cfg.lr_start)
    if cfg.lr_decay is None:
        decay = cfg.lr_end / cfg.lr_start
        sched = torch.optim.lr_scheduler.LambdaLR(
            adam, lambda e: 1.0 + (decay - 1.0) * min(e, cfg.epochs) / cfg.epochs)
    else:
        sched = torch.optim.lr_scheduler.StepLR(adam, cfg.lr_decay_every, cfg.lr_decay)
    return adam, sched


def run_lbfgs(params, closure: Callable[[], torch.Tensor], iters: int, dtype: torch.dtype) -> None:
    # 1e-14 is below float32 resolution, which is why L-BFGS used to stop after a few evals.
    tol = 1e-16 if dtype == torch.float64 else 1e-14
    opt = torch.optim.LBFGS(params, max_iter=iters, history_size=50, tolerance_grad=1e-12,
                            tolerance_change=tol, line_search_fn="strong_wolfe")

    def wrapped():
        opt.zero_grad()
        loss = closure()
        loss.backward()
        return loss

    opt.step(wrapped)
```

- [ ] **Step 5: Use dtype, sampler and optimizers in `train.py`**

Imports: add `from .functions.collocation import latin_hypercube_points, uniform_points` and `from .functions.optimizers import adam_with_decay, run_lbfgs`.

Replace `make_grid`:

```python
def make_grid(cfg: Config, device: torch.device) -> Tensor:
    t = (uniform_points(cfg.t_span, cfg.n_collocation) if cfg.collocation == "uniform"
         else latin_hypercube_points(cfg.t_span, cfg.n_collocation, cfg.seed))
    return torch.as_tensor(t, dtype=cfg.torch_dtype, device=device)
```

In `train()`: `model = PINN(cfg).to(device=device, dtype=cfg.torch_dtype)`; `t_ref, ys_ref = reference_trajectory(cfg, n=cfg.n_eval)`.

Replace the two optimiser lines:

```python
    adam, sched = adam_with_decay(model.parameters(), cfg)
```

Replace the whole `if cfg.lbfgs_iters > 0:` block with:

```python
    if cfg.lbfgs_iters > 0:
        def closure() -> Tensor:
            loss = pinn_loss(model, grid.clone().requires_grad_(True))
            history.loss.append(loss.item())
            n = len(history.loss) - history.adam_iters
            if cfg.print_every and n % 10 == 1:
                print(f"[lbfgs] eval {n:>5} | loss {history.loss[-1]:.4e} | "
                      f"{time.perf_counter() - t_start:6.1f}s elapsed", flush=True)
            return loss

        run_lbfgs(model.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
        history.wall_clock_s = time.perf_counter() - t_start
        if cfg.print_every:
            print(f"[lbfgs] done after {len(history.loss) - history.adam_iters} evals | "
                  f"final loss {history.loss[-1]:.4e}", flush=True)
```

In `predict` and `residual_at` use the model's dtype: `dtype=next(model.parameters()).dtype` instead of `torch.float32`. In `evaluate` and `collect_artifacts` use `n=cfg.n_eval` instead of `n=1001`. In `load_run` add `.to(dtype=cfg.torch_dtype)` after `.to(device)`.

In `run_summary` add to the configuration block:

```python
        "dtype": cfg.dtype, "ic_scale": cfg.ic_scale, "collocation": cfg.collocation,
        "n_eval": cfg.n_eval, "lr_decay": cfg.lr_decay if cfg.lr_decay is not None else float("nan"),
        "lr_decay_every": cfg.lr_decay_every,
```

- [ ] **Step 6: Round-trip the new knobs in `config_for`**

In `src/pinn/sweep.py` `config_for`, extend the `replace(...)` call:

```python
        t_span=(float(row["t_start"]), float(row["t_end"])),
        lbfgs_iters=int(row["lbfgs_iters"]), gamma=float(row["gamma"]),
        lr_start=float(row["lr_start"]), lr_end=float(row["lr_end"]),
        dtype=str(row.get("dtype", "float32")), ic_scale=str(row.get("ic_scale", "span")),
        collocation=str(row.get("collocation", "lhs")), n_eval=int(row.get("n_eval", 1001)),
        lr_decay=None if pd.isna(row.get("lr_decay", float("nan"))) else float(row["lr_decay"]),
        lr_decay_every=int(row.get("lr_decay_every", 5000)),
```

- [ ] **Step 7: Run the full suite**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 23 passed.

- [ ] **Step 8: Commit**

```bash
git add -A src/pinn
git commit -m "feat: float64, density, sampler and lr-decay knobs; L-BFGS tolerance by dtype

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 4: `Problem` and the reference solver

**Files:**
- Create: `src/pinn/problems.py`, `src/pinn/functions/reference.py`
- Modify: `src/pinn/config.py`, `src/pinn/pinn.py`, `src/pinn/train.py`, `src/pinn/sweep.py`
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces: `Problem` dataclass; `PROBLEMS` registry; `Config.problem: str`; `Config.spec -> Problem`; `solve_reference(problem, t) -> np.ndarray`; `PINN.rhs` routes through `Problem.rhs`; `loss_terms` adds `gamma * |u(tf) - end_state|^2` when `Problem.end_state` is set.

- [ ] **Step 1: Write the failing tests**

```python
def test_problem_registry_matches_locked_baseline():
    import numpy as np
    from pinn.config import reference_trajectory
    from pinn.functions.reference import solve_reference

    cfg = Config(t_span=(0.0, 2.0))
    t, ys = reference_trajectory(cfg, n=201)
    generic = solve_reference(cfg.spec, t)
    assert cfg.spec.name == "lorenz1960" and cfg.spec.dim == 3
    assert np.abs(generic - ys).max() < 1e-8


def test_end_state_penalty_enters_the_loss():
    import torch
    from dataclasses import replace
    from pinn.pinn import PINN, pinn_loss

    cfg = Config(depth=1, width=8)
    torch.manual_seed(0)
    plain = pinn_loss(PINN(cfg), torch.linspace(0, 1, 8).reshape(-1, 1).requires_grad_(True))
    torch.manual_seed(0)
    bvp = replace(cfg, end_state=(9.0, 9.0, 9.0))
    penalised = pinn_loss(PINN(bvp), torch.linspace(0, 1, 8).reshape(-1, 1).requires_grad_(True))
    assert penalised > plain
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k "problem_registry or end_state_penalty"`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError: 'Config' object has no attribute 'spec'`.

- [ ] **Step 3: Write `src/pinn/problems.py`**

```python
"""The equation being solved, as data: right-hand side, start state, window, reference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from torch import Tensor

from .functions.physics import lorenz1960_rhs


@dataclass(frozen=True)
class Problem:
    name: str
    dim: int
    rhs: Callable[[Tensor], Tensor]
    initial_state: tuple[float, ...]
    t_span: tuple[float, float]
    coefficients: np.ndarray | None = None       # used by the invariant figures
    end_state: tuple[float, ...] | None = None   # future two-point BVP
    reference: Callable[[np.ndarray], np.ndarray] | None = None


def lorenz1960(coefficients: np.ndarray, initial_state, t_span, end_state=None,
               reference=None) -> Problem:
    return Problem("lorenz1960", 3, lambda u: lorenz1960_rhs(u, coefficients),
                   tuple(initial_state), tuple(t_span), coefficients, end_state, reference)


PROBLEMS: dict[str, Callable[..., Problem]] = {"lorenz1960": lorenz1960}
```

- [ ] **Step 4: Write `src/pinn/functions/reference.py`**

```python
import numpy as np
import torch
from scipy.integrate import solve_ivp


def solve_reference(problem, t: np.ndarray) -> np.ndarray:
    """States at the requested (possibly unsorted) times; DOP853 at 1e-10 / 1e-12."""
    t = np.asarray(t, dtype=float).reshape(-1)
    if problem.reference is not None:
        return problem.reference(t)
    order = np.argsort(t)

    def f(_, u):
        return problem.rhs(torch.as_tensor(u, dtype=torch.float64).reshape(1, -1))[0].numpy()

    sol = solve_ivp(f, (problem.t_span[0], float(t[order][-1])), np.asarray(problem.initial_state, float),
                    method="DOP853", rtol=1e-10, atol=1e-12, t_eval=t[order])
    out = np.empty_like(sol.y.T)
    out[order] = sol.y.T
    return out
```

- [ ] **Step 5: Wire `Config.spec`, `PINN.rhs` and the end-state penalty**

In `src/pinn/config.py` add fields `problem: str = "lorenz1960"` and `end_state: tuple[float, float, float] | None = None`, and the property:

```python
    @property
    def spec(self):
        from .problems import PROBLEMS
        return PROBLEMS[self.problem](self.coefficients, self.initial_state, self.t_span,
                                      self.end_state, lambda t: reference_at(self, t))
```

(`reference_at` is defined later in the same module; the lambda resolves it at call time.)

In `src/pinn/pinn.py` `PINN.__init__` add `self.problem = cfg.spec` and `self.end_state = None if cfg.end_state is None else torch.tensor([cfg.end_state])`; replace `rhs`:

```python
    def rhs(self, u: Tensor) -> Tensor:
        return self.problem.rhs(u)
```

In `loss_terms`, after the `ic` branch:

```python
    if model.end_state is not None:
        tf = torch.full((1, 1), model.tf, dtype=t.dtype, device=t.device)
        ic = ic + (model(tf) - model.end_state.to(t)).pow(2).mean()
```

and in `pinn_loss`: `return res + model.gamma * ic if (model.ic == "soft" or model.end_state is not None) else res`.

In `run_summary` add `"problem": cfg.problem,` and in `config_for` add `problem=str(row.get("problem", "lorenz1960")),`.

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 25 passed.

- [ ] **Step 7: Commit**

```bash
git add -A src/pinn
git commit -m "feat: Problem registry and generic DOP853 reference; end-state penalty hook

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 5: Weight and gradient trails

**Files:**
- Modify: `src/pinn/history.py:53-80`, `src/pinn/train.py` (snapshot block, `save_results`)
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces: `TrainHistory.param_epochs: list[int]`, `TrainHistory.param_trail: list[np.ndarray]`, `TrainHistory.grad_trail: list[np.ndarray]`; file `<ckpt_dir>/param_trail.npz` with arrays `epochs`, `params`, `grads`; `flat_grads(model) -> Tensor`.

- [ ] **Step 1: Write the failing test**

```python
def test_trails_follow_the_snapshots(tmp_path):
    import numpy as np
    from pinn.history import flat_params
    from pinn.pinn import PINN
    from pinn.train import set_seed, train

    cfg = Config(depth=1, width=8, epochs=21, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    model, history = train(cfg)
    assert history.param_epochs == [0, 10, 20]
    n = sum(p.numel() for p in model.parameters())
    assert np.stack(history.param_trail).shape == np.stack(history.grad_trail).shape == (3, n)
    set_seed(cfg.seed)
    assert np.allclose(history.param_trail[0], flat_params(PINN(cfg)).cpu().numpy())
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k trails_follow`
Expected: FAIL with `AttributeError: 'TrainHistory' object has no attribute 'param_epochs'`.

- [ ] **Step 3: Implement**

`src/pinn/history.py`, after `flat_params`:

```python
def flat_grads(model: PINN) -> Tensor:
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()]).detach()
```

In `TrainHistory` after the `snapshots` field:

```python
    param_epochs: list[int] = field(default_factory=list, repr=False, compare=False)
    param_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    grad_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
```

`src/pinn/train.py`, in the snapshot block (after `history.snapshots.write(epoch, parts)`):

```python
            history.param_epochs.append(epoch)
            history.param_trail.append(flat_params(model).cpu().numpy())
            history.grad_trail.append(flat_grads(model).cpu().numpy())
```

(import `flat_grads` from `.history`). In `save_results` after `torch.save(...)`:

```python
    if history.param_trail:
        np.savez_compressed(cfg.ckpt_path / "param_trail.npz", epochs=np.asarray(history.param_epochs),
                            params=np.stack(history.param_trail), grads=np.stack(history.grad_trail))
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 26 passed.

- [ ] **Step 5: Commit**

```bash
git add -A src/pinn
git commit -m "feat: record weight and gradient trails at snapshot epochs

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 6: Visualisation suite

**Files:**
- Create: `src/pinn/viz/trajectory3d.py`, `src/pinn/viz/landscape.py`, `src/pinn/viz/compare.py`, `src/pinn/viz/training.py`, `src/pinn/viz/evaluation.py`
- Modify: `src/pinn/viz/__init__.py`, `src/pinn/train.py` (`write_breakdown_figures`), `src/pinn/history.py:277` (`residual_grid` gains `sqrt`), `src/pinn/viz/figures.py:828` (`write_residual_surface_html` gains `title`, `zlabel`)
- Create: `run_viz3d.py`, `run_landscape.py`, `run_compare.py` (thin wrappers)
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Consumes: `param_trail.npz` (Task 5), `breakdown/epoch_*.csv`, `history/*.csv`, `run_summary.csv`.
- Produces: `viz.generate_run_extras(run_dir: Path) -> list[Path]` writing into `<run>/figures/`: `trajectory.html`, `error_surface.html`, `residual_surface_error.html`, `loss_landscape.{png,html}`, `weight_path_pca3.html`, `layer_grad_norms.png`, `loss_phases.png`, `gradient_stability.png`, `gradient_histograms.png`, `ntk_spectrum.png`, `error_vs_t.png`, `error_growth.png`; `viz.compare_runs(run_dirs, out) -> Path`; `viz.precision_floor(run32, run64, out) -> Path`.

- [ ] **Step 1: Bring the three existing scripts in from the experimental branch**

```bash
git show feat/long-horizon-runs:run_viz3d.py     > src/pinn/viz/trajectory3d.py
git show feat/long-horizon-runs:run_landscape.py > src/pinn/viz/landscape.py
git show feat/long-horizon-runs:run_compare.py   > src/pinn/viz/compare.py
```

Then in each file: delete the `sys.path.insert(...)` line and the `from __future__` / `sys` imports that only served it; change `from pinn import figures` to `from . import figures`, `from pinn.history import residual_grid` to `from ..history import residual_grid`, `from pinn.pinn import PINN, loss_terms` to `from ..pinn import PINN, loss_terms`, `from pinn.sweep import config_for` to `from ..sweep import config_for`, `from pinn.train import make_grid` to `from ..train import make_grid`; wrap each `if __name__ == "__main__":` body into a function: `trajectory3d.write_all(run_dir: Path, names: list[str]) -> list[Path]`, `landscape.write_all(run_dir: Path) -> list[Path]`, `compare.write_page(run_dirs: list[Path]) -> Path`, taking the run directory instead of the `RUNS / name` lookup and writing into `run_dir / "figures"`. In `landscape.py` also change the `torch.float32` in `loss_on_plane` to the model's dtype: `dtype=next(model.parameters()).dtype`.

Apply the two small library changes from the experimental branch:

```bash
git show feat/long-horizon-runs:src/pinn/history.py | grep -n "sqrt" | head -3
```

In `src/pinn/history.py` `residual_grid` add parameter `sqrt: bool = True` and replace `magnitude = np.sqrt(frame[column].to_numpy())` with:

```python
        magnitude = frame[column].to_numpy()
        if sqrt:
            magnitude = np.sqrt(magnitude)
```

In `src/pinn/viz/figures.py` `write_residual_surface_html` add parameters `title: str = "Collocation residual surface", zlabel: str = "log10 |r|"` and use them in the `colorbar`, `title` and `zaxis_title`.

- [ ] **Step 2: Write the failing test**

```python
def test_run_extras_are_written(tmp_path):
    from pinn import viz
    from pinn.train import main

    cfg = Config(depth=1, width=8, epochs=21, n_collocation=32, snapshot_every=10,
                 log_every=10, eval_every=10, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    main(cfg)
    written = viz.generate_run_extras(tmp_path)
    names = {p.name for p in written}
    for expected in ("trajectory.html", "loss_landscape.png", "loss_phases.png",
                     "gradient_stability.png", "gradient_histograms.png", "ntk_spectrum.png",
                     "error_vs_t.png", "error_growth.png"):
        assert expected in names, expected
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k run_extras`
Expected: FAIL with `AttributeError: module 'pinn.viz' has no attribute 'generate_run_extras'`.

- [ ] **Step 4: Write `src/pinn/viz/training.py`**

```python
"""Training-phase figures: what the optimiser did, epoch by epoch."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .figures import save_figure


def fig_loss_phases(loss: np.ndarray, adam_iters: int, eps_marks: list[tuple[int, float]] = ()):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(loss, lw=0.6, color="0.6", label="per iteration")
    ax.semilogy(pd.Series(loss).rolling(101, center=True, min_periods=1).median(), "k", lw=1.2,
                label="rolling median")
    ax.axvspan(0, adam_iters, color="tab:blue", alpha=0.06, label="Adam")
    if adam_iters < len(loss):
        ax.axvspan(adam_iters, len(loss), color="tab:orange", alpha=0.1, label="L-BFGS")
    for it, eps in eps_marks:
        ax.axvline(it, color="tab:red", lw=0.8, ls="--")
        ax.annotate(f"eps={eps:g}", (it, loss.max()), fontsize=7, rotation=90, va="top")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("loss by phase: a flat stretch is a plateau; a step down at the L-BFGS band means "
                 "Adam had stalled", fontsize=9)
    return fig


def fig_gradient_stability(epochs: np.ndarray, grads: np.ndarray):
    g = grads / np.maximum(np.linalg.norm(grads, axis=1, keepdims=True), 1e-30)
    cos = (g[1:] * g[:-1]).sum(1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs[1:], cos, "tab:blue", lw=1, label="cos(g_k, g_k-1)")
    ax.set_ylim(-1.05, 1.05); ax.set_ylabel("cosine"); ax.set_xlabel("epoch")
    ax2 = ax.twinx(); ax2.semilogy(epochs, np.linalg.norm(grads, axis=1), "0.4", lw=0.8, label="|g|")
    ax2.set_ylabel("gradient norm")
    ax.set_title("gradient direction stability: near-zero cosine = wandering on a plateau; "
                 "steady positive = descending a valley", fontsize=9)
    fig.legend(loc="lower left", fontsize=8)
    return fig


def fig_gradient_histograms(epochs: np.ndarray, grads: np.ndarray, layer_sizes: list[tuple[str, int]]):
    picks = np.unique(np.linspace(0, len(epochs) - 1, 4).round().astype(int))
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.5), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), picks):
        start = 0
        for name, size in layer_sizes:
            block = grads[k, start:start + size]; start += size
            ax.hist(np.log10(np.abs(block) + 1e-20), bins=40, histtype="step", label=name)
        ax.set_title(f"epoch {epochs[k]}"); ax.set_xlabel("log10 |grad|")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("per-layer gradient magnitude: a layer whose histogram sits far left is not learning",
                 fontsize=9)
    return fig


def ntk_eigenvalues(model, t: torch.Tensor, n_sub: int = 256) -> np.ndarray:
    """Eigenvalues of J J^T for the residual on a subsample of points."""
    from ..pinn import residual_of
    idx = torch.linspace(0, len(t) - 1, min(n_sub, len(t))).long()
    tt = t[idx].clone().requires_grad_(True)
    r = residual_of(model, tt).reshape(-1)
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(len(r)):
        g = torch.autograd.grad(r[i], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                               for gi, p in zip(g, params)]))
    J = torch.stack(rows)
    return torch.linalg.eigvalsh(J @ J.T).flip(0).clamp_min(0).detach().cpu().numpy()


def fig_ntk_spectrum(spectra: dict[int, np.ndarray]):
    fig, ax = plt.subplots(figsize=(7, 4))
    for epoch, ev in spectra.items():
        ax.semilogy(np.maximum(ev, 1e-20), lw=1, label=f"epoch {epoch}")
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("NTK eigenvalue"); ax.legend(fontsize=8)
    ax.set_title("NTK spectrum: a fast-decaying tail means high-frequency residual modes learn slowly",
                 fontsize=9)
    return fig
```

- [ ] **Step 5: Write `src/pinn/viz/evaluation.py`**

```python
"""Evaluation-phase figures: where and how the trained curve is wrong."""
import matplotlib.pyplot as plt
import numpy as np


def fig_error_vs_t(t, pred, ref, residual, joints=()):
    err = np.linalg.norm(pred - ref, axis=1)
    res = np.linalg.norm(residual, axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(t, err, "tab:blue", lw=1, label="|u - ref|")
    ax.semilogy(t, res, "tab:red", lw=0.8, label="|residual|")
    for j in joints:
        ax.axvline(j, color="0.7", lw=0.5)
    ax.set_xlabel("t"); ax.legend(fontsize=8)
    ax.set_title("error vs residual along t: error growing while residual stays flat is the ODE "
                 "amplifying small residuals, not the network failing", fontsize=9)
    return fig


def fig_error_growth(t, pred, ref):
    err = np.linalg.norm(pred - ref, axis=1)
    m = (t > t[0] + 0.05 * (t[-1] - t[0])) & (err > 0)
    p = np.polyfit(np.log(t[m]), np.log(err[m]), 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(t[m], err[m], ".", ms=2, label="|error|")
    ax.loglog(t[m], np.exp(np.polyval(p, np.log(t[m]))), "k--", label=f"fit: t^{p[0]:.2f}")
    ax.set_xlabel("t"); ax.set_ylabel("|u - ref|"); ax.legend(fontsize=8)
    ax.set_title("error growth: exponent near 1-2 is polynomial (periodic system); a curve bending up "
                 "on log-log is exponential (chaos or a broken window)", fontsize=9)
    return fig


def fig_precision_floor(loss32, loss64):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss32, lw=0.7, label="float32")
    ax.semilogy(loss64, lw=0.7, label="float64")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("loss floor by precision: if float32 flattens where float64 keeps falling, "
                 "precision was the ceiling", fontsize=9)
    return fig
```

- [ ] **Step 6: Write `generate_run_extras` in `src/pinn/viz/__init__.py`**

Append:

```python
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import evaluation, landscape, trajectory3d, training


def generate_run_extras(run_dir: str | Path) -> list[Path]:
    from ..sweep import config_for
    from ..train import load_run, make_grid, predict, reference_trajectory, residual_at

    run = Path(run_dir)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_for(run)
    model, history, _ = load_run(cfg)
    written: list[Path] = []

    written += trajectory3d.write_all(run, [run.name])
    if (run / "history" / "param_trail.npz").exists():
        written += landscape.write_all(run)
        trail = np.load(run / "history" / "param_trail.npz")
        written += save_figure(training.fig_gradient_stability(trail["epochs"], trail["grads"]),
                               out, "gradient_stability", ("png",))
        sizes = [(n, p.numel()) for n, p in model.named_parameters() if n.endswith("weight")]
        sizes = [(n, p.numel()) for n, p in model.named_parameters()]
        written += save_figure(training.fig_gradient_histograms(trail["epochs"], trail["grads"], sizes),
                               out, "gradient_histograms", ("png",))
        grid = make_grid(cfg, next(model.parameters()).device)
        spectra = {}
        for k in np.unique(np.linspace(0, len(trail["epochs"]) - 1, 4).round().astype(int)):
            torch.nn.utils.vector_to_parameters(
                torch.as_tensor(trail["params"][k], dtype=next(model.parameters()).dtype), model.parameters())
            spectra[int(trail["epochs"][k])] = training.ntk_eigenvalues(model, grid)
        torch.nn.utils.vector_to_parameters(
            torch.as_tensor(trail["params"][-1], dtype=next(model.parameters()).dtype), model.parameters())
        written += save_figure(training.fig_ntk_spectrum(spectra), out, "ntk_spectrum", ("png",))
        model, history, _ = load_run(cfg)

    loss = np.asarray(history.loss)
    written += save_figure(training.fig_loss_phases(loss, history.adam_iters), out, "loss_phases", ("png",))
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
    pred = predict(model, t)
    written += save_figure(evaluation.fig_error_vs_t(t, pred, ref, residual_at(model, t)),
                           out, "error_vs_t", ("png",))
    written += save_figure(evaluation.fig_error_growth(t, pred, ref), out, "error_growth", ("png",))
    return written


def compare_runs(run_dirs: list[Path]) -> Path:
    from . import compare
    return compare.write_page([Path(r) for r in run_dirs])


def precision_floor(run32: Path, run64: Path, out: Path) -> list[Path]:
    l32 = pd.read_csv(Path(run32) / "history" / "loss_history.csv").loss.to_numpy()
    l64 = pd.read_csv(Path(run64) / "history" / "loss_history.csv").loss.to_numpy()
    return save_figure(evaluation.fig_precision_floor(l32, l64), Path(out), "precision_floor", ("png",))
```

(Remove the duplicated `sizes = ...` line that keeps only weights; keep the one with all parameters, which matches the flattened order of `flat_grads`.) `TrainHistory.adam_iters` must survive `from_saved`: in `history.py` `from_saved`, set `h.adam_iters = int(len(loss_history) - lbfgs_evals)` if not already restored; if the saved CSVs do not carry it, add a `"adam_iters"` column to `diagnostics_frame` (constant) and read it back.

- [ ] **Step 7: Call the extras from `train.py` and add the wrappers**

In `save_results`, after `write_breakdown_figures(cfg)` inside the `if history.snapshots is not None:` block, add:

```python
        figures.generate_run_extras(cfg.results_path)
```

`run_viz3d.py`, `run_landscape.py`, `run_compare.py` at the repo root each become:

```python
"""Regenerate the extra figures for finished runs: python run_viz3d.py runs/4x60 [...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pinn import viz  # noqa: E402

if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(*viz.generate_run_extras(run), sep="\n")
```

(`run_compare.py` calls `viz.compare_runs(sys.argv[1:])` instead and prints the page path.)

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 27 passed.

- [ ] **Step 9: Commit**

```bash
git add -A src/pinn run_viz3d.py run_landscape.py run_compare.py
git commit -m "feat: training- and evaluation-phase figures; 3-D, landscape and comparison views in pinn.viz

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 7: Run A script, checkpointing and the Colab notebook

**Files:**
- Create: `run_batch.py`, `colab.ipynb`
- Modify: `src/pinn/train.py` (checkpoint every 5000 epochs, resume)

**Interfaces:**
- Produces: `python run_batch.py` trains Run A into `runs/4x60_f64_unit/`; `Config.checkpoint_every: int = 0` (0 = off) writes `<ckpt_dir>/adam_<epoch>.pt` and `train()` resumes from the latest one when present.

- [ ] **Step 1: Write the failing test**

```python
def test_adam_checkpoint_resumes(tmp_path):
    from dataclasses import replace
    from pinn.train import train

    cfg = Config(depth=1, width=8, epochs=6, n_collocation=16, checkpoint_every=3,
                 log_every=3, eval_every=3, print_every=0, ckpt_dir=str(tmp_path))
    train(replace(cfg, epochs=3))
    assert (tmp_path / "adam_000003.pt").exists()
    _, history = train(cfg)
    assert history.resumed_from == 3 and len(history.loss) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k checkpoint_resumes`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'checkpoint_every'`.

- [ ] **Step 3: Implement checkpoint and resume**

`Config`: add `checkpoint_every: int = 0`. `TrainHistory`: add `resumed_from: int = 0`.

In `train()` after `adam, sched = adam_with_decay(...)`:

```python
    start = 0
    if cfg.checkpoint_every:
        cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
        saved = sorted(cfg.ckpt_path.glob("adam_*.pt"))
        if saved:
            state = torch.load(saved[-1], map_location=device)
            model.load_state_dict(state["model"]); adam.load_state_dict(state["adam"])
            sched.load_state_dict(state["sched"]); start = state["epoch"]
            history.resumed_from = start
```

change `for epoch in range(cfg.epochs):` to `for epoch in range(start, cfg.epochs):`, and at the end of the loop body:

```python
        if cfg.checkpoint_every and (epoch + 1) % cfg.checkpoint_every == 0:
            torch.save({"model": model.state_dict(), "adam": adam.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch + 1},
                       cfg.ckpt_path / f"adam_{epoch + 1:06d}.pt")
```

- [ ] **Step 4: Write `run_batch.py`**

```python
"""Run A: one 4x60 network over one closed orbit, float64, Adam then L-BFGS.

    python run_batch.py            # writes runs/4x60_f64_unit/
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402

T_LOOP = 13.26446   # first return of the reference orbit to u0 (|u(T) - u0| = 6e-9)

CFG = Config(t_span=(0.0, T_LOOP), points_per_unit=3000, eval_per_unit=1000,
             dtype="float64", ic_scale="unit", epochs=40000, lr_decay=0.9, lr_decay_every=5000,
             lbfgs_iters=5000, checkpoint_every=5000)
SNAPSHOT_EVERY = 500   # 50 (the sweep default) would write ~10 GB of breakdown CSVs at 40k points

if __name__ == "__main__":
    cfg = replace(sweep_config(CFG, CFG.depth, CFG.width), snapshot_every=SNAPSHOT_EVERY)
    row = run_one(cfg, resume=True)
    print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
               "wall_clock_s"]].to_string(index=False))
```

- [ ] **Step 5: Write `colab.ipynb`**

Generate it with:

```python
import json
cells = [
 ("markdown", "# Lorenz-1960 PINN on Colab\nRuntime > Change runtime type > T4 GPU. Set BRANCH, run all."),
 ("code", "BRANCH = 'feat/batch-precision-run'   # or feat/causal-window-training\n"
          "!git clone --branch $BRANCH --single-branch https://github.com/ihmorol/lorenz1960-pinn.git\n"
          "%cd lorenz1960-pinn\n!pip install -q -r requirements.txt"),
 ("code", "import torch; print(torch.__version__, torch.cuda.is_available())"),
 ("code", "!python run_batch.py      # Run A; for branch B use: !python run_causal.py"),
 ("code", "!zip -qr runs.zip runs && ls -la runs.zip\nfrom google.colab import files; files.download('runs.zip')"),
]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU"},
      "cells": [{"cell_type": k, "metadata": {}, "source": s, **({"outputs": [], "execution_count": None} if k == "code" else {})}
                for k, s in cells]}
open("colab.ipynb", "w").write(json.dumps(nb, indent=1))
```

- [ ] **Step 6: Run the full suite and a 30-epoch smoke of `run_batch.py`**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 28 passed.

Run: `python -c "import run_batch as r; from dataclasses import replace; from pinn.sweep import run_one, sweep_config; c=replace(r.CFG, epochs=30, lbfgs_iters=5, points_per_unit=20, eval_per_unit=10, checkpoint_every=0, runs_dir='runs/_smoke'); run_one(sweep_config(c,1,8))"`
Expected: prints a summary row; `runs/_smoke/1x8_f64_unit/figures/` contains the full suite including `loss_phases.png`. Then `rm -rf runs/_smoke`.

- [ ] **Step 7: Commit**

```bash
git add -A src/pinn run_batch.py colab.ipynb
git commit -m "feat: Run A script with Adam checkpoint/resume; Colab notebook

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 8: The Manim film (acts 1 and 2)

**Files:**
- Create: `src/pinn/viz/film/plan.md`, `src/pinn/viz/film/script.py`, `run_film.py`

**Interfaces:**
- Consumes: `<run>/breakdown/epoch_*.csv`, `<run>/history/param_trail.npz`, `<run>/history/loss_history.csv`, `<run>/figures/loss_landscape.npz` (written by `landscape.write_all`: arrays `a`, `b`, `logZ`, `proj`, `log_path`, `epochs`).
- Produces: `python run_film.py runs/<tag> [-qh]` renders `<run>/film/final.mp4`.

- [ ] **Step 1: Make `landscape.write_all` also save its arrays**

In `src/pinn/viz/landscape.py` after computing `logZ`, `proj`, `log_path`: `np.savez(out / "loss_landscape.npz", a=a, b=b, logZ=logZ, proj=proj, log_path=log_path, epochs=epochs)`.

- [ ] **Step 2: Write `src/pinn/viz/film/plan.md`**

```markdown
# Training film

Palette: Classic 3B1B (background #1C1C1C, reference grey 0.4 opacity, network BLUE, error hot colours, gradient arrow YELLOW).
Act 1, "Learning the loop" (30 s): reference orbit drawn once; the network's curve redrawn at each snapshot epoch, points coloured by log10 error; camera orbits slowly; epoch counter top-left. Aha: the curve collapses onto the fixed point, then snaps onto the loop.
Act 2, "Descending the surface" (30 s): PCA loss surface as a 3-D mesh; the Adam path traced epoch by epoch; at each point a yellow arrow shows the projected gradient, length = norm. Aha: the arrow shrinks and turns as the path crosses the ridge.
Act 3 (branch B) "The causal front" (20 s): surface of the temporal weights w(t) vs iteration sweeping forward as eps advances.
```

- [ ] **Step 3: Write `src/pinn/viz/film/script.py`**

```python
"""Render with: manim -ql script.py LearningTheLoop DescendingTheSurface   (env RUN=<run dir>)"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from manim import *

RUN = Path(os.environ.get("RUN", "runs/4x60_f64_unit"))
BG = "#1C1C1C"


def frames(n_frames=40, n_points=300):
    files = sorted((RUN / "breakdown").glob("epoch_*.csv"))
    pick = np.unique(np.linspace(0, len(files) - 1, n_frames).round().astype(int))
    out = []
    for i in pick:
        f = pd.read_csv(files[i]).sort_values("t")
        out.append(f.iloc[:: max(1, len(f) // n_points)])
    return out


class LearningTheLoop(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        fr = frames()
        ref = fr[0][["ref_x", "ref_y", "ref_z"]].to_numpy()
        scale = 2.0 / np.abs(ref).max()
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2]).set_opacity(0.15)
        self.set_camera_orientation(phi=65 * DEGREES, theta=-45 * DEGREES)
        self.add(axes)
        loop = VMobject(color=GREY, stroke_opacity=0.4).set_points_smoothly([*(ref * scale)])
        self.play(Create(loop), run_time=2); self.wait(1)
        counter = Text("epoch 0", font_size=28).to_corner(UL)
        self.add_fixed_in_frame_mobjects(counter)
        curve = None
        self.begin_ambient_camera_rotation(rate=0.05)
        for f in fr:
            pts = f[["x", "y", "z"]].to_numpy() * scale
            err = np.clip((np.log10(f.err_norm.to_numpy() + 1e-9) + 6) / 6, 0, 1)
            new = VGroup(*[Dot3D(p, radius=0.03, color=interpolate_color(BLUE, RED, e)) for p, e in zip(pts, err)])
            new_counter = Text(f"epoch {int(f.epoch.iloc[0])}", font_size=28).to_corner(UL)
            self.add_fixed_in_frame_mobjects(new_counter)
            anims = [Transform(counter, new_counter)]
            anims.append(Transform(curve, new) if curve is not None else FadeIn(new))
            self.play(*anims, run_time=0.5)
            curve = curve if curve is not None else new
        self.wait(2)


class DescendingTheSurface(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        d = np.load(RUN / "figures" / "loss_landscape.npz")
        a, b, Z, proj, lp = d["a"], d["b"], d["logZ"], d["proj"], d["log_path"]
        sx, sy = 4 / (a[-1] - a[0]), 4 / (b[-1] - b[0]); z0, sz = Z.min(), 3 / (Z.max() - Z.min())
        surf = Surface(lambda u, v: np.array([(u - a[0]) * sx - 2, (v - b[0]) * sy - 2,
                                              (np.interp(u, a, Z[np.abs(b - v).argmin()]) - z0) * sz]),
                       u_range=[a[0], a[-1]], v_range=[b[0], b[-1]], resolution=(30, 30),
                       fill_opacity=0.5, checkerboard_colors=[BLUE_E, BLUE_D])
        self.set_camera_orientation(phi=60 * DEGREES, theta=-60 * DEGREES)
        self.play(Create(surf), run_time=2); self.wait(1)
        path = [np.array([(p[0] - a[0]) * sx - 2, (p[1] - b[0]) * sy - 2, (l - z0) * sz]) for p, l in zip(proj, lp)]
        trace = VMobject(color=RED).set_points_as_corners(path[:2])
        dot = Dot3D(path[0], color=YELLOW, radius=0.06)
        self.add(trace, dot)
        for k in range(1, len(path)):
            step = path[k] - path[k - 1]
            arrow = Arrow3D(path[k - 1], path[k - 1] - 3 * step, color=YELLOW, thickness=0.01)
            trace.add_points_as_corners([path[k]])
            self.play(dot.animate.move_to(path[k]), FadeIn(arrow, run_time=0.2), run_time=0.15)
            self.remove(arrow)
        self.wait(2)
```

- [ ] **Step 4: Write `run_film.py`**

```python
"""python run_film.py runs/<tag> [-qh]   -> <run>/film/final.mp4"""
import os
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    run = Path(sys.argv[1]); quality = sys.argv[2] if len(sys.argv) > 2 else "-ql"
    out = run / "film"; out.mkdir(exist_ok=True)
    script = Path("src/pinn/viz/film/script.py")
    scenes = ["LearningTheLoop", "DescendingTheSurface"]
    env = {**os.environ, "RUN": str(run)}
    subprocess.run(["manim", quality, "--media_dir", str(out), str(script), *scenes], env=env, check=True)
    clips = sorted(out.rglob("*.mp4"))
    (out / "concat.txt").write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(out / "concat.txt"),
                    "-c", "copy", str(out / "final.mp4")], check=True)
    print(out / "final.mp4")
```

- [ ] **Step 5: Render on a tiny run**

Run the Task 7 smoke config again (keep `runs/_smoke`), then: `python run_film.py runs/_smoke/1x8_f64_unit`
Expected: `runs/_smoke/1x8_f64_unit/film/final.mp4` exists and plays two scenes. Then `rm -rf runs/_smoke`.

- [ ] **Step 6: Commit**

```bash
git add -A src/pinn/viz/film run_film.py src/pinn/viz/landscape.py
git commit -m "feat: Manim training film, acts 1-2 (trajectory learning, descent with gradient arrows)

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Part 2: branch `feat/causal-window-training` (stacked on Part 1)

```bash
git checkout -b feat/causal-window-training feat/batch-precision-run
```

### Task 9: Causal loss and window splitting

**Files:**
- Modify: `src/pinn/functions/losses.py`
- Create: `src/pinn/functions/windows.py`
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces: `causal_weights(L_sorted: Tensor, eps: float) -> Tensor` (detached); `causal_loss(r: Tensor, t: Tensor, eps: float) -> tuple[Tensor, Tensor]` returning `(loss, weights_in_time_order)`; `split_windows(t_span, n) -> list[tuple[float, float]]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_causal_weights_gate_later_times():
    import torch
    from pinn.functions.losses import causal_loss, causal_weights

    L = torch.tensor([1.0, 1.0, 0.0, 0.0])
    w = causal_weights(L, eps=1.0)
    assert w[0] == 1.0 and torch.all(w[1:] <= w[:-1]) and not w.requires_grad
    assert torch.allclose(causal_weights(torch.zeros(4), eps=100.0), torch.ones(4))
    t = torch.tensor([[0.3], [0.1], [0.2]]); r = torch.ones(3, 3)
    loss, w_sorted = causal_loss(r, t, eps=0.5)
    assert w_sorted.shape == (3,) and loss <= r.pow(2).mean()


def test_split_windows_tile_the_span():
    from pinn.functions.windows import split_windows

    w = split_windows((0.0, 2.0), 4)
    assert w == [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    assert split_windows((0.0, 2.0), 1) == [(0.0, 2.0)]
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k "causal_weights_gate or split_windows"`
Expected: FAIL with `ImportError: cannot import name 'causal_loss'`.

- [ ] **Step 3: Implement**

Append to `src/pinn/functions/losses.py`:

```python
import torch


def causal_weights(L_sorted: Tensor, eps: float) -> Tensor:
    """w_i = exp(-eps * sum_{k<i} L_k); Wang, Sankaran & Perdikaris 2024, eq. 3.5."""
    earlier = torch.cumsum(L_sorted, 0) - L_sorted
    return torch.exp(-eps * earlier).detach()


def causal_loss(r: Tensor, t: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    order = torch.argsort(t.reshape(-1))
    L = r[order].pow(2).mean(dim=1)
    w = causal_weights(L, eps)
    return (w * L).mean(), w
```

`src/pinn/functions/windows.py`:

```python
def split_windows(t_span: tuple[float, float], n: int) -> list[tuple[float, float]]:
    t0, tf = t_span
    edges = [t0 + (tf - t0) * k / n for k in range(n + 1)]
    return [(edges[k], edges[k + 1]) for k in range(n)]
```

- [ ] **Step 4: Run the full suite, commit**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 30 passed.

```bash
git add -A src/pinn
git commit -m "feat: causal residual weighting and window splitting functions

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 10: `WindowedPINN`

**Files:**
- Modify: `src/pinn/pinn.py`, `src/pinn/config.py`
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces: `Config.n_windows: int = 1`; `WindowedPINN(cfg)` with `.windows: nn.ModuleList[PINN]`, `.edges: list[float]`, `.window_of(t) -> LongTensor`, `forward(t)`, and `residual_parts(model, t)` dispatching per window; `set_window_start(k, state)` sets window k's `u0`; `build_model(cfg) -> PINN | WindowedPINN`; tag suffix `_win{n}`.

- [ ] **Step 1: Write the failing test**

```python
def test_windowed_pinn_routes_and_is_continuous():
    import torch
    from pinn.pinn import WindowedPINN, residual_parts

    cfg = Config(t_span=(0.0, 2.0), n_windows=4, depth=1, width=8)
    torch.manual_seed(0)
    m = WindowedPINN(cfg)
    assert len(m.windows) == 4 and cfg.arch == "1x8_win4"
    t = torch.tensor([[0.1], [0.6], [1.2], [1.9]], requires_grad=True)
    assert m.window_of(t).tolist() == [0, 1, 2, 3]
    m.set_window_start(1, m(torch.tensor([[0.5]]))[0])
    joint = torch.tensor([[0.5]])
    assert torch.allclose(m.windows[0](joint), m.windows[1](joint), atol=1e-6)
    parts = residual_parts(m, t)
    assert parts.r.shape == (4, 3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k windowed_pinn`
Expected: FAIL with `ImportError: cannot import name 'WindowedPINN'`.

- [ ] **Step 3: Implement**

`Config`: add `n_windows: int = 1`; in `arch` add `if self.n_windows > 1: tag += f"_win{self.n_windows}"`.

Append to `src/pinn/pinn.py`:

```python
from dataclasses import replace

from .functions.windows import split_windows


class WindowedPINN(nn.Module):
    """One PINN per time window; each starts from the previous window's end state."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        spans = split_windows(cfg.t_span, cfg.n_windows)
        self.edges = [a for a, _ in spans] + [spans[-1][1]]
        self.windows = nn.ModuleList(PINN(replace(cfg, t_span=s, n_windows=1)) for s in spans)
        self.ic, self.gamma, self.end_state = cfg.ic, cfg.gamma, None
        self.t0, self.tf = float(cfg.t_span[0]), float(cfg.t_span[1])
        self.u0 = self.windows[0].u0

    def window_of(self, t: Tensor) -> Tensor:
        inner = torch.as_tensor(self.edges[1:-1], dtype=t.dtype, device=t.device)
        return torch.bucketize(t.reshape(-1), inner, right=True)

    def set_window_start(self, k: int, state: Tensor) -> None:
        self.windows[k].u0.copy_(state.detach().reshape(1, -1))

    def forward(self, t: Tensor) -> Tensor:
        idx = self.window_of(t)
        out = torch.empty(t.shape[0], self.windows[0].u0.shape[1], dtype=t.dtype, device=t.device)
        for k, w in enumerate(self.windows):
            m = idx == k
            if m.any():
                out[m] = w(t[m])
        return out


def _windowed_parts(model: WindowedPINN, t: Tensor) -> ResidualParts:
    idx = model.window_of(t)
    cols = [torch.empty(t.shape[0], 3, dtype=t.dtype, device=t.device) for _ in ResidualParts._fields]
    for k, w in enumerate(model.windows):
        m = idx == k
        if m.any():
            for col, val in zip(cols, residual_parts(w, t[m])):
                col[m] = val
    return ResidualParts(*cols)


def build_model(cfg: Config):
    return WindowedPINN(cfg) if cfg.n_windows > 1 else PINN(cfg)
```

and make `residual_parts` dispatch: at its top add `if isinstance(model, WindowedPINN): return _windowed_parts(model, t)`. Because `t[m]` is a non-leaf slice, `time_derivative` needs a leaf: inside `_windowed_parts` use `tk = t[m].detach().clone().requires_grad_(True)` and call `residual_parts(w, tk)`.

In `train.py`, `model = build_model(cfg).to(...)` and `load_run` uses `build_model`.

- [ ] **Step 4: Run the full suite, commit**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 31 passed.

```bash
git add -A src/pinn
git commit -m "feat: WindowedPINN with per-window residual routing

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 11: The causal windowed training loop

**Files:**
- Modify: `src/pinn/train.py`, `src/pinn/config.py`, `src/pinn/history.py`
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces: `Config.causal_eps_schedule: tuple[float, ...] = ()`, `causal_delta: float = 0.99`, `causal_max_iters: int = 0`; `TrainHistory.eps_marks: list[tuple[int, float]]`, `min_w: list[float]`, `weight_profiles: list[np.ndarray]` (w in time order, per snapshot); `train_window(model, k, grid, cfg, history, t_start) -> None`; `train()` loops windows when `cfg.n_windows > 1`; tag suffix `_causal`.

- [ ] **Step 1: Write the failing test**

```python
def test_eps_advances_only_when_all_weights_exceed_delta():
    from pinn.train import train

    cfg = Config(t_span=(0.0, 0.2), n_windows=2, causal_eps_schedule=(1e-2, 1e-1), causal_delta=0.99,
                 causal_max_iters=15, depth=1, width=8, n_collocation=20, collocation="uniform",
                 lbfgs_iters=2, log_every=5, eval_every=5, print_every=0)
    model, history = train(cfg)
    assert cfg.arch == "1x8_win2_causal"
    assert len(history.eps_marks) == 2 * 2                      # every (window, eps) stage is marked
    for it, _ in history.eps_marks:
        assert history.min_w[it - 1] > 0.99 or it % 15 == 0     # advanced by delta or by the cap
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k eps_advances`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'causal_eps_schedule'`.

- [ ] **Step 3: Implement**

`Config`: add the three fields; in `arch` add `if self.causal_eps_schedule: tag += "_causal"`.

`TrainHistory`: add `eps_marks: list[tuple[int, float]] = field(default_factory=list)`, `min_w: list[float] = field(default_factory=list)`, `weight_profiles: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)`.

In `src/pinn/train.py` replace the Adam loop with a per-window routine and a driver. The single-window, non-causal path must produce exactly what it does today (same schedule, same logging).

```python
def train_window(model, k: int, grid: Tensor, cfg: Config, history: TrainHistory, t_start: float,
                 t_ref, ys_ref) -> None:
    sub = model.windows[k] if isinstance(model, WindowedPINN) else model
    idx = (model.window_of(grid) == k) if isinstance(model, WindowedPINN) else torch.ones(len(grid), dtype=torch.bool)
    pts = grid[idx]
    adam, sched = adam_with_decay(sub.parameters(), cfg)
    stages = list(cfg.causal_eps_schedule) or [None]
    cap = cfg.causal_max_iters if cfg.causal_eps_schedule else cfg.epochs

    for eps in stages:
        for it in range(cap):
            epoch = len(history.loss)
            adam.zero_grad()
            t = pts.clone().requires_grad_(True)
            res, ic, parts = loss_terms(sub, t)
            if eps is not None:
                res, w = causal_loss(parts.r, t, eps)
                history.min_w.append(float(w.min()))
            loss = res + sub.gamma * ic if (cfg.ic == "soft" or sub.end_state is not None) else res
            loss.backward()
            logging = epoch % cfg.log_every == 0
            if history.snapshots is not None and epoch % cfg.snapshot_every == 0:
                history.snapshots.write(epoch, residual_parts(model, grid.clone().requires_grad_(True)))
                history.param_epochs.append(epoch)
                history.param_trail.append(flat_params(model).cpu().numpy())
                history.grad_trail.append(flat_grads(model).cpu().numpy())
                if eps is not None:
                    history.weight_profiles.append(w.cpu().numpy())
            before = flat_params(sub) if logging else None
            lr = adam.param_groups[0]["lr"]
            adam.step(); sched.step()
            history.loss.append(loss.item())
            if logging:
                history.record_step(epoch, model=sub, loss=loss, residual=res, ic=ic, lr=lr, params_before=before)
            if epoch % cfg.eval_every == 0:
                history.record_reference(epoch, float(np.mean((predict(model, t_ref) - ys_ref) ** 2)))
            if cfg.print_every and epoch % cfg.print_every == 0:
                print(f"[adam]  window {k} eps {eps} it {it:>6} | loss {loss.item():.4e} | "
                      f"min_w {history.min_w[-1] if eps is not None else 1.0:.3f} | "
                      f"{time.perf_counter() - t_start:6.1f}s", flush=True)
            if eps is not None and history.min_w[-1] > cfg.causal_delta:
                break
        if eps is not None:
            history.eps_marks.append((len(history.loss), eps))

    if cfg.lbfgs_iters > 0:
        def closure() -> Tensor:
            loss = pinn_loss(sub, pts.clone().requires_grad_(True))
            history.loss.append(loss.item())
            return loss
        run_lbfgs(sub.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)

    if isinstance(model, WindowedPINN) and k + 1 < len(model.windows):
        with torch.no_grad():
            end = torch.tensor([[model.edges[k + 1]]], dtype=grid.dtype, device=grid.device)
            model.set_window_start(k + 1, sub(end)[0])
```

`train()` becomes: setup as today (seed, device, `model = build_model(cfg)`, grid, history, reference, snapshot writer, checkpoint resume for the single-window case), then

```python
    n_windows = len(model.windows) if isinstance(model, WindowedPINN) else 1
    for k in range(n_windows):
        train_window(model, k, grid, cfg, history, t_start, t_ref, ys_ref)
    history.adam_iters = len(history.loss) if cfg.lbfgs_iters == 0 else history.adam_iters
```

Keep the Task 7 checkpointing inside `train_window` for the single-window case (the `adam_*.pt` files), and for windows save `window_{k:02d}.pt` after each window and skip windows whose file exists on resume (load the state dict into `model.windows[k]` and call `set_window_start(k+1, ...)`). `history.adam_iters` is set once, after the last Adam iteration of the last window, before that window's L-BFGS: record it inside `train_window` as `history.adam_iters = len(history.loss)` just before the `if cfg.lbfgs_iters > 0:` block when `k == n_windows - 1` (pass `n_windows` in). Save `weight_profiles` in `param_trail.npz` as `weights` (object array padded with NaN to the longest window) when non-empty.

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 32 passed, including every pre-existing test (single-window path unchanged).

- [ ] **Step 5: Commit**

```bash
git add -A src/pinn
git commit -m "feat: causal windowed training loop with eps annealing and min-w stopping

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 12: Causal figures, joint continuity and film act 3

**Files:**
- Modify: `src/pinn/viz/training.py`, `src/pinn/viz/evaluation.py`, `src/pinn/viz/__init__.py`, `src/pinn/viz/film/script.py`, `run_film.py`
- Test: `src/pinn/test_pinn.py`

**Interfaces:**
- Produces: `training.fig_causal_weights(epochs, t_sorted, W) `, `training.fig_min_w(min_w, delta, eps_marks)`, `training.fig_window_grid(loss, window_marks)`, `evaluation.fig_joint_continuity(model, edges)`; files `causal_weights.png`, `causal_weights.html`, `min_w.png`, `window_grid.png`, `joint_continuity.png` in `<run>/figures/`; film scene `TheCausalFront`.

- [ ] **Step 1: Write the failing test**

```python
def test_causal_extras_are_written(tmp_path):
    from pinn import viz
    from pinn.train import main

    cfg = Config(t_span=(0.0, 0.2), n_windows=2, causal_eps_schedule=(1e-2,), causal_max_iters=12,
                 depth=1, width=8, n_collocation=20, collocation="uniform", snapshot_every=6,
                 log_every=6, eval_every=6, print_every=0,
                 results_dir=str(tmp_path), ckpt_dir=str(tmp_path / "history"))
    main(cfg)
    names = {p.name for p in viz.generate_run_extras(tmp_path)}
    for expected in ("causal_weights.png", "min_w.png", "window_grid.png", "joint_continuity.png"):
        assert expected in names, expected
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings -k causal_extras`
Expected: FAIL on the first missing name.

- [ ] **Step 3: Implement the figures**

Append to `src/pinn/viz/training.py`:

```python
def fig_causal_weights(epochs, t_sorted, W):
    fig, ax = plt.subplots(figsize=(8, 4))
    picks = np.unique(np.linspace(0, len(epochs) - 1, 6).round().astype(int))
    for k in picks:
        ax.plot(t_sorted[: len(W[k])], W[k], lw=1, label=f"epoch {epochs[k]}")
    ax.set_xlabel("t (window-local)"); ax.set_ylabel("temporal weight w"); ax.legend(fontsize=7)
    ax.set_title("causal weights: the front where w drops to 0 is where training currently stops; "
                 "it must reach the right edge before eps advances", fontsize=9)
    return fig


def fig_min_w(min_w, delta, eps_marks):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(min_w, lw=0.8); ax.axhline(delta, color="k", ls="--", lw=0.8, label=f"delta={delta}")
    for it, eps in eps_marks:
        ax.axvline(it, color="tab:red", lw=0.6)
    ax.set_xlabel("causal iteration"); ax.set_ylabel("min w"); ax.legend(fontsize=8)
    ax.set_title("min temporal weight: each crossing of delta ends an eps stage; a stage that never "
                 "crosses hit the iteration cap", fontsize=9)
    return fig


def fig_window_grid(loss, window_marks):
    n = len(window_marks) - 1
    cols = min(n, 5); rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.4 * rows), squeeze=False)
    for k in range(n):
        ax = axes[k // cols][k % cols]
        ax.semilogy(loss[window_marks[k]:window_marks[k + 1]], lw=0.7)
        ax.set_title(f"window {k}", fontsize=8)
    fig.suptitle("loss per window: a window that ends high inherits error into every later window",
                 fontsize=9)
    fig.tight_layout()
    return fig
```

Append to `src/pinn/viz/evaluation.py`:

```python
def fig_joint_continuity(model, edges, h=1e-4):
    import torch
    dtype = next(model.parameters()).dtype
    jumps, slopes = [], []
    for e in edges[1:-1]:
        left = torch.tensor([[e - h]], dtype=dtype); right = torch.tensor([[e + h]], dtype=dtype)
        with torch.no_grad():
            ul, ur = model(left)[0], model(right)[0]
            ul2, ur2 = model(left - h)[0], model(right + h)[0]
        jumps.append(float((ur - ul).norm()))
        slopes.append(float(((ur2 - ur) / h - (ul - ul2) / h).norm()))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.semilogy(edges[1:-1], jumps, "o-", ms=3, label="|u(t+) - u(t-)|")
    ax.semilogy(edges[1:-1], slopes, "s-", ms=3, label="slope mismatch")
    ax.set_xlabel("window joint t"); ax.legend(fontsize=8)
    ax.set_title("hand-off error at each joint: value jumps are the IC copy error; slope jumps show "
                 "the next window disagreeing with the physics at its start", fontsize=9)
    return fig
```

In `generate_run_extras` (viz `__init__.py`), after the trail block, add:

```python
    if history.min_w:
        marks = history.eps_marks
        written += save_figure(training.fig_min_w(np.asarray(history.min_w), cfg.causal_delta, marks),
                               out, "min_w", ("png",))
        window_marks = [0] + [it for it, _ in marks][len(cfg.causal_eps_schedule) - 1::len(cfg.causal_eps_schedule)]
        written += save_figure(training.fig_window_grid(loss, window_marks), out, "window_grid", ("png",))
        if "weights" in trail:
            t_sorted = np.sort(make_grid(cfg, "cpu").numpy().reshape(-1))[: trail["weights"].shape[1]]
            written += save_figure(training.fig_causal_weights(trail["epochs"], t_sorted, trail["weights"]),
                                   out, "causal_weights", ("png",))
    if hasattr(model, "edges"):
        written += save_figure(evaluation.fig_joint_continuity(model, model.edges), out, "joint_continuity", ("png",))
```

`history.min_w` and `eps_marks` must round-trip through `from_saved`: write them as `history/causal.csv` (columns `iteration, min_w`) and `history/eps_marks.csv` in `write_run_report`, and read them back in `from_saved` when present.

- [ ] **Step 4: Film act 3**

Append to `src/pinn/viz/film/script.py`:

```python
class TheCausalFront(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        d = np.load(RUN / "history" / "param_trail.npz")
        W, epochs = d["weights"], d["epochs"]
        W = np.nan_to_num(W, nan=0.0)
        n_t = W.shape[1]
        axes = ThreeDAxes(x_range=[0, 1], y_range=[0, 1], z_range=[0, 1]).set_opacity(0.15)
        self.set_camera_orientation(phi=60 * DEGREES, theta=-50 * DEGREES)
        self.add(axes)
        rows = VGroup()
        for k in np.unique(np.linspace(0, len(epochs) - 1, 30).round().astype(int)):
            pts = [np.array([i / n_t, k / len(epochs), W[k, i]]) for i in range(0, n_t, max(1, n_t // 100))]
            rows.add(VMobject(color=YELLOW).set_points_as_corners(pts))
            self.play(Create(rows[-1]), run_time=0.3)
        self.wait(2)
```

and add `"TheCausalFront"` to `scenes` in `run_film.py` when `(run / "history" / "param_trail.npz")` contains `weights`.

- [ ] **Step 5: Run the suite, commit**

Run: `python -m pytest src/pinn/test_pinn.py -q -p no:warnings`
Expected: 33 passed.

```bash
git add -A src/pinn run_film.py
git commit -m "feat: causal weight, min-w, per-window and joint-continuity figures; film act 3

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 13: Run B script

**Files:**
- Create: `run_causal.py`
- Modify: `colab.ipynb` (comment already points at it)

- [ ] **Step 1: Write `run_causal.py`**

```python
"""Run B: causal windowed training (Wang, Sankaran & Perdikaris 2024, Alg. 1 + App. E).

    python run_causal.py           # writes runs/4x60_f64_unit_win27_causal/
"""
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config  # noqa: E402
from pinn.sweep import run_one, sweep_config  # noqa: E402

T_LOOP = 13.26446

CFG = Config(t_span=(0.0, T_LOOP), points_per_unit=3000, eval_per_unit=1000,
             dtype="float64", ic_scale="unit", collocation="uniform", n_windows=27,
             causal_eps_schedule=(1e-2, 1e-1, 1.0, 10.0, 100.0), causal_delta=0.99,
             causal_max_iters=20000, lr_decay=0.9, lr_decay_every=5000, lbfgs_iters=5000)
SNAPSHOT_EVERY = 500

if __name__ == "__main__":
    cfg = replace(sweep_config(CFG, CFG.depth, CFG.width), snapshot_every=SNAPSHOT_EVERY)
    row = run_one(cfg, resume=True)
    print(row[["arch", "rmse_combined_l2", "max_abs_error_combined_l2", "final_loss",
               "wall_clock_s"]].to_string(index=False))
```

- [ ] **Step 2: Smoke it on a tiny config**

Run: `python -c "import run_causal as r; from dataclasses import replace; from pinn.sweep import run_one, sweep_config; c=replace(r.CFG, n_windows=3, causal_max_iters=20, lbfgs_iters=3, points_per_unit=30, eval_per_unit=10, runs_dir='runs/_smoke'); run_one(sweep_config(c,1,8))"`
Expected: summary row printed; `runs/_smoke/1x8_f64_unit_win3_causal/figures/` contains `causal_weights.png`, `min_w.png`, `window_grid.png`, `joint_continuity.png` plus the whole standard suite. Then `rm -rf runs/_smoke`.

- [ ] **Step 3: Commit**

```bash
git add run_causal.py
git commit -m "feat: Run B script, causal windowed training over one orbit

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

### Task 14: Execute the runs on Colab and compare

- [ ] **Step 1: Push both branches**

```bash
git push -u origin feat/batch-precision-run feat/causal-window-training
```

- [ ] **Step 2: Run A on Colab** with `colab.ipynb`, `BRANCH='feat/batch-precision-run'`. Download `runs.zip`, unzip into the local repo's `runs/`.

- [ ] **Step 3: Run B on Colab** with `BRANCH='feat/causal-window-training'` and the fourth cell changed to `!python run_causal.py`. Download and unzip.

- [ ] **Step 4: Comparison page, precision panel, films**

```bash
python run_compare.py runs/4x60_f64_unit runs/4x60_f64_unit_win27_causal
python -c "import sys; sys.path.insert(0,'src'); from pinn import viz; viz.precision_floor('runs_session_t10/4x60', 'runs/4x60_f64_unit', 'runs/figures')"
python run_film.py runs/4x60_f64_unit -qh
python run_film.py runs/4x60_f64_unit_win27_causal -qh
```

- [ ] **Step 5: Record the result** in `docs/superpowers/specs/2026-09-21-long-horizon-precision-and-causal-design.md` under a new `## 9. Outcome` section: RMSE, max error, final loss, wall clock for both runs, and whether the 1e-6 target was met.

---

## Self-review

**Spec coverage.** §2 runs: Tasks 7 and 13. §3 layout: Tasks 1, 2, 6 (viz), 2/3/4/9 (functions). §4.1 knobs: Tasks 3, 4, 7, 10, 11. §4.2 Problem: Task 4. §4.3 model: Tasks 2, 10. §4.4 loop: Task 11. §4.5 precision: Task 3. §5 figures 1-19: 1 (loss_phases, T6), 2 (residual surface HTML exists + error surface, T6), 3 (T12), 4 (T12), 5 (T6), 6 (T6 layer_grad_norms via landscape.write_all), 7 (T6), 8-11 (existing suite), 9 joints (T6/T12), 12 (T6), 13 (T6 precision_floor), 14 (T12), 15 (T6 compare), 16-18 (T6), 19 (T8, T12). §6 tests: each task carries its test; the run-of-record assertion is in T3. §7 compute: T7 checkpointing, colab.ipynb, T14.

**Placeholders.** None: every step shows code or an exact command. The one judgement call left to the implementer is the `from_saved` round trip of `adam_iters`, `min_w` and `eps_marks` (T6 step 6, T12 step 3), spelled out as CSV columns.

**Type consistency.** `residual_of` (T2) is what T6 `ntk_eigenvalues` imports. `causal_loss` returns `(loss, w)` in T9 and is consumed that way in T11. `WindowedPINN.edges`, `.windows`, `window_of`, `set_window_start` (T10) are used in T11 and T12. `flat_grads` (T5) is used in T11. `landscape.write_all` writes `loss_landscape.npz` (T8 step 1) which the film reads. `param_trail.npz` keys: `epochs`, `params`, `grads` (T5) plus `weights` (T11) read by T12.
