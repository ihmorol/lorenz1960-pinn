# Input and output of the PINN — one page, code level

**Scope:** only two things — how the 3000 collocation times enter the network,
and how its output comes out and is turned into the loss. Batched
single-domain mode (`Config()` defaults). Paths are under `src/pinn/`; line
numbers refer to branch `feat/sequential-rollout` (commit `b9279ca`).

## 1. The input: 3000 times, one column

**Where they come from — `train.py:27-32`**

```python
sample = qmc.LatinHypercube(d=1, seed=cfg.seed).random(cfg.n_collocation)   # (3000, 1) in [0, 1)
t = t0 + (tf - t0) * sample                                                  # stretch onto [t0, tf] = [0, 1]
return torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1) # shape (3000, 1)
```

- `n_collocation = 3000` and `t_span = (0.0, 1.0)` are plain config values
  (`config.py:45`, `config.py:30`).
- Latin-hypercube with `d=1` means: cut [0, 1] into 3000 equal bins and draw
  one random time inside each bin. Every bin gets exactly one point, so the
  coverage is even (mean gap 1/3000 ≈ 3.3e-4) but not a regular grid. The
  `seed` makes the same 3000 times every run.
- The tensor is **one column**: 3000 rows, 1 feature. Each row is a single
  number — a time. Nothing else goes in: no positions, no data, no labels.

**Built once, reused every epoch — `train.py:41`, `train.py:75`**

```python
grid = make_grid(cfg, device)                                 # line 41: created once
res, ic, parts = loss_terms(model, grid.clone().requires_grad_(True))   # line 75: every epoch
```

The same 3000 times are fed in all 20,000 epochs (no resampling, no
mini-batches). `.clone().requires_grad_(True)` marks the time column as
something PyTorch must differentiate *with respect to* — needed in step 3.

## 2. Through the network: `(3000, 1) → (3000, 3)`

**The layers — `pinn.py:33-37`**

```python
layers = [nn.Linear(1, cfg.width), act()]            # 1 → 60, tanh
for _ in range(cfg.depth - 1):                       # 3 more times:
    layers += [nn.Linear(cfg.width, cfg.width), act()]  # 60 → 60, tanh
layers.append(nn.Linear(cfg.width, 3))               # 60 → 3, no activation
self.net = nn.Sequential(*layers)
```

`depth = 4`, `width = 60`, `activation = "tanh"` (`config.py:32-34`). The
network is applied to all 3000 rows at once — `self.net(t)` (`pinn.py:130`) is
one matrix pass: `(3000,1) → (3000,60) → (3000,60) → (3000,60) → (3000,60) →
(3000,3)`. Row *i* of the output depends only on row *i* of the input; the
rows never mix. That is what "batched" means here.

**The raw output is `n`, shape `(3000, 3)`** — three numbers per time, but
they are *not yet* the answer. They are the network's free guess.

## 3. The output, step by step: from `n` to the loss

All of this is `residual_parts` (`pinn.py:128-144`) and `loss_terms`
(`pinn.py:151-164`); each intermediate is kept in a named tuple
(`pinn.py:14-26`) so it can be written to disk.

**Step 1 — put the start point in: `u = u0 + t · n` — `pinn.py:50-55`**

```python
g = (t - self.t0) / (self.tf - self.t0)   # = t, because t0 = 0, tf = 1
return self.u0 + g * n                     # (3000, 3)
```

`u0 = (0.5, 0.75, 1.0)` is a fixed buffer (`config.py:29`, `pinn.py:46`). The
state the network *claims* at time t is `u(t) = u0 + t·N(t)`. At t = 0 the
second term is zero, so `u(0) = u0` exactly for any weights — the initial
condition is built in, not learned. `u` is the trajectory; this is what
`predict()` returns after training (`train.py:138-142`).

**Step 2 — differentiate the claimed trajectory: `du/dt` — `pinn.py:141-142`**

```python
cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(3)]
dudt = torch.stack(cols, dim=1)            # (3000, 3)
```

For each of x, y, z, PyTorch computes the exact derivative of `u` with respect
to the input time column, at all 3000 rows in one call (the `.sum()` is the
trick that makes one call return all 3000 slopes; since rows don't mix, the
derivative of the sum at row *i* is the derivative of row *i*).
`create_graph=True` keeps this derivative differentiable, so the optimiser
can later change the weights *through* it. This is why step 1 needed
`requires_grad_(True)` on `t`.

**Step 3 — what the equation says the slope should be: `f(u)` — `pinn.py:117-121`**

```python
x, y, z = u[:, 0], u[:, 1], u[:, 2]
return torch.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], dim=1)   # (3000, 3)
```

The Lorenz-1960 right-hand side, `(−0.10·yz, 1.60·xz, −0.75·xy)`, evaluated
at the network's own claimed states. Coefficients come from
`config.coefficients` (`config.py:81-82`), stored as a buffer (`pinn.py:47`).

**Step 4 — the residual: `r = du/dt − f(u)` — `pinn.py:143-144`**

```python
f = ode_rhs(u, model.coeffs)
return ResidualParts(n=n, u=u, dudt=dudt, f=f, r=dudt - f)   # r: (3000, 3)
```

Row *i* of `r` is "how much the network's curve breaks the equation at
time t_i", one number each for x, y, z. A perfect solution has `r = 0`
everywhere.

**Step 5 — one number to minimise: `loss = mean(r²)` — `pinn.py:158`**

```python
res = parts.r.pow(2).mean()   # average over all 3000 × 3 = 9000 entries
```

With `ic = "hard"` there is no initial-condition term (`pinn.py:162-163`,
`ic = 0`), so the loss is purely physics: `pinn_loss = res` (`pinn.py:169`).

**Step 6 — the update — `train.py:74-77`, `train.py:87-88`**

```python
adam.zero_grad()
res, ic, parts = loss_terms(model, grid.clone().requires_grad_(True))
loss = res
loss.backward()     # gradient of the mean residual w.r.t. all 11,283 weights
adam.step()         # one Adam step
sched.step()        # learning rate 1e-3 → 1e-4, linearly over 20,000 epochs
```

One forward pass of all 3000 rows, one backward pass, one weight update —
that is an epoch. Repeat 20,000 times (`config.py:39-42`).

## The whole flow in one line

```
t (3000,1) ──net──▶ n (3000,3) ──u0 + t·n──▶ u (3000,3) ──autograd──▶ du/dt (3000,3)
                                              │                              │
                                              └──── f(u) (3000,3) ◀──────────┘
                                                         r = du/dt − f(u)  →  loss = mean(r²)
```

**Why the batch is exact:** because rows never mix in any step, checking
3000 points at once gives the *same* gradient as checking them one at a time
and adding up — proven by `test_pinn.py:480-525`
(`test_batched_and_per_point_accumulation_train_identically`, agreement to
32-bit round-off).
