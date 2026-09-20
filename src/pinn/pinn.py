"""From-scratch PyTorch PINN for the Lorenz-1960 ODE system."""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .config import Config

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "swish": nn.SiLU}


class ResidualParts(NamedTuple):
    """Every per-collocation-point quantity computed inside one residual evaluation.

    These are the intermediates the training step already builds; exposing them
    costs nothing and lets :mod:`pinn.history` snapshot the full state of the
    collocation batch without a second forward/backward pass.
    """

    n: Tensor       # raw network output N(t),          (Nc, 3)
    u: Tensor       # trial solution u_T(t),            (Nc, 3)
    dudt: Tensor    # du_T/dt: autograd, or the sequential walk's own step (Nc, 3)
    f: Tensor       # physics right-hand side f(u_T),   (Nc, 3)
    r: Tensor       # residual r = du_T/dt - f(u_T),    (Nc, 3)


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

        self.ic = cfg.ic
        self.gamma = cfg.gamma
        self.sequential = bool(cfg.sequential)
        self.causal_eps = float(cfg.causal_eps)   # mutable: annealed during training
        self.causal_chunks = int(cfg.causal_chunks)
        self.register_buffer("u0", torch.tensor([cfg.initial_state], dtype=torch.float32))
        self.register_buffer("coeffs", torch.as_tensor(cfg.coefficients, dtype=torch.float32).reshape(1, 3))
        self.t0, self.tf = float(cfg.t_span[0]), float(cfg.t_span[1])

    def trial(self, t: Tensor, n: Tensor) -> Tensor:
        """Map a raw network output to the trial solution that carries the IC."""
        if self.ic == "hard":
            g = (t - self.t0) / (self.tf - self.t0)
            return self.u0 + g * n
        return n

    def forward(self, t: Tensor) -> Tensor:
        if self.sequential:
            return sequential_rollout(self, t, self.net(t))
        return self.trial(t, self.net(t))


def sequential_rollout(model: PINN, t: Tensor, n: Tensor) -> Tensor:
    """Walk the collocation points in time order, each step anchored to the last.

    The trial solution becomes a discrete integral of the network's own output::

        u(t_start) = u0
        u_i        = u_{i-1} + (t_i - t_{i-1}) * s_i
        s_1        = N(t_1),   s_i = (N(t_{i-1}) + N(t_i)) / 2   for i > 1

    with the ``t_i`` in ascending order: the first interval steps with its own
    endpoint output (the walk has no left neighbour to average with), every later
    interval with the trapezoid rule. Anchoring each point to the walk rather
    than to a global constraint means information enters only through the initial
    condition and travels forward along the chain — and the IC is then exact at
    ``t_start`` by construction, not by penalty. The trapezoid slope makes the
    walk second-order accurate in the step size: fed the exact right-hand side,
    its error against the true solution drops fourfold per grid refinement
    instead of twofold, so the read-out grid no longer caps the accuracy the
    scheme can report.

    Written as a cumulative sum rather than a Python loop over the points. The
    recurrence above has the closed form ``u0 + cumsum(dt * s)``, which computes
    the same quantity in the same order and agrees with the literal loop to
    float32 round-off — but as one kernel instead of ``N_c`` of them. At 3000
    points over 20 000 epochs the literal loop would be 60 million tiny tensor
    ops. ``torch.argsort`` handles the fact that the Latin-hypercube design is
    unsorted: the walk runs on the sorted times and the result is scattered back
    to the caller's ordering.
    """
    return _sequential_walk(model, t, n)[0]


def _sequential_walk(model: PINN, t: Tensor, n: Tensor) -> tuple[Tensor, Tensor]:
    """The walk itself: returns ``(u, du/dt)``, both in the caller's point order.

    ``du/dt`` is the walk's own step — each interval's quadrature slope — so
    ``(u_i - u_{i-1}) / dt_i == du/dt_i`` holds identically and no autograd
    through ``t`` is needed anywhere.
    """
    flat = t.reshape(-1)
    order = torch.argsort(flat)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(flat.numel(), device=flat.device)

    start = torch.as_tensor([model.t0], dtype=flat.dtype, device=flat.device)
    edges = torch.cat([start, flat[order]])
    dt = (edges[1:] - edges[:-1]).reshape(-1, 1)
    n_sorted = n[order]
    trapezoid = 0.5 * (n_sorted[:-1] + n_sorted[1:])
    dudt = torch.cat([n_sorted[:1], trapezoid])   # first interval: no left neighbour
    u = model.u0 + torch.cumsum(dt * dudt, dim=0)
    return u[inv], dudt[inv]


def ode_rhs(u: Tensor, coeffs) -> Tensor:
    """The Lorenz-1960 right-hand side f(u), evaluated pointwise."""
    c = torch.as_tensor(coeffs, dtype=u.dtype, device=u.device).reshape(3)
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    return torch.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], dim=1)


def ode_residual(u: Tensor, dudt: Tensor, coeffs) -> Tensor:
    return dudt - ode_rhs(u, coeffs)


def residual_parts(model: PINN, t: Tensor) -> ResidualParts:
    """One residual evaluation, keeping every intermediate instead of discarding it."""
    n = model.net(t)
    if model.sequential:
        # The walk's own step IS the derivative estimate: consecutive walked points
        # differ by exactly dt * s_i, the interval's quadrature slope (the trapezoid
        # average of the two endpoint outputs; n_1 itself on the first interval), so
        # (u_i - u_{i-1}) / dt == s_i identically and no autograd through t is
        # needed. The residual is then the ODE residual of the discrete trajectory:
        # at zero loss it is the implicit-trapezoid solution on the grid.
        u, dudt = _sequential_walk(model, t, n)
    else:
        u = model.trial(t, n)
        cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
        dudt = torch.stack(cols, dim=1)
    f = ode_rhs(u, model.coeffs)
    return ResidualParts(n=n, u=u, dudt=dudt, f=f, r=dudt - f)


def residual(model: PINN, t: Tensor) -> Tensor:
    return residual_parts(model, t).r


def causal_loss(model: PINN, t: Tensor, r: Tensor) -> Tensor:
    """Causally weighted residual loss (Wang, Sankaran & Perdikaris 2024, eq. 12-13).

    Sort the points in time, cut them into consecutive chunks, take each chunk's
    mean squared residual L_i, and weight it by w_i = exp(-eps * sum_{k<i} L_k),
    with the weights detached so they steer but are not themselves optimised.
    eps is annealed in place on the model: x10 whenever every w_i > 0.99.
    """
    order = torch.argsort(t.reshape(-1))
    chunks = torch.chunk(r[order].pow(2).mean(dim=1), model.causal_chunks)
    L = torch.stack([c.mean() for c in chunks])
    prior = torch.cat([torch.zeros(1, dtype=L.dtype, device=L.device), torch.cumsum(L, 0)[:-1]])
    w = torch.exp(-model.causal_eps * prior).detach()
    if w.min() > 0.99 and model.causal_eps < 1e2:
        model.causal_eps *= 10
    return (w * L).mean()


def loss_terms(model: PINN, t: Tensor) -> tuple[Tensor, Tensor, ResidualParts]:
    """Return (residual loss, initial-condition loss, per-point intermediates).

    The third element is what the training loop hands to the snapshot writer; it
    is the same tensor set the loss was built from, so recording it is free.
    """
    parts = residual_parts(model, t)
    res = causal_loss(model, t, parts.r) if model.causal_eps > 0 else parts.r.pow(2).mean()
    if model.ic == "soft":
        t0 = torch.full((1, 1), model.t0, dtype=t.dtype, device=t.device)
        ic = (model(t0) - model.u0).pow(2).mean()
    else:
        ic = torch.zeros((), dtype=res.dtype, device=res.device)
    return res, ic, parts


def pinn_loss(model: PINN, t: Tensor) -> Tensor:
    res, ic, _ = loss_terms(model, t)
    return res + model.gamma * ic if model.ic == "soft" else res
