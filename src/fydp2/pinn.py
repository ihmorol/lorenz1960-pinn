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
    costs nothing and lets :mod:`fydp2.history` snapshot the full state of the
    collocation batch without a second forward/backward pass.
    """

    n: Tensor       # raw network output N(t),          (Nc, 3)
    u: Tensor       # trial solution u_T(t),            (Nc, 3)
    dudt: Tensor    # autograd time derivative du_T/dt,  (Nc, 3)
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
        return self.trial(t, self.net(t))


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
    u = model.trial(t, n)
    cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
    dudt = torch.stack(cols, dim=1)
    f = ode_rhs(u, model.coeffs)
    return ResidualParts(n=n, u=u, dudt=dudt, f=f, r=dudt - f)


def residual(model: PINN, t: Tensor) -> Tensor:
    return residual_parts(model, t).r


def loss_terms(model: PINN, t: Tensor) -> tuple[Tensor, Tensor, ResidualParts]:
    """Return (residual loss, initial-condition loss, per-point intermediates).

    The third element is what the training loop hands to the snapshot writer; it
    is the same tensor set the loss was built from, so recording it is free.
    """
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
