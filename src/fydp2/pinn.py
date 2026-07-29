"""From-scratch PyTorch PINN for the Lorenz-1960 ODE system."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import Config

_ACT = {"tanh": nn.Tanh, "relu": nn.ReLU, "sigmoid": nn.Sigmoid, "gelu": nn.GELU, "swish": nn.SiLU}


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

    def forward(self, t: Tensor) -> Tensor:
        n = self.net(t)
        if self.ic == "hard":
            g = (t - self.t0) / (self.tf - self.t0)
            return self.u0 + g * n
        return n


def ode_residual(u: Tensor, dudt: Tensor, coeffs) -> Tensor:
    c = torch.as_tensor(coeffs, dtype=u.dtype, device=u.device).reshape(3)
    x, y, z = u[:, 0], u[:, 1], u[:, 2]
    f = torch.stack([c[0] * y * z, c[1] * x * z, c[2] * x * y], dim=1)
    return dudt - f


def residual(model: PINN, t: Tensor) -> Tensor:
    u = model(t)
    cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
    dudt = torch.stack(cols, dim=1)
    return ode_residual(u, dudt, model.coeffs)


def loss_terms(model: PINN, t: Tensor) -> tuple[Tensor, Tensor]:
    """Return (residual loss, initial-condition loss) separately, for diagnostics."""
    res = residual(model, t).pow(2).mean()
    if model.ic == "soft":
        t0 = torch.full((1, 1), model.t0, dtype=t.dtype, device=t.device)
        ic = (model(t0) - model.u0).pow(2).mean()
    else:
        ic = torch.zeros((), dtype=res.dtype, device=res.device)
    return res, ic


def pinn_loss(model: PINN, t: Tensor) -> Tensor:
    res, ic = loss_terms(model, t)
    return res + model.gamma * ic if model.ic == "soft" else res
