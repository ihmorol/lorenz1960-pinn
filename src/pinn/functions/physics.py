import torch
from torch import Tensor


def system_rhs(u: Tensor, system, params) -> Tensor:
    return torch.stack(system.rhs(u.unbind(1), *params), dim=1)


def residual(u: Tensor, dudt: Tensor, system, params) -> Tensor:
    return dudt - system_rhs(u, system, params)
