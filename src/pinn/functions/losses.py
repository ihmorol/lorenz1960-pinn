from torch import Tensor


def mean_squared_residual(r: Tensor) -> Tensor:
    return r.pow(2).mean()
