import numpy as np
import torch
from scipy.stats import qmc


def latin_hypercube_points(t_span: tuple[float, float], n: int, seed: int) -> np.ndarray:
    t0, tf = t_span
    sample = qmc.LatinHypercube(d=1, seed=seed).random(n)
    return (t0 + (tf - t0) * sample).reshape(-1, 1)


def uniform_points(t_span: tuple[float, float], n: int) -> np.ndarray:
    return np.linspace(t_span[0], t_span[1], n).reshape(-1, 1)


def make_grid(cfg, device: torch.device) -> torch.Tensor:
    """The training collocation tensor for ``cfg``: LHS by default, uniform on request."""
    t = (uniform_points(cfg.t_span, cfg.n_collocation) if cfg.collocation == "uniform"
         else latin_hypercube_points(cfg.t_span, cfg.n_collocation, cfg.seed))
    return torch.as_tensor(t, dtype=cfg.torch_dtype, device=device)
