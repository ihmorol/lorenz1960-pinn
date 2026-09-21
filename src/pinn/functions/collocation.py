import numpy as np
from scipy.stats import qmc


def latin_hypercube_points(t_span: tuple[float, float], n: int, seed: int) -> np.ndarray:
    t0, tf = t_span
    sample = qmc.LatinHypercube(d=1, seed=seed).random(n)
    return (t0 + (tf - t0) * sample).reshape(-1, 1)


def uniform_points(t_span: tuple[float, float], n: int) -> np.ndarray:
    return np.linspace(t_span[0], t_span[1], n).reshape(-1, 1)
