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
