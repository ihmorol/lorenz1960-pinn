"""ODE systems the pipeline can solve. Each rhs works on numpy and torch alike."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from lorenz1960_baseline import lorenz1960_coefficients, rk4_solve


@dataclass(frozen=True)
class System:
    rhs: Callable                            # rhs(u, *params) -> components of du/dt
    params: tuple[str, ...]
    invariant_normal: Callable | None = None  # quadratic invariants are orthogonal to this


def lorenz1960(u, k, l):
    x, y, z = u
    a = lorenz1960_coefficients(k, l)
    return a[0] * y * z, a[1] * x * z, a[2] * x * y


def lorenz1963(u, sigma, rho, beta):
    x, y, z = u
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z


SYSTEMS = {
    "lorenz1960": System(lorenz1960, ("k", "l"), lorenz1960_coefficients),
    "lorenz1963": System(lorenz1963, ("sigma", "rho", "beta")),
}


def solve_scipy(problem: str, params, initial_state, t_span, t_eval,
                method: str = "DOP853", rtol: float = 1e-10, atol: float = 1e-12) -> np.ndarray:
    rhs = SYSTEMS[problem].rhs
    sol = solve_ivp(lambda _, u: rhs(u, *params), t_span, np.asarray(initial_state, float),
                    method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"SciPy solver failed: {sol.message}")
    return sol.y.T


def rk4_vs_scipy(problem: str, params, initial_state, t_span, h: float = 1e-3):
    rhs = SYSTEMS[problem].rhs
    t, rk4 = rk4_solve(lambda _, u: np.array(rhs(u, *params)), t_span, initial_state, h)
    return t, rk4, solve_scipy(problem, params, initial_state, t_span, t)
