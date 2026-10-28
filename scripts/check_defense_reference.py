import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pinn.config import Config, reference_trajectory


def rhs(_, u):
    x, y, z = u
    return np.array([-0.1 * y * z, 1.6 * x * z, -0.75 * x * y])


def rk4(n, end):
    h = end / n
    u = np.array([0.5, 0.75, 1.0])
    for _ in range(n):
        a = rhs(0, u)
        b = rhs(0, u + h * a / 2)
        c = rhs(0, u + h * b / 2)
        d = rhs(0, u + h * c)
        u = u + h * (a + 2 * b + 2 * c + d) / 6
    return u


def main():
    cfg = Config(t_span=(0.0, 13.26446), n_eval=13265)
    t, reference = reference_trajectory(cfg, cfg.n_eval)
    truth = solve_ivp(rhs, (0, 14), cfg.initial_state, method="DOP853",
                      rtol=2.3e-14, atol=1e-15, dense_output=True)
    alternate = solve_ivp(rhs, cfg.t_span, cfg.initial_state, method="Radau",
                          rtol=1e-11, atol=1e-13, t_eval=t)
    assert truth.success and alternate.success
    tighter = truth.sol(t).T
    period = brentq(lambda s: truth.sol(s)[1] - cfg.initial_state[1], 13.2, 13.3, xtol=1e-13)
    invariants = np.column_stack((16 * tighter[:, 0] ** 2 + tighter[:, 1] ** 2,
                                  15 * tighter[:, 0] ** 2 - 2 * tighter[:, 2] ** 2))
    data = {
        "reference_vs_tighter_max_state_norm": float(np.linalg.norm(reference - tighter, axis=1).max()),
        "reference_vs_radau_max_state_norm": float(np.linalg.norm(reference - alternate.y.T, axis=1).max()),
        "rounded_horizon_return_distance": float(np.linalg.norm(truth.sol(cfg.t_span[1]) - cfg.initial_state)),
        "estimated_full_period": float(period),
        "period_fit_return_distance": float(np.linalg.norm(truth.sol(period) - cfg.initial_state)),
        "explicit_invariant_max_absolute_drift": np.max(np.abs(invariants - invariants[0]), axis=0).tolist(),
        "rk4_endpoint_errors": {str(n): float(np.linalg.norm(rk4(n, cfg.t_span[1]) - tighter[-1]))
                               for n in (1000, 2000, 4000)},
    }
    assert data["reference_vs_tighter_max_state_norm"] < 1e-8
    assert data["reference_vs_radau_max_state_norm"] < 1e-8
    assert max(data["explicit_invariant_max_absolute_drift"]) < 1e-10
    assert data["rk4_endpoint_errors"]["4000"] < data["rk4_endpoint_errors"]["1000"] / 100
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
