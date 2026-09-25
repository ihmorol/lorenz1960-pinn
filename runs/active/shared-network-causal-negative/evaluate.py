from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor, nn


RUN_DIR = Path(__file__).resolve().parent
T_START = 0.0
T_END = 13.26446
N_WINDOWS = 27
N_EVAL = 13_265
INITIAL_STATE = (0.5, 0.75, 1.0)


class SharedNetworkCausalPINN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, 60), nn.Tanh()]
        for _ in range(3):
            layers.extend((nn.Linear(60, 60), nn.Tanh()))
        layers.append(nn.Linear(60, 3))
        self.net = nn.Sequential(*layers)
        self.register_buffer("coeffs", torch.empty(1, 3))
        self.register_buffer("edge_starts", torch.empty(N_WINDOWS, 3))
        self.edges = torch.linspace(T_START, T_END, N_WINDOWS + 1, dtype=torch.float64)

    def forward(self, t: Tensor) -> Tensor:
        inner_edges = self.edges[1:-1].to(device=t.device, dtype=t.dtype)
        window_index = torch.bucketize(t.reshape(-1), inner_edges, right=True)
        prediction = torch.empty((t.shape[0], 3), dtype=t.dtype, device=t.device)
        raw = self.net(t)
        for index in range(N_WINDOWS):
            mask = window_index == index
            if mask.any():
                local_t = t[mask] - self.edges[index].to(device=t.device, dtype=t.dtype)
                prediction[mask] = self.edge_starts[index] + local_t * raw[mask]
        return prediction


def load_model(checkpoint: Path) -> SharedNetworkCausalPINN:
    model = SharedNetworkCausalPINN().to(dtype=torch.float64)
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def reference_trajectory(t: np.ndarray) -> np.ndarray:
    coefficients = np.array((-0.1, 1.6, -0.75), dtype=float)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        x, y, z = state
        return np.array(
            (coefficients[0] * y * z, coefficients[1] * x * z, coefficients[2] * x * y)
        )

    solution = solve_ivp(
        rhs,
        (T_START, T_END),
        np.asarray(INITIAL_STATE, dtype=float),
        method="DOP853",
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(f"reference integration failed: {solution.message}")
    return solution.y.T


def evaluate(checkpoint: Path) -> dict[str, float]:
    model = load_model(checkpoint)
    t = np.linspace(T_START, T_END, N_EVAL)
    with torch.no_grad():
        prediction = model(torch.as_tensor(t, dtype=torch.float64).reshape(-1, 1)).numpy()
    error = prediction - reference_trajectory(t)
    error_norm = np.linalg.norm(error, axis=1)
    return {
        "mae_combined_l2": float(error_norm.mean()),
        "rmse_combined_l2": float(np.sqrt(np.mean(error_norm**2))),
        "max_abs_error_combined_l2": float(error_norm.max()),
    }


def recorded_rmse(summary: Path) -> float:
    with summary.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    return float(row["rmse_combined_l2"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reload and evaluate the preserved shared-network causal checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=RUN_DIR / "history" / "pinn.pt")
    parser.add_argument("--summary", type=Path, default=RUN_DIR / "run_summary.csv")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    metrics = evaluate(args.checkpoint)
    expected = recorded_rmse(args.summary)
    difference = abs(metrics["rmse_combined_l2"] - expected)
    for name, value in metrics.items():
        print(f"{name}: {value:.12g}")
    print(f"recorded_rmse_combined_l2: {expected:.12g}")
    print(f"absolute_difference: {difference:.3g}")
    if difference > args.tolerance:
        raise SystemExit(
            f"computed RMSE differs from the recorded run by more than {args.tolerance:g}"
        )


if __name__ == "__main__":
    main()
