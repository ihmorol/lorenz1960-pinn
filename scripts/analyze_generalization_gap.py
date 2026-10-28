"""Off-grid generalization-gap probe for a trained PINN run.

Tests the classic PINN overfitting failure mode: residual driven to zero at the
collocation points while the network violates the ODE between them. Compares
residual statistics on the exact training grid against a denser off-grid mesh
and the midpoints between neighbouring collocation points, plus the solution
error against an independent high-accuracy integration.

Usage:
    python scripts/analyze_generalization_gap.py [path/to/history/pinn.pt]

Defaults to the candidate-1536pt run-of-record checkpoint.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.config import Config  # noqa: E402
from pinn.functions.collocation import uniform_points  # noqa: E402
from pinn.pinn import build_model, residual_of  # noqa: E402

DEFAULT_CKPT = (_REPO / ".kilo/worktrees/merciful-timbale/runs/causal-window/"
                "candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/history/pinn.pt")

DENSE_MULTIPLIER = 10  # dense mesh has 10x the collocation point count


def load_model(ckpt: Path):
    record = json.loads(ckpt.with_name("config.json").read_text())
    cfg = Config.from_record(record)
    model = build_model(cfg)
    # nn.Linear layers ignore cfg.torch_dtype (training relied on the global default
    # dtype); force float64 before load_state_dict, which otherwise casts silently.
    model.double()
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()
    return cfg, model


def residual_stats(model, t_np: np.ndarray, tag: str) -> dict:
    t = torch.tensor(t_np, dtype=torch.float64).requires_grad_(True)
    r = residual_of(model, t).detach().numpy()
    return {
        "tag": tag,
        "n": len(t_np),
        "rms": float(np.sqrt(np.mean(r ** 2))),
        "max": float(np.max(np.abs(r))),
        "p99": float(np.percentile(np.abs(r), 99)),
    }


def window_midpoints(edges, t_coll: np.ndarray) -> np.ndarray:
    """Midpoints between consecutive collocation points inside each window."""
    mids = []
    for a, b in zip(edges[:-1], edges[1:]):
        pts = t_coll[(t_coll >= a) & (t_coll <= b)].ravel()
        if len(pts) > 1:
            mids.append(0.5 * (pts[:-1] + pts[1:]))
    return np.concatenate(mids).reshape(-1, 1)


def reference_solution(cfg: Config, t_eval: np.ndarray) -> np.ndarray:
    """Independent DOP853 integration with the run's own coefficients."""
    from lorenz1960_baseline import lorenz1960_coefficients

    cx, cy, cz = lorenz1960_coefficients(k=cfg.k, l=cfg.l)

    def rhs(_t, u):
        x, y, z = u
        return [cx * y * z, cy * x * z, cz * x * y]

    sol = solve_ivp(rhs, cfg.t_span, list(cfg.initial_state), t_eval=t_eval,
                    method="DOP853", rtol=1e-12, atol=1e-14)
    assert sol.success, sol.message
    return sol.y.T


def main() -> None:
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    cfg, model = load_model(ckpt)
    t0, tf = cfg.t_span
    print(f"checkpoint : {ckpt}")
    print(f"run        : {cfg.arch} | windows={cfg.n_windows} | coll={cfg.n_collocation} "
          f"| dtype={cfg.dtype} | t in [{t0}, {tf}]")

    t_coll = uniform_points(cfg.t_span, cfg.n_collocation, cfg.n_windows)
    n_dense = DENSE_MULTIPLIER * cfg.n_collocation + 1
    t_dense = np.linspace(t0, tf, n_dense).reshape(-1, 1)
    t_mid = window_midpoints(model.edges, t_coll)
    # Off-grid slice of the dense mesh (exclude points coinciding with the training grid)
    tol = 1e-12
    tc = np.sort(t_coll.ravel())
    j = np.searchsorted(tc, t_dense.ravel()).clip(1, len(tc) - 1)
    nearest = np.minimum(np.abs(t_dense.ravel() - tc[j - 1]), np.abs(t_dense.ravel() - tc[j]))
    mask = nearest > tol
    t_off = t_dense[mask]

    stats = [residual_stats(model, t_coll, "collocation (training grid)"),
             residual_stats(model, t_mid, "midpoints between coll. pts"),
             residual_stats(model, t_off, "dense mesh, off-grid only")]

    print(f"\n{'residual |r|':<32}{'n':>9}{'RMS':>13}{'p99':>13}{'max':>13}")
    for s in stats:
        print(f"{s['tag']:<32}{s['n']:>9}{s['rms']:>13.3e}{s['p99']:>13.3e}{s['max']:>13.3e}")

    rms_c = stats[0]["rms"]
    print(f"\ngap ratios (off-grid / on-grid RMS): "
          f"midpoints {stats[1]['rms'] / rms_c:8.2f}x | dense {stats[2]['rms'] / rms_c:8.2f}x")

    # Solution error vs independent reference, on-grid vs off-grid
    for tag, t_probe in (("at collocation pts", t_coll), ("off-grid (dense)", t_off)):
        u_model = model(torch.tensor(t_probe, dtype=torch.float64)).detach().numpy()
        u_ref = reference_solution(cfg, t_probe.ravel())
        err = np.abs(u_model - u_ref)
        print(f"solution error {tag:<20} max={err.max():.3e}  mean={err.mean():.3e}  "
              f"per-axis max=({err[:, 0].max():.2e}, {err[:, 1].max():.2e}, {err[:, 2].max():.2e})")

    # Where in time does the worst off-grid residual sit?
    r_off = residual_of(model, torch.tensor(t_off, dtype=torch.float64)
                        .requires_grad_(True)).detach().numpy().max(axis=1)
    i = int(np.argmax(r_off))
    print(f"worst off-grid residual {r_off[i]:.3e} at t = {t_off[i, 0]:.6f} "
          f"(window {int(np.searchsorted(model.edges[1:-1], t_off[i, 0]))} of {cfg.n_windows})")


if __name__ == "__main__":
    main()
