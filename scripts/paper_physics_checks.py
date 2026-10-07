"""Reference-free physics checks (R7) for the candidate and every finished ablation run.

Per run, on the 13,265-point evaluation grid, prediction vs DOP853 reference:
- invariant drift: max |E(t) - E(0)| / |E(0)| of the conserved E = k^4 x^2 + l^4 y^2
- orbit closure: ||u(T) - u(0)||; t_span is one closed orbit, so the reference is ~0

An FFT period is not reported: over exactly one orbit its lowest bin is the window
length itself, so prediction and reference agree by construction.

Usage:
    python scripts/paper_physics_checks.py [--out runs/paper_ablations]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.config import reference_trajectory  # noqa: E402
from pinn.sweep import config_for  # noqa: E402
from pinn.train import load_run, predict  # noqa: E402

CANDIDATE = _REPO / "runs" / "causal-window" / "candidate-1536pt" / "4x60_f64_unit_win27_causal_warm_f1cbeebd"


def invariant_drift(ys: np.ndarray, k: float, l: float) -> float:
    e = k ** 4 * ys[:, 0] ** 2 + l ** 4 * ys[:, 1] ** 2
    return float(np.abs(e - e[0]).max() / abs(e[0]))


def closure(ys: np.ndarray) -> float:
    return float(np.linalg.norm(ys[-1] - ys[0]))


def check(run_dir: Path) -> dict:
    model, _, cfg = load_run(config_for(run_dir))
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
    pred = predict(model, t)
    return {"run": run_dir.name, "eval_points": len(t),
            "invariant_drift_prediction": invariant_drift(pred, cfg.k, cfg.l),
            "invariant_drift_reference": invariant_drift(ref, cfg.k, cfg.l),
            "closure_prediction": closure(pred), "closure_reference": closure(ref)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/paper_ablations")
    out = Path(ap.parse_args().out)
    out = out if out.is_absolute() else _REPO / out
    runs = [CANDIDATE] + sorted(p.parent for p in out.glob("*_seed*/run_summary.csv"))
    table = pd.DataFrame([check(r) for r in runs if (r / "history" / "pinn.pt").exists()])
    table.to_csv(out / "physics_checks.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
