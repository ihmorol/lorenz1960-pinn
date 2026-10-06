"""Reference-solution sweep over k, l, and initial state, with physics validation.

For every (k, l, IC) cell: generate the DOP853 reference over the training span,
then independently verify the Lorenz-1960 physics on a long horizon -- conserved
quadratics (null-space invariants), RK4-vs-DOP853 cross-solver agreement, and the
dominant period from an FFT. Writes per-cell trajectory npz + 3D figure and one
manifest CSV.

Usage:
    python scripts/reference_sweep.py [output_dir]
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from pinn.config import Config, reference_trajectory  # noqa: E402
from pinn.viz.figures import invariant_series  # noqa: E402

import lorenz1960_baseline as baseline  # noqa: E402

KS = (1.5, 2.0, 3.0)
LS = (0.8, 1.0, 1.5)
ICS = ((0.5, 0.75, 1.0), (-1.0, 0.5, 0.75), (0.25, -1.5, 1.2))
TRAIN_SPAN = (0.0, 1.0)
PHYSICS_SPAN = (0.0, 10.0)
RK4_STEP = 1e-3


def dominant_period(t: np.ndarray, signal: np.ndarray) -> float:
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(t.size, t[1] - t[0])
    return float(1.0 / freqs[1:][np.argmax(spectrum[1:])])


def validate(k: float, l: float, ic: tuple[float, float, float]) -> dict:
    """Long-horizon physics checks, independent of the training-span reference."""
    cfg = Config(k=k, l=l, initial_state=ic, t_span=PHYSICS_SPAN)
    t, ys = reference_trajectory(cfg, n=4001)
    c = baseline.lorenz1960_coefficients(k, l)

    _, drift = invariant_series(ys, c)
    rk4_t, rk4_ys = baseline.rk4_solve(
        lambda tt, y: baseline.lorenz1960_rhs(tt, y, k=k, l=l), PHYSICS_SPAN, ic, RK4_STEP)
    cross = np.abs(rk4_ys - baseline.subsample_uniform_solution(t, ys, rk4_t)).max()

    return {
        "invariant_drift_max": float(drift.max()),
        "rk4_vs_dop853_max": float(cross),
        "period": dominant_period(t, ys[:, 1]),
        "amplitude_max": float(np.abs(ys).max()),
    }


def run(out_dir: Path) -> pd.DataFrame:
    rows = []
    for k, l, ic_idx in product(KS, LS, range(len(ICS))):
        ic = ICS[ic_idx]
        tag = f"k{k:g}_l{l:g}_ic{ic_idx + 1}"
        cell = out_dir / tag
        cell.mkdir(parents=True, exist_ok=True)

        t, ys = reference_trajectory(
            Config(k=k, l=l, initial_state=ic, t_span=TRAIN_SPAN), n=1001)
        np.savez(cell / "reference.npz", t=t, ys=ys, k=k, l=l, initial_state=ic)
        fig, _ = baseline.plot_3d_trajectory(
            ys, title=f"Lorenz 1960 reference, k={k:g}, l={l:g}, IC={ic}")
        fig.savefig(cell / "trajectory_3d.png", dpi=200)
        import matplotlib.pyplot as plt
        plt.close(fig)

        checks = validate(k, l, ic)
        rows.append({"tag": tag, "k": k, "l": l, "initial_state": ic,
                     "coefficients": tuple(baseline.lorenz1960_coefficients(k, l)),
                     **checks})
        print(f"[{tag}] drift={checks['invariant_drift_max']:.1e} "
              f"cross={checks['rk4_vs_dop853_max']:.1e} "
              f"period={checks['period']:.3f}", flush=True)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _REPO / "runs" / "reference_sweep"
    manifest = run(out)
    # thresholds: invariant drift stays under 1e-8 at DOP853 tolerance; RK4 at h=1e-3
    # is itself only ~1e-5-accurate, so cross-solver agreement below 1e-5 is exact enough.
    ok = (manifest.invariant_drift_max < 1e-8).all() and (manifest.rk4_vs_dop853_max < 1e-5).all()
    print(f"\n{'PASS' if ok else 'FAIL'}: {len(manifest)} cells -> {out / 'manifest.csv'}")
