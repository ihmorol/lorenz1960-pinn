"""Sweep (k, l) combinations through the reference Lorenz-1960 solver and map the closed loop.

No PINN training involved. Every (k, l) pair drives the locked-baseline RHS
(lorenz1960_rhs, the same equations as the SciPy reference) from the same
initial state. The system carries two quadratic invariants (c*x^2 - a*z^2 and
c*y^2 - b*z^2, with a, b, c the coefficient functions of k, l), so every
trajectory lies on a closed curve and each (k, l) pair gives one closed loop
-- with one exception: on the two separatrix curves K1 = c*x0^2 - a*z0^2 = 0
(for k > l) and K2 = c*y0^2 - b*z0^2 = 0 (for k < l) the invariant curve
passes through an equilibrium, the loop degenerates, and its period diverges;
trajectories there creep toward the equilibrium instead of closing within the
horizon. (On the exact diagonal k = l, z freezes and (x, y) rotates with
period 4*pi -- still a closed loop.) The sweep maps all of this out.

The full (k, l) grid is integrated simultaneously with one vectorized RK4
march at the locked baseline step (h = 1e-3), so 20k-30k combinations cost a
single O(steps) pass instead of one solve_ivp call per pair. A handful of
combinations is re-integrated with the SciPy DOP853 reference afterwards and
the results must agree before anything is written.

Per combination the sweep records the oscillation period (zero crossings of
x - mean(x), which works even for loops that never cross x = 0), the loop
amplitudes (max |x|, |y|, |z|), the time-mean state, and a classification
into resolved loop / unresolved (period longer than the horizon) / unbounded.

Example:
    python scripts/k_l_loop_sweep.py                      # 170x170 grid, 15 time units
    python scripts/k_l_loop_sweep.py --n-per-axis 200     # 40,000 combinations
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src" / "baseline"))

from lorenz1960_baseline import lorenz1960_coefficients  # noqa: E402

INITIAL_STATE = (0.5, 0.75, 1.0)   # the locked default initial state
ESCAPE_MAX_Y = 50.0                # max |y| beyond which the trajectory is called unbounded


def coefficients_grid(k_vals: np.ndarray, l_vals: np.ndarray):
    """a, b, c for every (k, l) pair on the grid, shape (n_pairs, 3), plus the (k, l) meshes."""
    kk, ll = np.meshgrid(k_vals, l_vals, indexing="ij")
    k, l = kk.ravel(), ll.ravel()
    a = k * l * (1.0 / (k**2 + l**2) - 1.0 / k**2)
    b = k * l * (1.0 / l**2 - 1.0 / (k**2 + l**2))
    c = 0.5 * k * l * (1.0 / k**2 - 1.0 / l**2)
    return np.stack([a, b, c], axis=1), kk, ll


def _march(coeffs: np.ndarray, horizon: float, step: float, collect: str) -> dict[str, np.ndarray]:
    """RK4-march all (k, l) pairs at once. collect='mean' returns time means only;
    collect='stats' returns amplitudes, mean-relative crossing counts, and the final state."""
    n_pairs = coeffs.shape[0]
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
    x, y, z = (np.full(n_pairs, v) for v in INITIAL_STATE)

    if collect == "mean":
        sum_states = np.stack([x, y, z], axis=1).astype(float)
        n_steps = int(round(horizon / step))
        for step_idx in range(1, n_steps + 1):
            x, y, z = _rk4_step(a, b, c, x, y, z, step)
            sum_states[:, 0] += x; sum_states[:, 1] += y; sum_states[:, 2] += z
        return {"mean_x": sum_states[:, 0] / n_steps,
                "mean_y": sum_states[:, 1] / n_steps,
                "mean_z": sum_states[:, 2] / n_steps}

    max_abs = np.stack([x, y, z], axis=1).astype(float)
    final = max_abs.copy()
    mean_ref = {s: np.asarray(mean_reference["mean_" + s]) for s in "xyz"}
    cross_x = np.zeros(n_pairs); cross_y = np.zeros(n_pairs)
    prev_side_x = np.sign(x - mean_ref["x"]); prev_side_y = np.sign(y - mean_ref["y"])
    n_steps = int(round(horizon / step))
    t_start = time.perf_counter()
    for step_idx in range(1, n_steps + 1):
        x, y, z = _rk4_step(a, b, c, x, y, z, step)
        np.maximum(max_abs[:, 0], np.abs(x), out=max_abs[:, 0])
        np.maximum(max_abs[:, 1], np.abs(y), out=max_abs[:, 1])
        np.maximum(max_abs[:, 2], np.abs(z), out=max_abs[:, 2])
        side_x = np.sign(x - mean_ref["x"]); side_y = np.sign(y - mean_ref["y"])
        cross_x += (side_x != prev_side_x) & (side_x != 0)
        cross_y += (side_y != prev_side_y) & (side_y != 0)
        prev_side_x, prev_side_y = side_x, side_y
        if step_idx % 2500 == 0:
            print(f"  step {step_idx}/{n_steps} ({time.perf_counter() - t_start:.1f} s)", flush=True)
    final[:, 0], final[:, 1], final[:, 2] = x, y, z
    return {"max_x": max_abs[:, 0], "max_y": max_abs[:, 1], "max_z": max_abs[:, 2],
            "x_end": final[:, 0], "y_end": final[:, 1], "z_end": final[:, 2],
            "mean_crossings_x": cross_x, "mean_crossings_y": cross_y}


def _rk4_step(a, b, c, x, y, z, step):
    def deriv(x_, y_, z_):
        return a * y_ * z_, b * x_ * z_, c * x_ * y_

    ax1, ay1, az1 = deriv(x, y, z)
    ax2, ay2, az2 = deriv(x + 0.5 * step * ax1, y + 0.5 * step * ay1, z + 0.5 * step * az1)
    ax3, ay3, az3 = deriv(x + 0.5 * step * ax2, y + 0.5 * step * ay2, z + 0.5 * step * az2)
    ax4, ay4, az4 = deriv(x + step * ax3, y + step * ay3, z + step * az3)
    return (x + step / 6.0 * (ax1 + 2 * ax2 + 2 * ax3 + ax4),
            y + step / 6.0 * (ay1 + 2 * ay2 + 2 * ay3 + ay4),
            z + step / 6.0 * (az1 + 2 * az2 + 2 * az3 + az4))


mean_reference: dict[str, np.ndarray] = {}   # set between the two marches in run_sweep


def run_sweep(coeffs: np.ndarray, horizon: float, step: float) -> dict[str, np.ndarray]:
    """Two passes: pass 1 measures each loop's time-mean, pass 2 counts crossings
    of x - mean(x) (a periodic signal crosses its own mean twice per period, so
    this detects every closed loop, including ones that never cross zero)."""
    global mean_reference
    stats = {}
    print("  pass 1/2: time means", flush=True)
    mean_reference = _march(coeffs, horizon, step, collect="mean")
    stats.update(mean_reference)
    print("  pass 2/2: amplitudes and mean-relative crossings", flush=True)
    stats.update(_march(coeffs, horizon, step, collect="stats"))

    crossings = stats["mean_crossings_x"]
    stats["period_x"] = 2.0 * horizon / np.maximum(crossings, 2.0)   # lower bound on the plateau
    stats["n_cycles_x"] = np.floor(crossings / 2.0)
    stats["unbounded"] = stats["max_y"] > ESCAPE_MAX_Y
    stats["loop_resolved"] = (stats["n_cycles_x"] >= 1) & ~stats["unbounded"]
    return stats


def validate_against_scipy(k_list, l_list, horizon: float) -> float:
    """Re-integrate a few combinations with the SciPy reference and compare final states."""
    from scipy.integrate import solve_ivp

    worst = 0.0
    for k, l in zip(k_list, l_list):
        a, b, c = lorenz1960_coefficients(k, l)
        sol = solve_ivp(lambda _, u: [a * u[1] * u[2], b * u[0] * u[2], c * u[0] * u[1]],
                        (0.0, horizon), INITIAL_STATE, method="DOP853",
                        rtol=1e-10, atol=1e-12, t_eval=[horizon])
        assert sol.success
        coeffs, _, _ = coefficients_grid(np.array([k]), np.array([l]))
        stats = run_sweep(coeffs, horizon, 1e-3)
        rk4_final = np.array([stats["x_end"][0], stats["y_end"][0], stats["z_end"][0]])
        err = np.abs(rk4_final - sol.y[:, -1]).max()
        worst = max(worst, err)
        print(f"  validation k={k:.3g} l={l:.3g}: |RK4 - DOP853| final state = {err:.3e}")
    return worst


def surface_figure(kk, ll, zz, title, zlabel, out_path, log_color=False):
    """Single 3D surface over the (k, l) plane; NaN cells are left as gaps."""
    z = np.ma.masked_invalid(zz.copy())
    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")
    if log_color:
        finite = np.isfinite(zz)
        norm = LogNorm(vmin=np.nanpercentile(zz[finite], 2), vmax=np.nanpercentile(zz[finite], 98))
    else:
        norm = None
    surf = ax.plot_surface(kk, ll, z, cmap="viridis", norm=norm,
                           rstride=2, cstride=2, linewidth=0, antialiased=False)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("log color scale" if log_color else zlabel)
    ax.set_xlabel("k"); ax.set_ylabel("l"); ax.set_zlabel(zlabel)
    fig.suptitle(title, fontsize=13, y=0.96)
    ax.view_init(elev=30, azim=-60)
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def phase_diagram(kk, ll, stats, out_path: Path):
    """2D regime map with the two analytic separatrix curves.

    On K1 = c*x0^2 - a*z0^2 = 0 (k > l side) and K2 = c*y0^2 - b*z0^2 = 0
    (k < l side) the invariant curve through the initial state passes through
    an equilibrium: the closed loop degenerates and its period diverges, so
    those cells do not resolve a full cycle within the horizon.
    """
    unresolved = ~stats["loop_resolved"].reshape(kk.shape)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    cmap = matplotlib.colors.ListedColormap(["#2a9d8f", "#bdbdbd"])
    ax.pcolormesh(kk, ll, unresolved.astype(int), cmap=cmap, vmin=0, vmax=1,
                  shading="nearest", linewidth=0)

    x0, y0, z0 = INITIAL_STATE
    k_plot = np.linspace(kk.min(), kk.max(), 800)
    l_plot = np.linspace(ll.min(), ll.max(), 800)
    KG, LG = np.meshgrid(k_plot, l_plot, indexing="ij")
    a = KG * LG * (1.0 / (KG**2 + LG**2) - 1.0 / KG**2)
    b = KG * LG * (1.0 / LG**2 - 1.0 / (KG**2 + LG**2))
    c = 0.5 * KG * LG * (1.0 / KG**2 - 1.0 / LG**2)
    ax.contour(KG, LG, c * x0**2 - a * z0**2, levels=[0.0], colors="k",
               linewidths=1.6, linestyles="--")
    ax.contour(KG, LG, c * y0**2 - b * z0**2, levels=[0.0], colors="crimson",
               linewidths=1.6, linestyles=":")

    handles = [Line2D([], [], marker="s", ls="", color="#2a9d8f", ms=6,
                      label="closed loop resolved within horizon"),
               Line2D([], [], marker="s", ls="", color="#bdbdbd", ms=6,
                      label="no full cycle within horizon (period diverges)"),
               Line2D([], [], color="k", ls="--",
                      label="separatrix K1 = c·x₀² − a·z₀² = 0"),
               Line2D([], [], color="crimson", ls=":",
                      label="separatrix K2 = c·y₀² − b·z₀² = 0")]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5)
    ax.set_xlabel("k"); ax.set_ylabel("l")
    ax.set_title("Lorenz-1960 trajectory regime per (k, l), same initial state (0.5, 0.75, 1.0)")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def interactive_html(k_vals, l_vals, surfaces: dict[str, np.ndarray], out_path: Path):
    """Plotly 3D surface with a dropdown to switch the plotted loop property."""
    import plotly.graph_objects as go

    kk, ll = np.meshgrid(k_vals, l_vals, indexing="ij")
    names = list(surfaces)
    fig = go.Figure()
    for i, name in enumerate(names):
        fig.add_trace(go.Surface(x=kk, y=ll, z=surfaces[name], colorscale="Viridis",
                                 visible=i == 0, colorbar=dict(title=name), name=name))
    buttons = [dict(label=name, method="update",
                    args=[{"visible": [j == i for j in range(len(names))]},
                          {"title": f"Lorenz-1960 loop over (k, l): {name}"}])
               for i, name in enumerate(names)]
    fig.update_layout(
        title=f"Lorenz-1960 loop over (k, l): {names[0]}",
        scene=dict(xaxis_title="k", yaxis_title="l"),
        updatemenus=[dict(buttons=buttons, x=0.0, xanchor="left", y=1.15)],
        width=1000, height=750,
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def representative_loops_figure(k_list, l_list, horizon: float, out_path: Path):
    """A few closed loops in phase space, one per (k, l), to show 'one pair, one loop'."""
    from scipy.integrate import solve_ivp

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("plasma")
    for i, (k, l) in enumerate(zip(k_list, l_list)):
        a, b, c = lorenz1960_coefficients(k, l)
        t_eval = np.linspace(0.0, horizon, 4000)
        sol = solve_ivp(lambda _, u, a=a, b=b, c=c: [a * u[1] * u[2], b * u[0] * u[2], c * u[0] * u[1]],
                        (0.0, horizon), INITIAL_STATE, method="DOP853",
                        rtol=1e-10, atol=1e-12, t_eval=t_eval)
        color = cmap(i / max(len(k_list) - 1, 1))
        ax.plot(sol.y[0], sol.y[1], sol.y[2], color=color, linewidth=1.6, label=f"k={k:g}, l={l:g}")
        ax.scatter(*sol.y[:, 0], color=color, s=18)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("One (k, l) combination = one closed loop (same initial state)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=200)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k-min", type=float, default=0.5)
    parser.add_argument("--k-max", type=float, default=5.0)
    parser.add_argument("--l-min", type=float, default=0.5)
    parser.add_argument("--l-max", type=float, default=5.0)
    parser.add_argument("--n-per-axis", type=int, default=170,
                        help="grid points per axis; total combinations = n^2")
    parser.add_argument("--horizon", type=float, default=15.0,
                        help="integration horizon in time units per (k, l)")
    parser.add_argument("--step", type=float, default=1e-3, help="RK4 step (baseline lock: 1e-3)")
    parser.add_argument("--out-dir", default="runs/k_l_loop_sweep")
    args = parser.parse_args()

    out_dir = _REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    k_vals = np.linspace(args.k_min, args.k_max, args.n_per_axis)
    l_vals = np.linspace(args.l_min, args.l_max, args.n_per_axis)
    coeffs, kk, ll = coefficients_grid(k_vals, l_vals)
    n_pairs = coeffs.shape[0]
    print(f"Integrating {n_pairs} (k, l) combinations for {args.horizon} time units, "
          f"RK4 step {args.step}, initial state {INITIAL_STATE}", flush=True)

    stats = run_sweep(coeffs, args.horizon, args.step)

    print("Validating the vectorized RK4 against the SciPy DOP853 reference ...", flush=True)
    worst = validate_against_scipy([2.0, 3.0, 0.7, 0.5], [1.0, 2.0, 4.5, 5.0], args.horizon)
    if worst > 1e-5:
        raise RuntimeError(f"vectorized RK4 disagrees with the SciPy reference (max err {worst:.3e}); "
                           "results not written")

    header = ["k", "l"] + list(stats)
    rows = np.column_stack([kk.ravel(), ll.ravel()] + [stats[name] for name in stats])
    csv_path = out_dir / "kl_loop_sweep.csv"
    np.savetxt(csv_path, rows, delimiter=",", header=",".join(header), comments="", fmt="%.6g")
    print(f"wrote {csv_path} ({n_pairs} rows)")

    n_resolved = int(stats["loop_resolved"].sum())
    n_unbounded = int(stats["unbounded"].sum())
    print(f"closed loops resolved within the horizon: {n_resolved}/{n_pairs}; "
          f"unbounded trajectories: {n_unbounded}; "
          f"slower than the horizon: {n_pairs - n_resolved - n_unbounded}")

    surface_figure(kk, ll, np.where(stats["loop_resolved"], stats["period_x"], np.nan).reshape(kk.shape),
                   f"Lorenz-1960 loop period over (k, l), horizon {args.horizon:g} "
                   "(gap = no resolved loop; plateau edge = period approaching the horizon)",
                   "period (time units)", out_dir / "loop_period_3d", log_color=True)
    surface_figure(kk, ll, np.where(stats["unbounded"], np.nan, stats["max_x"]).reshape(kk.shape),
                   "Lorenz-1960 loop amplitude over (k, l)", "max |x|",
                   out_dir / "loop_amplitude_3d")
    phase_diagram(kk, ll, stats, out_dir / "kl_phase_diagram")

    interactive_html(k_vals, l_vals,
                     {name: stats[name].reshape(kk.shape)
                      for name in ("period_x", "max_x", "max_y", "max_z")},
                     out_dir / "kl_loop_sweep_interactive.html")

    representative_loops_figure([2.0, 3.0, 4.0, 2.0, 4.0],
                                [1.0, 1.0, 1.0, 2.0, 2.0],
                                args.horizon, out_dir / "representative_loops_3d")
    print("figures written to", out_dir)


if __name__ == "__main__":
    main()
