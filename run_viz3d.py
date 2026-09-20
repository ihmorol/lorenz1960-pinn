"""Rotatable 3-D views of how a run's curve, raw output and error move over training.

    python run_viz3d.py 4x60                 # one run
    python run_viz3d.py 4x60 4x60_seq        # several runs in the same animation

Reads runs/<name>/breakdown/epoch_*.csv (the per-point snapshots) and writes, into
runs/figures/viz3d/ :

    trajectory_<a>_vs_<b>.html   animated (x, y, z) path per run against the reference,
                                 epoch slider, coloured by log10 |error|; the dashed
                                 curve is the raw network output N(t) before the
                                 u0 + t*N(t) wrapper
    error_surface_<name>.html    surface of log10 |error| over (t, epoch), one per run
    residual_surface_<name>.html same for the physics residual |r|
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn import figures  # noqa: E402
from pinn.history import residual_grid  # noqa: E402

RUNS = Path(__file__).resolve().parent / "runs"
OUT = RUNS / "figures" / "viz3d"
N_FRAMES = 60        # epochs shown on the slider
N_POINTS = 300       # collocation points drawn per curve (every 10th of 3000)


def load_frames(run: str) -> list[pd.DataFrame]:
    files = sorted((RUNS / run / "breakdown").glob("epoch_*.csv"))
    pick = np.unique(np.linspace(0, len(files) - 1, N_FRAMES).round().astype(int))
    out = []
    for i in pick:
        f = pd.read_csv(files[i]).sort_values("t")
        out.append(f.iloc[:: max(1, len(f) // N_POINTS)])
    return out


def trajectory_html(runs: dict[str, list[pd.DataFrame]]) -> Path:
    import plotly.graph_objects as go

    ref = next(iter(runs.values()))[0]
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    def traces(k: int) -> list:
        tr = [go.Scatter3d(x=ref.ref_x, y=ref.ref_y, z=ref.ref_z, mode="lines",
                           line={"color": "black", "width": 6}, name="reference (RK45)")]
        for c, (name, frames) in zip(palette, runs.items()):
            f = frames[k]
            err = np.log10(np.maximum(f.err_norm, 1e-9))
            tr.append(go.Scatter3d(
                x=f.x, y=f.y, z=f.z, mode="lines+markers", name=f"{name}  u(t)",
                marker={"size": 3, "color": err, "colorscale": "Inferno_r", "cmin": -6, "cmax": 0,
                        "colorbar": {"title": "log10 |err|"} if c == palette[0] else None},
                line={"color": c, "width": 3},
                text=[f"t={t:.3f}<br>|err|={e:.2e}" for t, e in zip(f.t, f.err_norm)],
                hoverinfo="text+name"))
            tr.append(go.Scatter3d(
                x=f.n_x, y=f.n_y, z=f.n_z, mode="lines", name=f"{name}  raw N(t)",
                line={"color": c, "width": 2, "dash": "dash"}, opacity=0.5))
        return tr

    epochs = [int(f.epoch.iloc[0]) for f in next(iter(runs.values()))]
    fig = go.Figure(
        data=traces(0),
        frames=[go.Frame(data=traces(k), name=str(e)) for k, e in enumerate(epochs)],
    )
    fig.update_layout(
        title="Predicted trajectory over training (drag to rotate, slider = epoch)",
        scene={"xaxis_title": "x", "yaxis_title": "y", "zaxis_title": "z",
               "aspectmode": "cube"},
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        updatemenus=[{"type": "buttons", "showactive": False, "y": 0, "x": 0,
                      "buttons": [{"label": "Play", "method": "animate",
                                   "args": [None, {"frame": {"duration": 150}, "fromcurrent": True}]},
                                  {"label": "Pause", "method": "animate",
                                   "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}]}],
        sliders=[{"currentvalue": {"prefix": "epoch "},
                  "steps": [{"label": str(e), "method": "animate",
                             "args": [[str(e)], {"frame": {"duration": 0}, "mode": "immediate"}]}
                            for e in epochs]}],
    )
    path = OUT / f"trajectory_{'_vs_'.join(runs)}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


if __name__ == "__main__":
    names = sys.argv[1:] or ["4x60"]
    OUT.mkdir(parents=True, exist_ok=True)
    print(trajectory_html({n: load_frames(n) for n in names}))
    for n in names:
        bd = RUNS / n / "breakdown"
        for column, sqrt, tag, zlabel in (("err_norm", False, "error", "log10 |u - ref|"),
                                          ("r_sq", True, "residual", "log10 |r|")):
            e, t, v = residual_grid(bd, column=column, sqrt=sqrt)
            print(figures.write_residual_surface_html(
                e, t, v, OUT / f"{tag}_surface_{n}.html", n,
                title=f"{tag.capitalize()} over (t, epoch)", zlabel=zlabel))
