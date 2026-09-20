"""Gradient descent in 3-D: the loss surface around the optimiser's own path.

    python run_landscape.py runs/4x60

The path Adam took lives in an 11 283-dimensional weight space. Following
Li et al. 2018 ("Visualizing the loss landscape of neural nets", NeurIPS) we take
the recorded weight trail (history/param_trail.npz, one row per snapshot epoch),
find the two directions along which it moved most (PCA), and evaluate the true
training loss on a grid in that plane. The trail is then drawn on top of the
surface, so what you see is the actual descent, not a cartoon of one.

Writes into <run>/figures/landscape/ :

    loss_landscape.html     rotatable surface + optimiser path, epoch on hover
    loss_landscape.png      the same, static, plus a top-down contour view
    weight_path_pca3.html   the path in the first three PCA directions, coloured
                            by epoch, sized by gradient norm
    layer_grad_norms.png    per-layer gradient norm vs epoch (which layers learn when)

Caveat printed with the figures: the plane explains only part of the trail's
variance; the surface is a 2-D slice of a much larger space.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.pinn import PINN, loss_terms  # noqa: E402
from pinn.sweep import config_for  # noqa: E402
from pinn.train import make_grid  # noqa: E402

GRID = 41          # loss evaluations per axis
MARGIN = 0.25      # extend the plane this far beyond the path's extent


def loss_on_plane(model: PINN, grid: torch.Tensor, centre: np.ndarray, d1: np.ndarray, d2: np.ndarray,
                  a: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = np.empty((len(b), len(a)))
    for i, bb in enumerate(b):
        for j, aa in enumerate(a):
            theta = torch.as_tensor(centre + aa * d1 + bb * d2, dtype=torch.float32)
            torch.nn.utils.vector_to_parameters(theta, model.parameters())
            res, ic, _ = loss_terms(model, grid.clone().requires_grad_(True))
            z[i, j] = (res + model.gamma * ic if model.ic == "soft" else res).item()
    return z


if __name__ == "__main__":
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/4x60")
    out = run / "figures" / "landscape"
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_for(run)
    trail = np.load(run / "history" / "param_trail.npz")
    epochs, P = trail["epochs"], trail["params"].astype(np.float64)
    losses = pd.read_csv(run / "history" / "loss_history.csv").loss.to_numpy()[epochs]
    diag = pd.read_csv(run / "history" / "training_diagnostics.csv")

    # PCA of the trail about its end point: the directions Adam actually moved along.
    centre = P[-1]
    X = P - centre
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S**2 / (S**2).sum()
    d1, d2, d3 = Vt[0], Vt[1], Vt[2]
    proj = X @ Vt[:3].T                       # (n_snapshots, 3)
    print(f"[pca] plane explains {100 * var[:2].sum():.1f}% of the path's variance "
          f"(PC1 {100 * var[0]:.1f}%, PC2 {100 * var[1]:.1f}%, PC3 {100 * var[2]:.1f}%)")

    span = proj[:, :2].max(0) - proj[:, :2].min(0)
    lo, hi = proj[:, :2].min(0) - MARGIN * span, proj[:, :2].max(0) + MARGIN * span
    a, b = np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID)

    torch.manual_seed(cfg.seed)
    model = PINN(cfg)
    grid = make_grid(cfg, torch.device("cpu"))
    print(f"[loss] evaluating {GRID * GRID} points on the plane ...", flush=True)
    Z = loss_on_plane(model, grid, centre, d1, d2, a, b)
    logZ = np.log10(Z)
    log_path = np.log10(losses)

    # ---- rotatable surface + path -------------------------------------------
    import plotly.graph_objects as go
    hover = [f"epoch {e}<br>loss {l:.2e}" for e, l in zip(epochs, losses)]
    fig = go.Figure([
        go.Surface(x=a, y=b, z=logZ, colorscale="Viridis", opacity=0.85,
                   colorbar={"title": "log10 loss"}, name="loss surface"),
        go.Scatter3d(x=proj[:, 0], y=proj[:, 1], z=log_path, mode="lines+markers",
                     line={"color": "red", "width": 4},
                     marker={"size": 3, "color": epochs, "colorscale": "Hot", "showscale": False},
                     text=hover, hoverinfo="text", name="Adam path"),
        go.Scatter3d(x=[proj[0, 0]], y=[proj[0, 1]], z=[log_path[0]], mode="markers",
                     marker={"size": 7, "color": "white", "line": {"color": "black", "width": 2}}, name="start"),
        go.Scatter3d(x=[0], y=[0], z=[log_path[-1]], mode="markers",
                     marker={"size": 7, "color": "black"}, name="end"),
    ])
    fig.update_layout(
        title=f"Loss surface in the plane of the optimiser's path — {cfg.label}<br>"
              f"<sup>plane = top-2 PCA directions of the weight trail, "
              f"{100 * var[:2].sum():.0f}% of path variance; red = actual Adam path</sup>",
        scene={"xaxis_title": "PC1 (weight units)", "yaxis_title": "PC2", "zaxis_title": "log10 loss"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0})
    fig.write_html(str(out / "loss_landscape.html"), include_plotlyjs="cdn")

    # ---- static version ------------------------------------------------------
    A, B = np.meshgrid(a, b)
    fig2 = plt.figure(figsize=(14, 5.5))
    ax = fig2.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(A, B, logZ, cmap="viridis", alpha=0.75, linewidth=0)
    ax.plot(proj[:, 0], proj[:, 1], log_path, "r.-", lw=1.5, ms=3, label="Adam path")
    ax.scatter(proj[0, 0], proj[0, 1], log_path[0], c="w", edgecolors="k", s=60, label="start", zorder=5)
    ax.scatter(0, 0, log_path[-1], c="k", s=60, label="end", zorder=5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("log10 loss"); ax.legend(fontsize=8)
    ax.set_title("loss surface around the path")
    ax = fig2.add_subplot(1, 2, 2)
    cs = ax.contourf(A, B, logZ, levels=30, cmap="viridis")
    plt.colorbar(cs, ax=ax, label="log10 loss")
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=epochs, cmap="hot", s=8, zorder=3)
    ax.plot(proj[:, 0], proj[:, 1], "r-", lw=0.6, zorder=2)
    for k in np.linspace(0, len(epochs) - 1, 6).astype(int):
        ax.annotate(str(epochs[k]), proj[k, :2], fontsize=7, color="w")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("top-down: same surface, path labelled by epoch")
    fig2.suptitle(f"{cfg.label}  |  plane explains {100 * var[:2].sum():.0f}% of the path's variance")
    fig2.tight_layout(); fig2.savefig(out / "loss_landscape.png", dpi=120)

    # ---- path in 3 PCA directions --------------------------------------------
    gn = np.interp(epochs, diag.epoch, diag.grad_norm)
    fig3 = go.Figure(go.Scatter3d(
        x=proj[:, 0], y=proj[:, 1], z=proj[:, 2], mode="lines+markers",
        line={"color": "gray", "width": 2},
        marker={"size": 3 + 12 * gn / gn.max(), "color": epochs, "colorscale": "Viridis",
                "colorbar": {"title": "epoch"}},
        text=[f"epoch {e}<br>loss {l:.2e}<br>|grad| {g:.2e}" for e, l, g in zip(epochs, losses, gn)],
        hoverinfo="text"))
    fig3.update_layout(
        title=f"Weight-space path in its top-3 PCA directions ({100 * var[:3].sum():.0f}% of variance)"
              "<br><sup>marker size = gradient norm; a long straight stretch is a plateau being crossed slowly</sup>",
        scene={"xaxis_title": "PC1", "yaxis_title": "PC2", "zaxis_title": "PC3"},
        margin={"l": 0, "r": 0, "t": 60, "b": 0})
    fig3.write_html(str(out / "weight_path_pca3.html"), include_plotlyjs="cdn")

    # ---- which layers learn when ----------------------------------------------
    layers = [c for c in diag.columns if c.startswith("layer_") and c.endswith("_grad_norm")]
    fig4, ax = plt.subplots(figsize=(9, 4))
    smooth = lambda v: pd.Series(v).rolling(25, center=True, min_periods=1).median()
    for c in layers:
        ax.semilogy(diag.epoch, smooth(diag[c]), lw=1, label=c.replace("_grad_norm", ""))
    ax.semilogy(diag.epoch, diag.grad_norm, color="0.8", lw=0.5, label="total (raw)")
    ax.semilogy(diag.epoch, smooth(diag.grad_norm), "k-", lw=1.4, label="total (rolling median)")
    ax.set_xlabel("epoch"); ax.set_ylabel("gradient norm"); ax.legend(ncol=3, fontsize=8)
    ax.set_title("gradient norm per layer: flat = plateau, rising = the escape, falling = convergence")
    fig4.tight_layout(); fig4.savefig(out / "layer_grad_norms.png", dpi=120)

    for f in sorted(out.iterdir()):
        print(f)
