"""Trajectory-across-epochs figures: how the whole collocation batch moves in training.

Pure plotting, like :mod:`pinn.figures` — arrays and DataFrames in, ``Figure``
out, no torch. The arrays come from :mod:`pinn.trajectory` and are always in the
problem's own units: nothing here rescales the states, because the point of these
figures is to watch the trajectory change on the scale the physics lives on.

Read them together:

- :func:`fig_trajectory_3d_panels` — the orbit itself at a handful of epochs.
- :func:`fig_trajectory_projections` — every epoch's orbit on one pair of axes.
- :func:`fig_trajectory_time_slices` — the same data as a function of ``t``,
  which is where "the curve bends" is easiest to see.
- :func:`fig_trajectory_extent` — the batch's min/max envelope against the
  reference band, so the contraction is a single readable curve.
- :func:`animate_trajectory` — the first three, as a GIF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from .figures import _positive, state_colors

STATE = ("x", "y", "z")
PLANES = ((0, 1), (0, 2), (1, 2))
EPOCH_CMAP = "viridis"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _shared_limits(pred: np.ndarray, ref: np.ndarray, pad: float = 0.06):
    """Per-axis limits covering every selected frame *and* the reference.

    Panels must not autoscale independently: if each one fits its own data, a
    trajectory that shrank by two orders of magnitude would look identical in
    every panel, which is the opposite of the point.
    """
    lo = np.minimum(pred.reshape(-1, 3).min(axis=0), ref.min(axis=0))
    hi = np.maximum(pred.reshape(-1, 3).max(axis=0), ref.max(axis=0))
    span = np.where(hi > lo, hi - lo, 1.0)
    return lo - pad * span, hi + pad * span


def _epoch_norm(epochs: np.ndarray) -> Normalize:
    return Normalize(vmin=float(epochs.min()), vmax=float(epochs.max()) if len(epochs) > 1 else float(epochs.min()) + 1.0)


def _add_epoch_colorbar(fig: Figure, ax: plt.Axes, epochs: np.ndarray) -> None:
    sm = ScalarMappable(cmap=EPOCH_CMAP, norm=_epoch_norm(epochs))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="epoch", fraction=0.046, pad=0.02)


def _line_collection(ax, xs, ys, values, labels=None, cmap=EPOCH_CMAP, norm=None, lw=1.1, alpha=0.9):
    """One coloured polyline per row of ``xs``/``ys``, as a single collection.

    A collection rather than a ``plot`` call per epoch keeps the PDF to one
    drawing object per panel instead of hundreds.
    """
    norm = norm or Normalize(vmin=float(np.min(values)), vmax=float(np.max(values)))
    segments, shades = [], []
    for k, value in enumerate(values):
        pts = np.column_stack([xs[k], ys[k]]).reshape(-1, 1, 2)
        seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
        segments.append(seg)
        shades.append(np.full(len(seg), float(value)))
    coll = LineCollection(np.concatenate(segments), cmap=cmap, norm=norm,
                          linewidths=lw, alpha=alpha, label=labels)
    coll.set_array(np.concatenate(shades))
    ax.add_collection(coll)
    ax.autoscale_view()
    return coll


def _thin_columns(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def _pick_lines(n_frames: int, max_lines: int, geometric: bool = False) -> np.ndarray:
    """Frame indices to draw, spread over the run.

    ``geometric=True`` spends more picks early, where the trajectory shape
    actually changes; a linear spread puts most of its panels on the flat tail.
    Both always keep the first and last frame.
    """
    if n_frames <= max_lines:
        return np.arange(n_frames)
    if not geometric or n_frames < 4:
        return np.unique(np.linspace(0, n_frames - 1, max_lines).round().astype(int))
    raw = np.geomspace(1.0, float(n_frames), max_lines) - 1.0
    return np.unique(np.concatenate([[0], raw.round().astype(int), [n_frames - 1]]))


# --------------------------------------------------------------------------
# the orbit, epoch by epoch
# --------------------------------------------------------------------------
def fig_trajectory_3d_panels(
    epochs: np.ndarray, pred: np.ndarray, ref: np.ndarray,
    n_panels: int = 6, mode: str = "line", label: str = "",
    rmse: np.ndarray | None = None, picks: np.ndarray | None = None,
) -> Figure:
    """The 3-D phase-space orbit at several epochs, on one common axes box.

    ``pred`` is ``(n_frames, N_c, 3)``; rows must already be sorted by ``t`` so
    that joining consecutive points traces the orbit. ``mode="scatter"`` shows
    the individual collocation points instead of the joined curve.

    Axes are shared across the panels and sized from the frames shown *plus* the
    reference, so the panels are directly comparable and a shrinking trajectory
    looks like it shrinks. Pass ``picks`` to choose the frames yourself — e.g. the
    tail of training, where the box then closes in on the final orbit.
    """
    n_panels = int(np.clip(n_panels, 2, 12))
    if picks is None:
        picks = _pick_lines(pred.shape[0], n_panels, geometric=True)
    picks = np.asarray(picks, dtype=int)
    cols = min(3, len(picks))
    rows = int(np.ceil(len(picks) / cols))
    lo, hi = _shared_limits(pred[picks], ref)
    color = sns.color_palette()[0]

    fig = plt.figure(figsize=(4.3 * cols, 4.0 * rows))
    for k, frame in enumerate(picks):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="0.72", lw=3.5,
                solid_capstyle="round", label="reference", zorder=1)
        u = pred[frame]
        if mode == "scatter":
            ax.scatter(u[:, 0], u[:, 1], u[:, 2], s=3, color=color, alpha=0.7,
                       depthshade=False, label="prediction")
        else:
            ax.plot(u[:, 0], u[:, 1], u[:, 2], color=color, lw=1.6, label="prediction")
        ax.scatter([ref[0, 0]], [ref[0, 1]], [ref[0, 2]], color="crimson", s=30,
                   zorder=5, label="IC")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((1.0, 1.0, 0.85), zoom=1.05)
        ax.view_init(elev=20, azim=-58)
        ax.tick_params(labelsize="x-small", pad=0)
        title = f"epoch {int(epochs[frame])}"
        if rmse is not None:
            title += f"   RMSE {rmse[frame]:.2e}"
        ax.set_title(title, fontsize="medium")
        if k == 0:
            ax.set_xlabel("$x$", labelpad=-2)
            ax.set_ylabel("$y$", labelpad=-2)
            ax.set_zlabel("$z$", labelpad=-2)
            ax.legend(loc="upper left", fontsize="x-small")

    fig.suptitle(
        f"Predicted orbit over training, raw units — "
        f"axes fixed across panels{f' — {label}' if label else ''}"
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# every epoch on one pair of axes
# --------------------------------------------------------------------------
def fig_trajectory_projections(
    epochs: np.ndarray, pred: np.ndarray, ref: np.ndarray,
    max_lines: int = 100, thin: int = 4, equal_aspect: bool = False, label: str = "",
) -> Figure:
    """The orbit in each coordinate plane, one polyline per epoch, raw units.

    Colour encodes epoch, so a fan collapsing onto the reference is training
    converging. With ``equal_aspect=True`` the panels show the true geometry of
    the orbit; with ``False`` (the default here) each panel fills its axes, which
    exaggerates the narrow directions but makes the movement legible. Both are
    honest renderings of unnormalised values — they differ only in aspect ratio.
    """
    keep = _pick_lines(pred.shape[0], max_lines)
    sub = pred[keep]
    cols = _thin_columns(pred.shape[1], max(2, pred.shape[1] // thin))
    norm = _epoch_norm(epochs[keep])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, (i, j) in zip(axes, PLANES):
        ax.plot(ref[:, i], ref[:, j], color="0.55", lw=3.2, solid_capstyle="round",
                label="reference", zorder=3)
        _line_collection(ax, sub[:, cols, i], sub[:, cols, j], epochs[keep], norm=norm)
        ax.scatter([ref[0, i]], [ref[0, j]], color="crimson", s=34, zorder=5, label="IC")
        ax.set_xlabel(f"${STATE[i]}$")
        ax.set_ylabel(f"${STATE[j]}$")
        ax.set_title(f"${STATE[i]}$–${STATE[j]}$ plane")
        if equal_aspect:
            ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best", fontsize="x-small")
    _add_epoch_colorbar(fig, axes[-1], epochs[keep])

    aspect = "true geometry (equal aspect)" if equal_aspect else "axes auto-scaled per panel"
    fig.suptitle(
        f"Orbit in the coordinate planes across epochs, raw units — {aspect}"
        f"{f' — {label}' if label else ''}"
    )
    fig.tight_layout()
    return fig


def fig_trajectory_time_slices(
    epochs: np.ndarray, t: np.ndarray, pred: np.ndarray, ref: np.ndarray,
    max_lines: int = 100, thin: int = 4, label: str = "",
) -> Figure:
    """The predicted trajectory against ``t``, one curve per epoch, raw units.

    This is the view where the deformation is easiest to read: the reference is a
    single curve, and the fan of previous epochs shows which parts of the time
    domain moved first and which lagged.
    """
    keep = _pick_lines(pred.shape[0], max_lines)
    sub = pred[keep]
    cols = _thin_columns(pred.shape[1], max(2, pred.shape[1] // thin))
    norm = _epoch_norm(epochs[keep])

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True)
    for ax, j in zip(axes, range(3)):
        ax.plot(t[cols], ref[cols, j], color="black", ls="--", lw=1.8,
                label="reference", zorder=4)
        _line_collection(ax, np.tile(t[cols], (len(keep), 1)), sub[:, cols, j],
                         epochs[keep], norm=norm)
        ax.set_xlabel("$t$")
        ax.set_ylabel(f"${STATE[j]}(t)$")
        ax.set_title(f"${STATE[j]}$")
        if j == 0:
            ax.legend(loc="best", fontsize="x-small")
    _add_epoch_colorbar(fig, axes[-1], epochs[keep])

    fig.suptitle(
        f"Trajectory against time across epochs, raw units{f' — {label}' if label else ''}"
    )
    fig.tight_layout()
    return fig


def fig_trajectory_extent(
    epochs: np.ndarray, pred: np.ndarray, ref: np.ndarray, log_epoch: bool = True, label: str = "",
) -> Figure:
    """Min/max envelope of the whole collocation batch per epoch, against the truth.

    Collapses 3000 points into two curves per state, which makes the question
    "is the batch converging onto the reference, and when" readable at a glance
    without a single rescaling.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = state_colors()
    shift = log_epoch and epochs.min() < 1
    x = epochs + 1 if shift else epochs

    for ax, j in zip(axes, range(3)):
        lo, hi = pred[:, :, j].min(axis=1), pred[:, :, j].max(axis=1)
        ax.fill_between(x, lo, hi, color=colors[STATE[j]], alpha=0.25,
                        label=r"prediction envelope ($N_c$ points)")
        ax.plot(x, lo, color=colors[STATE[j]], lw=1.0)
        ax.plot(x, hi, color=colors[STATE[j]], lw=1.0)
        for value, tag in ((ref[:, j].min(), "reference min"), (ref[:, j].max(), "reference max")):
            ax.axhline(value, color="black", ls="--", lw=1.2, label=tag)
        ax.set_xlabel("epoch + 1" if shift else "epoch")
        ax.set_ylabel(f"${STATE[j]}$")
        ax.set_title(f"${STATE[j]}$")
        if log_epoch:
            ax.set_xscale("log")
        if j == 0:
            ax.legend(loc="best", fontsize="x-small")
    fig.suptitle(
        f"Batch envelope against the reference band, raw units{f' — {label}' if label else ''}"
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# the batching question
# --------------------------------------------------------------------------
def fig_gradient_equivalence(report: dict, label: str = "") -> Figure:
    """How far the batched gradient is from the accumulated per-point gradient.

    Bars are relative differences on a log axis; the shaded band is the float32
    round-off scale, so a bar inside it means the two are the same number.
    """
    names = ["relative difference in loss", "relative difference in gradient"]
    values = [abs(report["rel_loss_diff"]), abs(report["rel_grad_diff"])]
    bound = report["roundoff_bound"]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax = axes[0]
    ax.axvspan(bound / 10, bound * 10, color="0.85", zorder=0,
               label=rf"float32 round-off scale ($\sqrt{{3N_c}}\,\epsilon$)")
    ax.barh(names, _positive(values), color=sns.color_palette()[:2], height=0.5, zorder=2)
    ax.axvline(report["float32_eps"], color="crimson", ls="--", lw=1.4, zorder=3,
               label=r"$\epsilon_{32}$")
    for y, value in enumerate(values):
        ax.text(value * 1.5, y, f"{value:.2e}", va="center", fontsize="small")
    ax.set_xscale("log")
    ax.set_xlabel("relative difference (log scale)")
    ax.set_title("(a) Batched vs point-by-point")
    ax.legend(loc="lower right", fontsize="x-small")

    ax = axes[1]
    ax.axis("off")
    table = [
        ["collocation points", f"{int(report['n_points']):,}"],
        ["network parameters", f"{int(report['n_params']):,}"],
        ["loss, one batched pass", f"{report['loss_sum_batched']:.9g}"],
        ["loss, summed point by point", f"{report['loss_sum_looped']:.9g}"],
        [r"$\|\nabla L\|$, batched", f"{report['grad_norm_batched']:.6g}"],
        [r"$\|\nabla L\|$, point by point", f"{report['grad_norm_looped']:.6g}"],
        ["cosine similarity", f"{report['grad_cosine']:.9f}"],
        ["max abs. gradient difference", f"{report['max_abs_grad_diff']:.3e}"],
        [r"float32 $\epsilon$", f"{report['float32_eps']:.3e}"],
        [r"round-off bound $\sqrt{3N_c}\,\epsilon$", f"{bound:.3e}"],
    ]
    body = ax.table(cellText=table, loc="center", cellLoc="left", colWidths=[0.62, 0.38])
    body.auto_set_font_size(False)
    body.set_fontsize(9.5)
    body.scale(1.0, 1.55)
    for row in range(len(table)):
        body[row, 0].set_facecolor("0.94")
    ax.set_title("(b) The two computations side by side")

    fig.suptitle(
        "One batched backward pass equals the sum of the per-point gradients"
        f"{f' — {label}' if label else ''}"
    )
    fig.tight_layout()
    return fig


def fig_batch_size_comparison(
    curves: pd.DataFrame, timing: dict[str, float], label: str = ""
) -> Figure:
    """Loss against compute for full-batch and mini-batch training.

    Left: against cumulative point evaluations, the fair axis — every arm sees
    the same collocation points, only the number of optimiser steps differs.
    Right: against wall-clock, which is what a long run actually costs.
    """
    order = list(timing.keys())
    palette = dict(zip(order, sns.color_palette("colorblind", len(order))))

    def nice(key: str) -> str:
        return "full batch (1 step/epoch)" if key == "full" else f"mini-batch {key}"

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    for key in order:
        sub = curves[curves["batch_size"] == key]
        ax.loglog(_positive(sub["points_evaluated"].to_numpy()),
                  _positive(sub["loss"].to_numpy()), color=palette[key], lw=1.6,
                  label=f"{nice(key)}  ({sub['step'].iloc[-1] + 1:,} steps)")
    ax.set_xlabel("collocation-point evaluations")
    ax.set_ylabel("loss")
    ax.set_title("(a) Loss against equal compute")
    ax.legend(loc="best", fontsize="x-small")

    ax = axes[1]
    for key in order:
        sub = curves[curves["batch_size"] == key]
        # Each arm was timed end to end, so its per-step cost is the total over
        # its step count — the axis is the wall-clock that arm actually spent.
        elapsed = timing[key] * sub["points_evaluated"].to_numpy() / sub["points_evaluated"].max()
        ax.loglog(_positive(elapsed), _positive(sub["loss"].to_numpy()),
                  color=palette[key], lw=1.6, label=nice(key))
    ax.set_xlabel("wall-clock spent (s)")
    ax.set_ylabel("loss")
    ax.set_title("(b) Loss against wall-clock")
    ax.legend(loc="best", fontsize="x-small")

    fig.suptitle(
        f"Batch size under a matched point-evaluation budget{f' — {label}' if label else ''}"
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# animation
# --------------------------------------------------------------------------
def animate_trajectory(
    epochs: np.ndarray, t: np.ndarray, pred: np.ndarray, ref: np.ndarray,
    path: Path | str, n_frames: int = 80, fps: int = 12, thin: int = 4,
    label: str = "", dpi: int = 110,
) -> Path:
    """Write the training trajectory as a GIF: orbit plus its three projections.

    Axes are fixed across every frame — the reference sets the box, so what moves
    is the prediction and nothing else. Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = _pick_lines(pred.shape[0], n_frames)
    cols = _thin_columns(pred.shape[1], max(2, pred.shape[1] // thin))
    err = np.sqrt(((pred - ref[None]) ** 2).mean(axis=(1, 2)))
    lo, hi = _shared_limits(pred, ref)
    color = sns.color_palette()[0]

    fig = plt.figure(figsize=(10.5, 7.6))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_planes = [fig.add_subplot(2, 2, k + 2) for k in range(3)]

    ax3d.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="0.75", lw=3.2, label="reference")
    ax3d.scatter([ref[0, 0]], [ref[0, 1]], [ref[0, 2]], color="crimson", s=34, label="IC")
    ax3d.set_xlim(lo[0], hi[0])
    ax3d.set_ylim(lo[1], hi[1])
    ax3d.set_zlim(lo[2], hi[2])
    ax3d.set_box_aspect((1.0, 1.0, 0.85), zoom=1.05)
    ax3d.view_init(elev=20, azim=-58)
    ax3d.tick_params(labelsize="x-small", pad=0)
    ax3d.set_xlabel("$x$", labelpad=-4)
    ax3d.set_ylabel("$y$", labelpad=-4)
    ax3d.set_zlabel("$z$", labelpad=-4)
    ax3d.legend(loc="upper left", fontsize="x-small")
    (orbit,) = ax3d.plot([], [], [], color=color, lw=1.8)

    plane_lines = []
    for ax, (i, j) in zip(ax_planes, PLANES):
        ax.plot(ref[:, i], ref[:, j], color="0.75", lw=3.0, label="reference")
        ax.scatter([ref[0, i]], [ref[0, j]], color="crimson", s=30, label="IC")
        ax.set_xlim(lo[i], hi[i])
        ax.set_ylim(lo[j], hi[j])
        ax.set_xlabel(f"${STATE[i]}$")
        ax.set_ylabel(f"${STATE[j]}$")
        ax.set_title(f"${STATE[i]}$–${STATE[j]}$")
        (line,) = ax.plot([], [], color=color, lw=1.6)
        plane_lines.append(line)
    ax_planes[0].legend(loc="best", fontsize="x-small")

    stamp = fig.suptitle("")
    fig.tight_layout()

    def update(k: int):
        frame = keep[k]
        u = pred[frame]
        orbit.set_data_3d(u[cols, 0], u[cols, 1], u[cols, 2])
        for line, (i, j) in zip(plane_lines, PLANES):
            line.set_data(u[cols, i], u[cols, j])
        stamp.set_text(
            f"epoch {int(epochs[frame])} of {int(epochs[-1])}   |   "
            f"RMSE {err[frame]:.3e}   |   {label}"
        )
        return [orbit, *plane_lines, stamp]

    anim = FuncAnimation(fig, update, frames=len(keep), interval=1000 / fps,
                         blit=False, repeat=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# stage boundaries (time marching)
# --------------------------------------------------------------------------
def fig_marching_seams(
    t: np.ndarray, pred: np.ndarray, ref: np.ndarray, edges: np.ndarray,
    stage_rmse: np.ndarray | None = None, label: str = "",
) -> Figure:
    """The piecewise solution against the reference, with its stage seams marked.

    A marching run is stitched from ``n_segments`` independently-trained pieces,
    so the question is not only "is it close" but "is it continuous". The dashed
    verticals are the stage boundaries; a jump across one is error the stitching
    itself introduced, and it is invisible in any metric that averages over ``t``.
    """
    t = np.asarray(t, dtype=float)
    pred, ref = np.asarray(pred, dtype=float), np.asarray(ref, dtype=float)
    edges = np.asarray(edges, dtype=float)
    inner = edges[1:-1]
    colors = state_colors()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.4), gridspec_kw={"width_ratios": [1.45, 1]})

    ax = axes[0]
    for j, name in enumerate(STATE):
        ax.plot(t, ref[:, j], color="0.55", lw=2.6, solid_capstyle="round", zorder=1)
        ax.plot(t, pred[:, j], color=colors[name], lw=1.5, label=f"${name}$", zorder=3)
    for e in inner:
        ax.axvline(e, color="crimson", ls="--", lw=1.2, zorder=2, label="stage boundary")
    ax.set_xlabel("$t$")
    ax.set_ylabel("state")
    ax.set_title("(a) Piecewise solution (thick grey) against the reference")
    handles, labels = ax.get_legend_handles_labels()
    seen, uniq = set(), []
    for h, lab in zip(handles, labels):
        if lab not in seen:
            seen.add(lab)
            uniq.append((h, lab))
    ax.legend(*zip(*uniq), loc="best", fontsize="x-small")

    ax = axes[1]
    ax.semilogy(t, _positive(np.linalg.norm(pred - ref, axis=1)), color="black", lw=1.4,
                label=r"$\|u_{\rm pred}-u_{\rm ref}\|_2$")
    for e in inner:
        ax.axvline(e, color="crimson", ls="--", lw=1.2)
    if stage_rmse is not None:
        for k, value in enumerate(np.asarray(stage_rmse)):
            mid = 0.5 * (edges[k] + edges[k + 1])
            ax.annotate(f"{value:.1e}", (mid, value), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize="x-small", color="crimson")
        ax.semilogy(0.5 * (edges[:-1] + edges[1:]), _positive(np.asarray(stage_rmse)),
                    "o", color="crimson", ms=4, label="per-stage RMSE")
    ax.set_xlabel("$t$")
    ax.set_ylabel("error (log scale)")
    ax.set_title("(b) Where the error sits")
    ax.legend(loc="best", fontsize="x-small")

    fig.suptitle(
        f"Time-marching seams: {len(edges) - 1} stages"
        f"{f' — {label}' if label else ''}"
    )
    fig.tight_layout()
    return fig
