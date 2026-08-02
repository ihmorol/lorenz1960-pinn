"""Publication-quality figure generation for the FYDP-2 Lorenz-1960 PINN.

Every function here is pure plotting: it takes NumPy arrays / DataFrames and
returns a matplotlib ``Figure``. Nothing in this module imports torch, so the
figures can be regenerated from saved arrays without a training environment.

Styling is seaborn ``whitegrid`` on a colourblind-safe palette with serif type,
sized for a two-column report and saved to both PNG (slides) and PDF (LaTeX).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy.linalg import null_space

from .config import compute_error_metrics

if TYPE_CHECKING:
    from .history import TrainHistory

STATE = ("x", "y", "z")
FORMATS = ("png", "pdf")
DPI = 300


# --------------------------------------------------------------------------
# style
# --------------------------------------------------------------------------
def set_style(context: str = "paper", font_scale: float = 1.15) -> None:
    """Apply the project-wide academic plotting style."""
    sns.set_theme(
        context=context,
        style="whitegrid",
        palette="colorblind",
        font_scale=font_scale,
        rc={
            "figure.dpi": 110,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.titlesize": "medium",
            "axes.titleweight": "bold",
            "axes.labelsize": "medium",
            "axes.edgecolor": "0.35",
            "axes.linewidth": 0.9,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.45,
            "lines.linewidth": 1.6,
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.fontsize": "small",
        },
    )


def state_colors() -> dict[str, tuple[float, float, float]]:
    palette = sns.color_palette("colorblind")
    return dict(zip(STATE, palette[:3]))


set_style()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def save_figure(
    fig: Figure, outdir: str | Path, name: str, formats: Iterable[str] = FORMATS
) -> list[Path]:
    """Write ``fig`` as ``<outdir>/<name>.<ext>`` for each format, then close it."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in formats:
        path = out / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI)
        paths.append(path)
    plt.close(fig)
    return paths


def _log_trend(values: Sequence[float], frac: float = 0.02) -> np.ndarray:
    """Centred rolling geometric mean.

    Loss and gradient curves span several decades, so the trend has to be taken
    in log space — a linear moving average is dominated by the first few large
    values and lags the curve by thousands of epochs.
    """
    v = _positive(values)
    if v.size == 0:
        return v
    window = max(5, int(frac * v.size))
    smoothed = pd.Series(np.log10(v)).rolling(window, center=True, min_periods=1).mean()
    return np.power(10.0, smoothed.to_numpy())


def _thin(n: int, max_points: int = 4000) -> np.ndarray:
    """Index subsample so 20k-epoch curves stay light in vector output."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, max_points).astype(int))


def _positive(values: np.ndarray) -> np.ndarray:
    """Clip to the smallest positive value so log axes never see zeros."""
    v = np.asarray(values, dtype=float)
    good = v[v > 0]
    floor = good.min() if good.size else 1e-16
    return np.where(v > 0, v, floor)


def _mark_optimizer_switch(ax: plt.Axes, history: "TrainHistory") -> None:
    total = len(history.loss)
    if history.adam_iters < total:
        ax.axvspan(history.adam_iters, total, color="0.85", zorder=0)
        ax.axvline(history.adam_iters, color="crimson", ls="--", lw=1.2, label="L-BFGS start")


def _long_layer_frame(history: "TrainHistory") -> pd.DataFrame:
    """Tidy per-layer gradient norms, smoothed so layers stay comparable."""
    rows = []
    for layer, norms in history.layer_grad_norms.items():
        rows.append(pd.DataFrame({
            "epoch": history.log_epoch,
            "layer": layer,
            "grad_norm": norms,
            "trend": _log_trend(norms, frac=0.05),
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _thin_ticks(ax: plt.Axes, nbins: int = 6) -> None:
    """Keep long epoch axes from colliding their tick labels."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins))


def _error_frame(t: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> pd.DataFrame:
    err = pred - ref
    return pd.DataFrame(
        {
            "t": np.tile(t, len(STATE)),
            "state": np.repeat(STATE, t.size),
            "error": err.T.reshape(-1),
            "abs_error": np.abs(err).T.reshape(-1),
            "pred": pred.T.reshape(-1),
            "ref": ref.T.reshape(-1),
        }
    )


# --------------------------------------------------------------------------
# training / optimisation
# --------------------------------------------------------------------------
def fig_training_dynamics(history: "TrainHistory", label: str = "") -> Figure:
    """Loss convergence, loss decomposition, LR schedule, and true-error tracking."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    loss = _positive(history.loss)
    it = np.arange(loss.size)
    idx = _thin(loss.size)

    ax = axes[0, 0]
    ax.semilogy(it[idx], loss[idx], color=sns.color_palette()[0], lw=0.9,
                alpha=0.85, label="per-iteration loss")
    ax.semilogy(it[idx], _log_trend(loss)[idx], color="0.15", lw=1.4, label="rolling geometric mean")
    _mark_optimizer_switch(ax, history)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\mathcal{L}$")
    ax.set_title("(a) Total loss convergence")
    ax.legend(loc="best")

    ax = axes[0, 1]
    soft_ic = bool(np.any(np.asarray(history.ic_loss, dtype=float) > 0))
    if history.log_epoch:
        ax.semilogy(history.log_epoch, _positive(history.residual_loss),
                    label=r"residual $\mathcal{L}_r$")
        if soft_ic:
            ax.semilogy(history.log_epoch, _positive(history.ic_loss),
                        label=r"initial condition $\mathcal{L}_0$")
            ax.legend(loc="best")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss term")
    ax.set_title("(b) Loss decomposition" if soft_ic
                 else "(b) Loss decomposition (hard IC: no penalty term)")

    ax = axes[1, 0]
    if history.log_epoch:
        ax.plot(history.log_epoch, history.lr, color=sns.color_palette()[4])
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("(c) Learning-rate schedule")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[1, 1]
    if history.ref_epoch:
        ax.semilogy(history.ref_epoch, _positive(history.ref_mse), label="MSE vs reference solution")
        train_at_eval = [history.loss[min(e, len(history.loss) - 1)] for e in history.ref_epoch]
        ax.semilogy(history.ref_epoch, _positive(train_at_eval), ls="--", label="training loss")
        ax.legend(loc="best")
    ax.set_xlabel("epoch")
    ax.set_ylabel("error")
    ax.set_title("(d) Physics loss vs true error")

    for ax in axes.flat:
        _thin_ticks(ax)
    fig.suptitle(f"Training dynamics{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


def fig_gradient_diagnostics(history: "TrainHistory", label: str = "") -> Figure:
    """Gradient-descent health: global and per-layer norms, step size, loss coupling."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    if history.log_epoch:
        gn = _positive(history.grad_norm)
        ax.semilogy(history.log_epoch, gn, color=sns.color_palette()[0], lw=0.9,
                    alpha=0.85, label="observed")
        ax.semilogy(history.log_epoch, _log_trend(gn), color="0.15", lw=1.4,
                    label="rolling geometric mean")
        ax.legend(loc="best")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    ax.set_title("(a) Global gradient norm")

    ax = axes[0, 1]
    layers = _long_layer_frame(history)
    if not layers.empty:
        # Trend, not raw: with a spiking loss the raw traces overlap into a solid
        # block and the layer-to-layer comparison this panel exists for is lost.
        sns.lineplot(data=layers, x="epoch", y="trend", hue="layer",
                     ax=ax, lw=1.3, palette="viridis", legend="brief")
        ax.set_yscale("log")
        ax.legend(loc="best", ncol=2, fontsize="x-small", title=None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("layer gradient norm (trend)")
    ax.set_title("(b) Per-layer gradient magnitude")

    ax = axes[1, 0]
    if history.log_epoch and any(u > 0 for u in history.update_norm):
        ax.semilogy(history.log_epoch, _positive(history.update_norm),
                    color=sns.color_palette()[2], label=r"$\|\theta_{k+1}-\theta_k\|_2$")
        twin = ax.twinx()
        twin.plot(history.log_epoch, history.lr, color=sns.color_palette()[4], ls=":", lw=1.2)
        twin.set_ylabel("learning rate", color=sns.color_palette()[4])
        twin.grid(False)
        ax.legend(loc="best")
    ax.set_xlabel("epoch")
    ax.set_ylabel("parameter update norm")
    ax.set_title("(c) Effective step size")

    ax = axes[1, 1]
    if history.log_epoch:
        sc = ax.scatter(
            _positive(history.grad_norm), _positive(history.logged_loss),
            c=history.log_epoch, cmap="viridis", s=9, alpha=0.75, edgecolors="none",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.colorbar(sc, ax=ax, label="epoch")
    ax.set_xlabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
    ax.set_ylabel(r"$\mathcal{L}$")
    ax.set_title("(d) Gradient norm vs loss")

    for ax in axes.flat[:3]:  # (d) is a scatter over gradient norm, not epoch
        _thin_ticks(ax)
    fig.suptitle(f"Gradient-descent diagnostics{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# collocation points
# --------------------------------------------------------------------------
def fig_collocation_points(
    collocation_t: np.ndarray,
    t_span: tuple[float, float],
    residual: np.ndarray | None = None,
    label: str = "",
) -> Figure:
    """Quality of the Latin-hypercube collocation design and where physics is violated."""
    t = np.asarray(collocation_t, dtype=float).reshape(-1)
    t0, tf = t_span
    n = t.size
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    sns.histplot(x=t, bins=40, stat="density", kde=True, ax=ax, kde_kws={"cut": 0},
                 color=sns.color_palette()[0], edgecolor="white", alpha=0.7)
    sns.rugplot(x=t, ax=ax, height=0.04, color="0.3", lw=0.4, alpha=0.5)
    ax.axhline(1.0 / (tf - t0), color="crimson", ls="--", lw=1.2, label="uniform density")
    ax.set_xlabel("t")
    ax.set_ylabel("density")
    ax.set_title(f"(a) Collocation distribution (n = {n})")
    ax.legend(loc="best")

    ax = axes[0, 1]
    ts = np.sort(t)
    ecdf = np.arange(1, n + 1) / n
    uniform = (ts - t0) / (tf - t0)
    ks = float(np.abs(ecdf - uniform).max())
    ax.plot(ts, ecdf, label="empirical CDF")
    ax.plot(ts, uniform, ls="--", color="crimson", label="uniform CDF")
    ax.set_xlabel("t")
    ax.set_ylabel("F(t)")
    ax.set_title(f"(b) Uniformity check ($D_n$ = {ks:.4f})")
    ax.legend(loc="best")

    ax = axes[1, 0]
    gaps = np.diff(ts) * n / (tf - t0)
    sns.histplot(x=gaps, bins=40, stat="density", ax=ax,
                 color=sns.color_palette()[2], edgecolor="white", alpha=0.75)
    ax.axvline(1.0, color="crimson", ls="--", lw=1.2, label="ideal spacing")
    ax.set_xlabel("normalised nearest-neighbour gap")
    ax.set_ylabel("density")
    ax.set_title("(c) Spacing regularity")
    ax.legend(loc="best")

    ax = axes[1, 1]
    if residual is not None:
        r = np.abs(np.asarray(residual, dtype=float))
        mag = _positive(np.linalg.norm(r, axis=1) if r.ndim == 2 else r)
        ax.scatter(t, mag, s=6, alpha=0.35, color=sns.color_palette()[3], edgecolors="none")
        bins = np.linspace(t0, tf, 41)
        centre = 0.5 * (bins[:-1] + bins[1:])
        which = np.clip(np.digitize(t, bins) - 1, 0, bins.size - 2)
        median = np.array([np.median(mag[which == i]) if np.any(which == i) else np.nan
                           for i in range(bins.size - 1)])
        ax.plot(centre, median, color="black", lw=1.6, label="binned median")
        ax.set_yscale("log")
        ax.legend(loc="best")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|r(t)\|_2$")
    ax.set_title("(d) Trained residual at collocation points")

    fig.suptitle(f"Collocation design{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# solution and error
# --------------------------------------------------------------------------
def fig_solution_vs_reference(
    t: np.ndarray, pred: np.ndarray, ref: np.ndarray, label: str = ""
) -> Figure:
    """Per-state PINN solution against the locked baseline, with the signed error beside it."""
    colors = state_colors()
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True,
                             gridspec_kw={"width_ratios": [1.6, 1.0]})

    for j, name in enumerate(STATE):
        left, right = axes[j, 0], axes[j, 1]
        # Reference drawn as a wide halo: at ~1e-5 error a dashed overlay is
        # invisible under the prediction, which reads as a missing curve.
        left.plot(t, ref[:, j], color="0.72", lw=4.0, solid_capstyle="round",
                  label="reference (DOP853)")
        left.plot(t, pred[:, j], color=colors[name], lw=1.5, label="PINN")
        left.set_ylabel(f"${name}(t)$")
        if j == 0:
            left.legend(loc="best", ncol=2)

        right.plot(t, pred[:, j] - ref[:, j], color=colors[name], lw=1.4)
        right.axhline(0.0, color="0.4", lw=0.8)
        right.set_ylabel(f"$e_{name}$")
        right.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    axes[0, 0].set_title("Solution")
    axes[0, 1].set_title("Signed error")
    fig.suptitle(f"PINN solution vs reference{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


def fig_error_analysis(
    t: np.ndarray, pred: np.ndarray, ref: np.ndarray, label: str = ""
) -> Figure:
    """Error growth, distribution, and predicted-vs-true agreement."""
    colors = state_colors()
    frame = _error_frame(t, pred, ref)
    err = pred - ref
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    for j, name in enumerate(STATE):
        ax.semilogy(t, _positive(np.abs(err[:, j])), color=colors[name], label=f"$|e_{name}|$")
    ax.semilogy(t, _positive(np.linalg.norm(err, axis=1)), color="black", lw=1.2,
                ls="--", label=r"$\|e\|_2$")
    ax.set_xlabel("t")
    ax.set_ylabel("absolute error")
    ax.set_title("(a) Error growth in time")
    ax.legend(loc="best", ncol=2)

    ax = axes[0, 1]
    sns.violinplot(data=frame, x="state", y="error", ax=ax, hue="state",
                   palette=[colors[s] for s in STATE], inner="quartile", legend=False, cut=0)
    ax.axhline(0.0, color="0.4", lw=0.8)
    ax.set_xlabel("state variable")
    ax.set_ylabel("signed error")
    ax.set_title("(b) Error distribution")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[1, 0]
    for j, name in enumerate(STATE):
        ss_res = float(np.sum(err[:, j] ** 2))
        ss_tot = float(np.sum((ref[:, j] - ref[:, j].mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.scatter(ref[:, j], pred[:, j], s=8, alpha=0.5, color=colors[name],
                   edgecolors="none", label=f"${name}$  $R^2$={r2:.5f}")
    lims = [min(ref.min(), pred.min()), max(ref.max(), pred.max())]
    ax.plot(lims, lims, color="0.3", ls="--", lw=1.0)
    ax.set_xlabel("reference value")
    ax.set_ylabel("PINN prediction")
    ax.set_title("(c) Parity plot")
    ax.legend(loc="best")

    ax = axes[1, 1]
    l2 = np.linalg.norm(err, axis=1)
    ref_norm = np.linalg.norm(ref, axis=1)
    ax.plot(t, 100.0 * l2 / np.where(ref_norm > 0, ref_norm, np.nan),
            color=sns.color_palette()[3])
    ax.set_xlabel("t")
    ax.set_ylabel("relative error (%)")
    ax.set_title("(d) Pointwise relative error")

    fig.suptitle(f"Error analysis{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


def fig_phase_portraits(pred: np.ndarray, ref: np.ndarray, label: str = "") -> Figure:
    """Projections of the trajectory plus the 3-D orbit, PINN over reference."""
    fig = plt.figure(figsize=(11, 8))
    planes = ((0, 1), (0, 2), (1, 2))
    titles = ("(a)", "(b)", "(c)")

    for pos, ((i, j), tag) in enumerate(zip(planes, titles), start=1):
        ax = fig.add_subplot(2, 2, pos)
        ax.plot(ref[:, i], ref[:, j], color="0.72", lw=4.0, solid_capstyle="round",
                label="reference")
        ax.plot(pred[:, i], pred[:, j], color=sns.color_palette()[0], lw=1.5, label="PINN")
        ax.scatter([ref[0, i]], [ref[0, j]], color="crimson", s=28, zorder=5, label="start")
        ax.set_xlabel(f"${STATE[i]}$")
        ax.set_ylabel(f"${STATE[j]}$")
        ax.set_title(f"{tag} ${STATE[i]}$–${STATE[j]}$ plane")
        if pos == 1:
            ax.legend(loc="best")

    ax = fig.add_subplot(2, 2, 4, projection="3d")
    ax.plot(ref[:, 0], ref[:, 1], ref[:, 2], color="0.72", lw=4.0, label="reference")
    ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], color=sns.color_palette()[0], lw=1.5,
            label="PINN")
    ax.scatter([ref[0, 0]], [ref[0, 1]], [ref[0, 2]], color="crimson", s=28, label="start")
    ax.view_init(elev=20, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.85), zoom=1.05)
    ax.set_xlabel("$x$", labelpad=2)
    ax.set_ylabel("$y$", labelpad=2)
    ax.set_zlabel("$z$", labelpad=2)
    ax.tick_params(labelsize="small", pad=1)
    ax.set_title("(d) Phase-space trajectory")  # legend shared with panel (a)

    fig.suptitle(f"Phase portraits{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def fig_metrics_summary(metrics: pd.DataFrame, label: str = "") -> Figure:
    """Grouped error metrics per state variable plus the numeric table."""
    long = metrics.melt(id_vars="state", var_name="metric", value_name="value")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.5, 1.0]})

    ax = axes[0]
    sns.barplot(data=long, x="state", y="value", hue="metric", ax=ax, edgecolor="white")
    ax.set_yscale("log")
    ax.set_xlabel("state variable")
    ax.set_ylabel("error (log scale)")
    ax.set_title("(a) Error metrics by state")
    ax.legend(loc="best", title=None, ncol=3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1e", fontsize="xx-small", padding=2, rotation=90)

    ax = axes[1]
    ax.axis("off")
    formatted = metrics.copy()
    for col in formatted.columns:
        if col != "state":
            formatted[col] = formatted[col].map(lambda v: f"{v:.3e}")
    table = ax.table(cellText=formatted.values, colLabels=formatted.columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    for col in range(len(formatted.columns)):
        table[0, col].set_facecolor("0.9")
        table[0, col].set_text_props(weight="bold")
    ax.set_title("(b) Metric table")

    fig.suptitle(f"Accuracy summary{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# physics consistency
# --------------------------------------------------------------------------
def fig_physics_residual(t: np.ndarray, residual: np.ndarray, label: str = "") -> Figure:
    """Where and how badly the trained network violates the governing ODEs."""
    r = np.asarray(residual, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1.0]})
    colors = state_colors()

    ax = axes[0]
    for j, name in enumerate(STATE):
        ax.semilogy(t, _positive(np.abs(r[:, j])), color=colors[name], lw=1.2,
                    label=rf"$|r_{name}|$")
    ax.semilogy(t, _positive(np.linalg.norm(r, axis=1)), color="black", ls="--", lw=1.2,
                label=r"$\|r\|_2$")
    ax.set_xlabel("t")
    ax.set_ylabel("residual magnitude")
    ax.set_title("(a) ODE residual over the domain")
    ax.legend(loc="best", ncol=2)

    ax = axes[1]
    frame = pd.DataFrame({
        "state": np.repeat(STATE, t.size),
        "log_abs_residual": np.log10(_positive(np.abs(r)).T.reshape(-1)),
    })
    sns.boxplot(data=frame, x="state", y="log_abs_residual", ax=ax, hue="state",
                palette=[colors[s] for s in STATE], legend=False, fliersize=1.5)
    ax.set_xlabel("residual component")
    ax.set_ylabel(r"$\log_{10}|r|$")
    ax.set_title("(b) Residual distribution")

    fig.suptitle(f"Physics-residual diagnostics{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


def fig_invariant_drift(
    t: np.ndarray, pred: np.ndarray, ref: np.ndarray, coefficients: np.ndarray, label: str = ""
) -> Figure:
    """Conservation check.

    The Lorenz-1960 reduced system :math:`\\dot{u}_i = a_i u_j u_k` conserves every
    quadratic form :math:`I = \\alpha x^2 + \\beta y^2 + \\gamma z^2` whose weights
    are orthogonal to ``a``, since :math:`\\dot{I} = 2xyz(\\alpha a_1 + \\beta a_2 +
    \\gamma a_3)`. Two independent invariants exist; drift in them measures how much
    physics the network has quietly discarded.
    """
    basis = null_space(np.asarray(coefficients, dtype=float).reshape(1, 3))
    n_inv = basis.shape[1]
    fig, axes = plt.subplots(1, max(n_inv, 1), figsize=(5.5 * max(n_inv, 1), 4.2), squeeze=False)

    for i in range(n_inv):
        w = basis[:, i]
        ax = axes[0, i]
        for name, traj, style in (("PINN", pred, "-"), ("reference", ref, "--")):
            inv = (traj**2) @ w
            scale = abs(inv[0]) if abs(inv[0]) > 0 else 1.0
            ax.semilogy(t, _positive(np.abs(inv - inv[0]) / scale), ls=style, label=name)
        weights = ", ".join(f"{v:+.3f}" for v in w)
        ax.set_xlabel("t")
        ax.set_ylabel("relative drift")
        ax.set_title(f"$I_{i + 1}$ weights ({weights})")
        ax.legend(loc="best")

    fig.suptitle(f"Quadratic-invariant drift{f' — {label}' if label else ''}")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# PinnDE-style single-panel summary (kept for continuity with earlier results)
# --------------------------------------------------------------------------
def fig_paper_summary(
    t: np.ndarray, pred: np.ndarray, ref: np.ndarray, history: "TrainHistory"
) -> Figure:
    """The three-panel layout used in the FYDP-2 report: solution | error | loss."""
    colors = state_colors()
    mse = float(np.mean((pred - ref) ** 2))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    for j, name in enumerate(STATE):
        ax[0].plot(t, ref[:, j], color="0.72", lw=4.0, solid_capstyle="round",
                   label="reference" if j == 0 else None)
        ax[0].plot(t, pred[:, j], color=colors[name], lw=1.5, label=f"${name}$")
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("$x, y, z$")
    ax[0].set_title("Neural network solution vs reference")
    ax[0].legend(loc="best", ncol=2)

    for j, name in enumerate(STATE):
        ax[1].plot(t, pred[:, j] - ref[:, j], color=colors[name], label=f"$e_{name}$")
    ax[1].set_xlabel("t")
    ax[1].set_ylabel("error")
    ax[1].set_title(f"MSE: {mse:.2e}")
    ax[1].legend(loc="best")

    ax[2].semilogy(_positive(history.loss), lw=0.9)
    _mark_optimizer_switch(ax[2], history)
    if ax[2].get_legend_handles_labels()[0]:
        ax[2].legend(loc="best")
    ax[2].set_xlabel("iteration")
    ax[2].set_ylabel("loss")
    ax[2].set_title("Training loss")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# sweep figures (architecture search — used once multi-run results exist)
# --------------------------------------------------------------------------
def fig_architecture_heatmap(
    sweep: pd.DataFrame, value: str = "rmse", label: str = ""
) -> Figure:
    """Depth x width heatmap of a chosen metric, one panel per activation."""
    activations = sorted(sweep["activation"].unique())
    fig, axes = plt.subplots(1, len(activations), figsize=(4.2 * len(activations), 4.0),
                             squeeze=False, sharey=True)
    vmin, vmax = sweep[value].min(), sweep[value].max()

    for ax, act in zip(axes[0], activations):
        grid = (sweep[sweep["activation"] == act]
                .pivot_table(index="depth", columns="width", values=value, aggfunc="mean"))
        sns.heatmap(grid, ax=ax, annot=True, fmt=".1e", cmap="rocket_r",
                    vmin=vmin, vmax=vmax, cbar=(ax is axes[0, -1]),
                    annot_kws={"fontsize": "x-small"})
        ax.set_title(act)
        ax.set_xlabel("width")
        ax.set_ylabel("depth" if ax is axes[0, 0] else "")

    fig.suptitle(f"Architecture sweep — {value}{f' ({label})' if label else ''}")
    fig.tight_layout()
    return fig


def fig_activation_comparison(sweep: pd.DataFrame, value: str = "rmse") -> Figure:
    """Metric spread per activation across all architectures in the sweep."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    order = sweep.groupby("activation")[value].median().sort_values().index
    sns.boxplot(data=sweep, x="activation", y=value, order=order, ax=ax,
                hue="activation", palette="colorblind", legend=False, fliersize=0)
    sns.stripplot(data=sweep, x="activation", y=value, order=order, ax=ax,
                  color="0.2", size=3.5, alpha=0.6, jitter=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("activation function")
    ax.set_ylabel(value)
    ax.set_title(f"Activation comparison — {value}")
    fig.tight_layout()
    return fig


def fig_seed_robustness(sweep: pd.DataFrame, value: str = "rmse", by: str = "activation") -> Figure:
    """Run-to-run variability across random seeds, grouped by any config column."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.pointplot(data=sweep, x=by, y=value, ax=ax, errorbar=("pi", 95),
                  linestyle="none", capsize=0.15, color=sns.color_palette()[0])
    sns.stripplot(data=sweep, x=by, y=value, ax=ax, color="0.35", size=3.5, alpha=0.6, jitter=0.2)
    ax.set_yscale("log")
    ax.set_xlabel(by)
    ax.set_ylabel(value)
    ax.set_title(f"Seed robustness — {value} (95% interval)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class RunArtifacts:
    """Everything one training run produces that the figure suite consumes."""

    t: np.ndarray
    pred: np.ndarray
    ref: np.ndarray
    history: "TrainHistory"
    coefficients: np.ndarray
    collocation_t: np.ndarray
    collocation_residual: np.ndarray | None = None
    dense_residual: np.ndarray | None = None
    t_span: tuple[float, float] = (0.0, 1.0)
    label: str = ""

    @property
    def metrics(self) -> pd.DataFrame:
        return compute_error_metrics(self.ref, self.pred)


def write_run_report(
    run: RunArtifacts,
    outdir: str | Path,
    formats: Iterable[str] = FORMATS,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Write the results bundle: the metric table and every figure into ``outdir``.

    Bulk telemetry (per-iteration loss, sampled diagnostics) goes to ``data_dir``
    — it is regenerable and megabytes wide, so it belongs beside the checkpoint
    rather than in the tracked results tree. Defaults to ``outdir``.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir) if data_dir is not None else out
    data.mkdir(parents=True, exist_ok=True)

    metrics = run.metrics
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame({"iteration": np.arange(len(run.history.loss)), "loss": run.history.loss}).to_csv(
        data / "loss_history.csv", index=False
    )
    run.history.diagnostics_frame().to_csv(data / "training_diagnostics.csv", index=False)
    run.history.reference_frame().to_csv(data / "reference_error.csv", index=False)
    # The report figure keeps its original filename; diagnostics live beside it.
    save_figure(fig_paper_summary(run.t, run.pred, run.ref, run.history), out, "results", ("png",))
    generate_all(run, out / "figures", formats)
    return metrics


def generate_all(
    run: RunArtifacts, outdir: str | Path, formats: Iterable[str] = FORMATS
) -> dict[str, list[Path]]:
    """Render and save the full figure suite for a single run."""
    set_style()
    out = Path(outdir)
    written: dict[str, list[Path]] = {}

    def emit(name: str, fig: Figure) -> None:
        written[name] = save_figure(fig, out, name, formats)

    emit("training_dynamics", fig_training_dynamics(run.history, run.label))
    emit("gradient_diagnostics", fig_gradient_diagnostics(run.history, run.label))
    emit("collocation_points",
         fig_collocation_points(run.collocation_t, run.t_span, run.collocation_residual, run.label))
    emit("solution_vs_reference", fig_solution_vs_reference(run.t, run.pred, run.ref, run.label))
    emit("error_analysis", fig_error_analysis(run.t, run.pred, run.ref, run.label))
    emit("phase_portraits", fig_phase_portraits(run.pred, run.ref, run.label))
    emit("metrics_summary", fig_metrics_summary(run.metrics, run.label))
    emit("invariant_drift",
         fig_invariant_drift(run.t, run.pred, run.ref, run.coefficients, run.label))
    if run.dense_residual is not None:
        emit("physics_residual", fig_physics_residual(run.t, run.dense_residual, run.label))
    return written


def generate_sweep(
    sweep: pd.DataFrame, outdir: str | Path, value: str = "rmse",
    formats: Iterable[str] = FORMATS,
) -> dict[str, list[Path]]:
    """Render the architecture-search figures from a tidy multi-run results table.

    ``sweep`` needs one row per run with at least ``depth``, ``width``,
    ``activation``, ``seed`` and the metric column named by ``value``.
    """
    set_style()
    out = Path(outdir)
    written: dict[str, list[Path]] = {}
    if {"depth", "width", "activation"} <= set(sweep.columns):
        written["architecture_heatmap"] = save_figure(
            fig_architecture_heatmap(sweep, value), out, "architecture_heatmap", formats)
    if "activation" in sweep.columns:
        written["activation_comparison"] = save_figure(
            fig_activation_comparison(sweep, value), out, "activation_comparison", formats)
    if "seed" in sweep.columns:
        written["seed_robustness"] = save_figure(
            fig_seed_robustness(sweep, value), out, "seed_robustness", formats)
    return written
