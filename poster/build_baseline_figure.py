from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn.config import Config, reference_trajectory  # noqa: E402
from pinn.pinn import build_model  # noqa: E402


RUN = ROOT / "runs" / "4x60_f64_unit"
OUTPUT = ROOT / "poster" / "assets" / "defense_baseline.svg"
COLORS = {"x": "#167D9A", "y": "#D66A3A", "z": "#2A9D6F"}


def load_saved_run() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    summary = pd.read_csv(RUN / "run_summary.csv").iloc[0]
    assert summary["arch"] == "4x60_f64_unit"
    assert int(summary["depth"]) == 4 and int(summary["width"]) == 60
    assert summary["dtype"] == "float64" and summary["ic"] == "hard"

    cfg = Config(
        t_span=(float(summary["t_start"]), float(summary["t_end"])),
        depth=int(summary["depth"]),
        width=int(summary["width"]),
        activation=str(summary["activation"]),
        ic=str(summary["ic"]),
        ic_scale=str(summary["ic_scale"]),
        dtype=str(summary["dtype"]),
        epochs=int(summary["epochs"]),
        lbfgs_iters=int(summary["lbfgs_iters"]),
        n_collocation=int(summary["n_collocation"]),
        n_eval=int(summary["n_eval"]),
        lr_start=float(summary["lr_start"]),
        lr_end=float(summary["lr_end"]),
        lr_decay=float(summary["lr_decay"]),
        lr_decay_every=int(summary["lr_decay_every"]),
        seed=int(summary["seed"]),
    )
    model = build_model(cfg).to(dtype=torch.float64)
    model.load_state_dict(torch.load(RUN / "history" / "pinn.pt", map_location="cpu", weights_only=True))
    model.eval()

    t, reference = reference_trajectory(cfg, n=2401)
    with torch.no_grad():
        prediction = model(torch.as_tensor(t[:, None], dtype=torch.float64)).numpy()
    loss = pd.read_csv(RUN / "history" / "loss_history.csv")
    return t, reference, prediction, loss, summary


def build() -> None:
    t, reference, prediction, loss, summary = load_saved_run()
    error = np.linalg.norm(prediction - reference, axis=1)
    best_loss = np.minimum.accumulate(loss["loss"].to_numpy())

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "svg.fonttype": "none",
    })
    fig = plt.figure(figsize=(12, 3), facecolor="white")
    grid = fig.add_gridspec(2, 4, width_ratios=(1, 1, 1, 1.25), hspace=0.52, wspace=0.32)

    states = (("x", r"$x(t)$"), ("y", r"$y(t)$"), ("z", r"$z(t)$"))
    for idx, (name, label) in enumerate(states):
        ax = fig.add_subplot(grid[:, idx])
        ax.plot(t, reference[:, idx], color="#27364A", linewidth=3.8,
                label="DOP853 reference" if idx == 0 else None)
        ax.plot(t, prediction[:, idx], color=COLORS[name], linewidth=2.8,
                linestyle=(0, (5, 3)), label="full-interval PINN" if idx == 0 else None)
        ax.set_title(label, loc="left", pad=4)
        ax.grid(color="#DCE3EA", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel(r"time $t$")
        if idx == 0:
            ax.legend(loc="lower left", frameon=False)

    loss_ax = fig.add_subplot(grid[0, 3])
    iterations = loss["iteration"].to_numpy()
    loss_ax.semilogy(iterations, loss["loss"], color="#AAB4C0", linewidth=1.0,
                     alpha=0.7, label="training loss")
    loss_ax.semilogy(iterations, best_loss, color="#6C4BC1", linewidth=3.0,
                     label="best-so-far")
    loss_ax.axvline(int(summary["epochs"]), color="#56616F", linewidth=1.5,
                    linestyle="--")
    loss_ax.text(int(summary["epochs"]) + 500, loss_ax.get_ylim()[0] * 1.7, "L-BFGS",
                 ha="left", va="bottom", color="#56616F", fontsize=9)
    loss_ax.set_title("Loss history", loc="left", pad=3)
    loss_ax.set_ylabel("residual loss")
    loss_ax.grid(color="#DCE3EA", linewidth=0.8, which="major")
    loss_ax.spines[["top", "right"]].set_visible(False)
    loss_ax.legend(loc="lower left", frameon=False, fontsize=9)

    error_ax = fig.add_subplot(grid[1, 3])
    error_ax.fill_between(t, error, color="#D8EEF4", alpha=0.9)
    error_ax.plot(t, error, color="#167D9A", linewidth=2.5)
    error_ax.set_title("State error", loc="left", pad=2)
    error_ax.set_xlabel(r"time $t$")
    error_ax.set_ylabel(r"$\|u_{PINN}-u_{ref}\|_2$")
    error_ax.grid(axis="y", color="#DCE3EA", linewidth=0.8)
    error_ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(left=0.045, right=0.992, bottom=0.18, top=0.93, hspace=0.72)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, format="svg", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
