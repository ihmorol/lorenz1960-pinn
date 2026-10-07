"""Render the single-network full-interval training-plateau figure from runs/4x60_f64_unit/history/loss_history.csv.

Saves a vector PDF (runs/figures/plateau_training.pdf) and a PNG preview next to it.
"""
from pathlib import Path

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "runs" / "4x60_f64_unit" / "history" / "loss_history.csv"
OUT_DIR = ROOT / "runs" / "figures"

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#999999"

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
    }
)


def load_loss_history(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["iteration"], data["loss"]


def plateau_start(iteration, loss, factor=1.1):
    threshold = factor * loss[-1]
    return iteration[np.max(np.nonzero(loss > threshold)) + 1]


def main():
    iteration, loss = load_loss_history(CSV)
    start = plateau_start(iteration, loss)
    mask = iteration >= start
    span = (loss[mask].max() - loss[mask].min()) / loss[mask].min()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(3.5, 2.1), constrained_layout=True
    )

    ax_a.plot(iteration, loss, color=BLUE, lw=0.8)
    ax_a.set_yscale("log")
    ax_a.axvspan(start, iteration[-1], color=VERMILLION, alpha=0.15, lw=0)
    ax_a.annotate(
        f"plateau\n(epochs $\\geq$ {int(start)})",
        xy=(0.98, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        color=VERMILLION,
        fontsize=8,
    )
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Training loss")
    ax_a.set_xlim(0, iteration[-1])

    ax_b.plot(iteration[mask], loss[mask], color=BLUE, lw=0.8)
    ax_b.set_yscale("log")
    ax_b.set_ylim(loss[mask].min() * 0.9995, loss[mask].max() * 1.0005)
    ax_b.yaxis.set_major_locator(
        plt.FixedLocator([loss[mask].min(), 0.5 * (loss[mask].min() + loss[mask].max()), loss[mask].max()])
    )
    ax_b.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v * 1e4:.3f}")
    )
    ax_b.yaxis.set_minor_formatter(plt.NullFormatter())
    ax_b.yaxis.set_minor_locator(plt.NullLocator())
    ax_b.set_ylabel(r"Loss ($\times 10^{-4}$)")
    ax_b.annotate(
        f"variation {span * 100:.2f}%",
        xy=(0.05, 0.9),
        xycoords="axes fraction",
        ha="left",
        va="top",
        color="0.2",
        fontsize=8,
    )
    ax_b.set_xlabel("Epoch (zoom)")

    for ax in (ax_a, ax_b):
        ax.tick_params(width=0.6, length=3)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"plateau_training.{ext}", dpi=300)
    print(f"plateau starts at epoch {int(start)}, variation {span * 100:.2f}%")


if __name__ == "__main__":
    main()
