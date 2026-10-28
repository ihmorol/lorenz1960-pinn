#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs" / "causal-window" / "candidate-1536pt"
RUN = next(path for path in RUN_ROOT.iterdir() if path.is_dir())
OUTPUT = ROOT / "poster" / "assets" / "defense_result.svg"


def _plotly_traces(path: Path) -> list[dict]:
    source = path.read_text(encoding="utf-8")
    cursor = source.index("Plotly.newPlot(") + len("Plotly.newPlot(")
    decoder = json.JSONDecoder()
    args = source[cursor:].lstrip()
    _, end = decoder.raw_decode(args)
    args = args[end:].lstrip()
    if not args.startswith(","):
        raise ValueError("unexpected Plotly.newPlot structure")
    traces, _ = decoder.raw_decode(args[1:].lstrip())
    return traces


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    traces = _plotly_traces(RUN / "figures" / "trajectory.html")
    reference = next(trace for trace in traces if trace["name"] == "reference")
    prediction = next(trace for trace in traces if trace["name"].endswith("u(t)"))
    ref = np.column_stack([reference[key] for key in ("x", "y", "z")]).astype(float)
    pred = np.column_stack([prediction[key] for key in ("x", "y", "z")]).astype(float)
    time = np.array(
        [float(re.match(r"t=([0-9.]+)", label).group(1)) for label in prediction["text"]]
    )
    if ref.shape != pred.shape or ref.shape[0] != time.size:
        raise ValueError("saved trajectory traces have inconsistent lengths")

    summary = _rows(RUN / "run_summary.csv")[0]
    combined = next(row for row in _rows(RUN / "metrics.csv") if row["state"] == "combined_l2")
    rmse = float(summary["rmse_combined_l2"])
    maximum = float(summary["max_abs_error_combined_l2"])
    if not np.isclose(rmse, float(combined["rmse"]), rtol=1e-8):
        raise ValueError("combined RMSE differs between run_summary.csv and metrics.csv")
    if not np.isclose(maximum, float(combined["max_abs_error"]), rtol=1e-8):
        raise ValueError("maximum error differs between run_summary.csv and metrics.csv")

    error_norm = np.linalg.norm(pred - ref, axis=1)
    colors = ("#087f83", "#b97020", "#8b5fa8")
    names = ("x", "y", "z")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 18,
            "axes.titlesize": 23,
            "axes.titleweight": "bold",
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 17,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.5), sharex=True)
    fig.patch.set_facecolor("white")
    for axis in axes.flat:
        axis.set_facecolor("#f7fbfc")
        axis.grid(True, color="#c9d9e3", alpha=0.55, linewidth=0.8)
        axis.xaxis.set_major_locator(MaxNLocator(6))

    for index, (axis, name, color) in enumerate(zip(axes.flat[:3], names, colors)):
        axis.plot(time, ref[:, index], color="#aab8c2", linewidth=5.2,
                  solid_capstyle="round", label="DOP853 reference")
        axis.plot(time, pred[:, index], color=color, linewidth=2.2,
                  label="PINN prediction")
        axis.set_title(f"{name}(t)", color="#17344d", pad=8)
        axis.set_ylabel("State value")
        if index == 0:
            axis.legend(loc="upper right", frameon=False, ncol=2,
                        handlelength=2.6, columnspacing=1.2)

    error_axis = axes[1, 1]
    positive_error = error_norm[error_norm > 0]
    plotted_error = np.where(error_norm > 0, error_norm, np.nan)
    error_axis.semilogy(time, plotted_error, color="#d55e00", linewidth=2.4)
    error_axis.set_ylim(positive_error.min() * 0.65, maximum * 1.45)
    error_axis.axhline(maximum, color="#17344d", linestyle="--", linewidth=1.5,
                       label=f"saved maximum = {maximum:.2e}")
    error_axis.set_title(r"State error norm $\|\hat{u}(t)-u_{ref}(t)\|_2$",
                         color="#17344d", pad=8)
    error_axis.set_ylabel("Error norm (log scale)")
    error_axis.legend(loc="lower right", frameon=False)
    for axis in axes[1, :]:
        axis.set_xlabel("Time t")

    fig.suptitle("Candidate-1536pt: PINN prediction and DOP853 reference",
                 fontsize=29, fontweight="bold", color="#17344d", y=0.985)
    fig.text(
        0.5,
        0.012,
        (
            f"Candidate-1536pt  |  combined RMSE = {rmse:.2e}  |  "
            f"maximum state error = {maximum:.2e}  |  {int(summary['n_eval']):,} evaluation points\n"
            "Curves: saved trajectory.html (301 display samples). Aggregates: run_summary.csv, "
            "cross-checked against metrics.csv. Reference: DOP853, rtol 1e-10, atol 1e-12."
        ),
        ha="center",
        va="bottom",
        fontsize=14.5,
        color="#526b7e",
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.15,
                        wspace=0.18, hspace=0.26)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUTPUT}")
    print(f"verified combined RMSE={rmse:.9g}, max={maximum:.9g}")
    print(f"displayed samples={time.size}, displayed max error={error_norm.max():.9g}")


if __name__ == "__main__":
    main()
