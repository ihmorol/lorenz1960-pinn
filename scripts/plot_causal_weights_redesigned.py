"""Redesigned causal-weights figure from saved run data (no retraining).

Panel (a): temporal weight w(t) profiles within the active window at sampled
           training states, from history/param_trail.npz.
Panel (b): minimum temporal weight over the whole training run on the
           Adam-iteration axis (L-BFGS-only stretches, where min_w is not
           logged, are compressed out, matching fig_min_w in pinn.viz),
           with window boundaries marked.

Data source: runs/4x60_f64_unit_win27_causal_warm/ (the locally archived run
of record for the causal-window mechanism; the candidate-1536pt checkpoint
stayed on Colab, so its full profiles are unavailable).

Usage: python scripts/plot_causal_weights_redesigned.py [run_dir]
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN = Path(sys.argv[1] if len(sys.argv) > 1
           else "runs/4x60_f64_unit_win27_causal_warm")
OUT = RUN / "figures"

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})

GREEN = "#117733"   # Okabe-Ito colour-blind-safe accent
GRAY = "#bbbbbb"

# ---- panel (a): weight profiles -----------------------------------------
trail = np.load(RUN / "history" / "param_trail.npz")
epochs, W = trail["epochs"], trail["weights"]
ks = np.unique(np.linspace(0, len(epochs) - 1, 6).round().astype(int))
cmap = plt.get_cmap("viridis")
norm = plt.Normalize(vmin=0, vmax=epochs[ks[-1]])

fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                              gridspec_kw={"width_ratios": [1.05, 1]})
ends = []
for i, k in enumerate(ks):
    w = W[k][~np.isnan(W[k])]
    x = np.linspace(0, 1, len(w))
    c = cmap(norm(epochs[k]))
    lw = 2.6 if i == len(ks) - 1 else 1.9
    ax.plot(x, w, color=c, lw=lw, zorder=3)
    ends.append([w[-1], c, epochs[k]])
ax.axhline(1.0, color=GRAY, ls="--", lw=1, zorder=1)
ax.text(0.02, 1.03, "fully trusted (w = 1)", fontsize=9, color="#666666")

# right-edge epoch labels, nudged apart so they never overlap
ends.sort(key=lambda e: e[0])
min_gap = 0.055
ys = [e[0] for e in ends]
for j in range(1, len(ys)):
    ys[j] = max(ys[j], ys[j - 1] + min_gap)
shift = max(0.0, ys[0] - ends[0][0])
ys = [y - shift for y in ys]
for (w_end, c, ep), y in zip(ends, ys):
    ax.annotate(f"epoch {ep // 1000}k", xy=(1.01, y), color=c,
                fontsize=9, va="center", fontweight="bold",
                annotation_clip=False)
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.20)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xlabel("position within the time window  (0 = start, 1 = end)")
ax.set_ylabel("temporal weight  w")
ax.set_title("(a)  The gate opens as the window is solved",
             loc="left", fontweight="bold")
ax.annotate("early training: the future\nis not trusted yet", xy=(0.42, 0.40),
            xytext=(0.46, 0.62), fontsize=9.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=1))

# ---- panel (b): min weight over the whole run, Adam-iteration axis -------
causal = pd.read_csv(RUN / "history" / "causal.csv")
marks = pd.read_csv(RUN / "history" / "marks.csv")
eps_it = marks.loc[marks["kind"] == "eps", "iteration"].values
win_it = marks.loc[marks["kind"] == "window", "iteration"].values
# Adam-space position of each window end: within a window, Adam iterations
# occupy [start, adam_end] in global numbering (last eps mark), and L-BFGS
# fills the rest — so the boundary in Adam space is the cumulative Adam count.
win_adam, start, cum = [], 0, 0
for end in win_it:
    inside = eps_it[(eps_it > start) & (eps_it <= end)]
    adam_end = inside.max() if len(inside) else end
    cum += adam_end - start
    win_adam.append(cum)
    start = end

axb.plot(causal["min_w"].values, color=GREEN, lw=0.9, zorder=3)
for it in win_adam:
    axb.axvline(it, color=GRAY, lw=0.6, zorder=1)
axb.axhline(1.0, color="#888888", ls="--", lw=1, zorder=2)
axb.text(0.99, 1.03, "fully trusted", fontsize=9, color="#666666",
         ha="right", transform=axb.get_yaxis_transform())
axb.set_ylim(0, 1.12)
axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
axb.set_xlabel("Adam training iteration  (window boundaries in grey)")
axb.set_ylabel("minimum weight  min w")
axb.set_title("(b)  Every window re-opens the gate briefly",
              loc="left", fontweight="bold")
axb.annotate("new window:\ngate re-opens", xy=(win_adam[1] + 40, 0.62),
             xytext=(win_adam[2] + 1500, 0.20), fontsize=9.5, color="#555555",
             arrowprops=dict(arrowstyle="->", color="#555555", lw=1))

fig.suptitle("Causal weighting: the model earns the right to fit the future",
             fontsize=13, fontweight="bold", y=1.00)
fig.tight_layout(rect=(0, 0, 1, 0.96))
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"causal_weights_redesigned.{ext}",
                bbox_inches="tight")
print("wrote", OUT / "causal_weights_redesigned.{png,pdf}")
