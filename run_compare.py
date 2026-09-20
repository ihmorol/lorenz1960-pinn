"""One page comparing two runs of the same problem: parallel vs sequential, say.

    python run_compare.py runs/t10/4x60 runs/t10/4x60_seq

Reads each run's last breakdown snapshot and its loss history and writes
<second run>/../compare_<a>_vs_<b>.png with six panels:

    1-3  x(t), y(t), z(t): each run against the reference     -> did it learn the shape?
    4    |error| vs t, log scale                              -> where along t is it wrong?
    5    |residual| vs t, log scale                           -> where is the physics violated?
    6    loss vs epoch, log scale                             -> how did training go?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def last_snapshot(run: Path) -> pd.DataFrame:
    return pd.read_csv(sorted((run / "breakdown").glob("epoch_*.csv"))[-1]).sort_values("t")


if __name__ == "__main__":
    runs = [Path(a) for a in sys.argv[1:]] or [Path("runs/t10/4x60"), Path("runs/t10/4x60_seq")]
    snaps = {r.name: last_snapshot(r) for r in runs}
    losses = {r.name: pd.read_csv(r / "history" / "loss_history.csv") for r in runs}
    colors = dict(zip(snaps, ["tab:blue", "tab:red", "tab:green"]))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ref = next(iter(snaps.values()))
    for ax, var in zip(axes[0], "xyz"):
        ax.plot(ref.t, ref[f"ref_{var}"], "k-", lw=2.5, label="reference (RK45)")
        for name, s in snaps.items():
            ax.plot(s.t, s[var], color=colors[name], lw=1.2, label=name)
        ax.set_title(f"{var}(t)"); ax.set_xlabel("t"); ax.grid(alpha=.3)
    axes[0][0].legend()

    ax = axes[1][0]
    for name, s in snaps.items():
        ax.semilogy(s.t, s.err_norm, color=colors[name], lw=1, label=name)
    ax.set_title("|u(t) - reference|   (lower is better)"); ax.set_xlabel("t"); ax.grid(alpha=.3); ax.legend()

    ax = axes[1][1]
    for name, s in snaps.items():
        r = np.sqrt(s.r_x**2 + s.r_y**2 + s.r_z**2)
        ax.semilogy(s.t, r, color=colors[name], lw=1, label=name)
    ax.set_title("|residual|  = |du/dt - f(u)|   (what the loss sees)"); ax.set_xlabel("t"); ax.grid(alpha=.3); ax.legend()

    ax = axes[1][2]
    for name, h in losses.items():
        ax.semilogy(h.iteration, h.loss, color=colors[name], lw=1, label=name)
    ax.set_title("training loss"); ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend()

    fig.suptitle(f"t in [{ref.t.min():.0f}, {ref.t.max():.0f}]  |  " + "  vs  ".join(snaps))
    fig.tight_layout()
    out = runs[-1].parent / f"compare_{'_vs_'.join(snaps)}.png"
    fig.savefig(out, dpi=120)
    print(out)
    for name, s in snaps.items():
        print(f"{name:10s} rmse {np.sqrt((s.err_norm**2).mean()):.2e}  max err {s.err_norm.max():.2e}  "
              f"err@t<1 {s.err_norm[s.t < 1].max():.2e}  err@t>9 {s.err_norm[s.t > 9].max():.2e}")
