"""One page comparing several finished runs of the same problem."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = ("tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange")


def last_snapshot(run: Path) -> pd.DataFrame:
    return pd.read_csv(sorted((run / "breakdown").glob("epoch_*.csv"))[-1]).sort_values("t")


def write_page(run_dirs: list[Path], out: Path | None = None) -> Path:
    runs = [Path(r) for r in run_dirs]
    snaps = {r.name: last_snapshot(r) for r in runs}
    losses = {r.name: pd.read_csv(r / "history" / "loss_history.csv") for r in runs}
    colors = dict(zip(snaps, COLORS))
    ref = next(iter(snaps.values()))

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, var in zip(axes[0], "xyz"):
        ax.plot(ref.t, ref[f"ref_{var}"], "k-", lw=2.5, label="reference")
        for name, s in snaps.items():
            ax.plot(s.t, s[var], color=colors[name], lw=1.2, label=name)
        ax.set_title(f"{var}(t)"); ax.set_xlabel("t"); ax.grid(alpha=.3)
    axes[0][0].legend(fontsize=8)
    ax = axes[1][0]
    for name, s in snaps.items():
        ax.semilogy(s.t, s.err_norm, color=colors[name], lw=1, label=name)
    ax.set_title("|u(t) - reference|"); ax.set_xlabel("t"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax = axes[1][1]
    for name, s in snaps.items():
        ax.semilogy(s.t, np.sqrt(s.r_sq), color=colors[name], lw=1, label=name)
    ax.set_title("|residual| = |du/dt - f(u)|"); ax.set_xlabel("t"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax = axes[1][2]
    for name, h in losses.items():
        ax.semilogy(h.iteration, h.loss, color=colors[name], lw=1, label=name)
    ax.set_title("training loss"); ax.set_xlabel("iteration"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("  vs  ".join(snaps))
    fig.tight_layout()
    path = Path(out) if out else runs[-1].parent / f"compare_{'_vs_'.join(snaps)}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
