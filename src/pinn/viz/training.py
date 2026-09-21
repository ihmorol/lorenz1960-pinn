"""Training-phase figures: what the optimiser did, epoch by epoch."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def fig_loss_phases(loss: np.ndarray, adam_iters: int, eps_marks=()):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(loss, lw=0.6, color="0.6", label="per iteration")
    ax.semilogy(pd.Series(loss).rolling(101, center=True, min_periods=1).median(), "k", lw=1.2,
                label="rolling median")
    ax.axvspan(0, adam_iters, color="tab:blue", alpha=0.06, label="Adam")
    if adam_iters < len(loss):
        ax.axvspan(adam_iters, len(loss), color="tab:orange", alpha=0.1, label="L-BFGS")
    for it, eps in eps_marks:
        ax.axvline(it, color="tab:red", lw=0.8, ls="--")
        ax.annotate(f"eps={eps:g}", (it, loss.max()), fontsize=7, rotation=90, va="top")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("loss by phase: a flat stretch is a plateau; a step down in the L-BFGS band means "
                 "Adam had stalled", fontsize=9)
    return fig


def fig_gradient_stability(epochs: np.ndarray, grads: np.ndarray):
    g = grads / np.maximum(np.linalg.norm(grads, axis=1, keepdims=True), 1e-30)
    cos = (g[1:] * g[:-1]).sum(1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs[1:], cos, "tab:blue", lw=1, label="cos(g_k, g_k-1)")
    ax.set_ylim(-1.05, 1.05); ax.set_ylabel("cosine"); ax.set_xlabel("epoch")
    ax2 = ax.twinx()
    ax2.semilogy(epochs, np.maximum(np.linalg.norm(grads, axis=1), 1e-30), "0.4", lw=0.8, label="|g|")
    ax2.set_ylabel("gradient norm")
    ax.set_title("gradient direction stability: near-zero cosine = wandering on a plateau; "
                 "steady positive = descending a valley", fontsize=9)
    fig.legend(loc="lower left", fontsize=8)
    return fig


def fig_gradient_histograms(epochs: np.ndarray, grads: np.ndarray, layer_sizes: list[tuple[str, int]]):
    picks = np.unique(np.linspace(0, len(epochs) - 1, 4).round().astype(int))
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 3.5), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], picks):
        start = 0
        for name, size in layer_sizes:
            block = grads[k, start:start + size]
            start += size
            ax.hist(np.log10(np.abs(block) + 1e-20), bins=40, histtype="step", label=name)
        ax.set_title(f"epoch {epochs[k]}"); ax.set_xlabel("log10 |grad|")
    axes[0][0].legend(fontsize=6)
    fig.suptitle("per-layer gradient magnitude: a layer whose histogram sits far left is not learning",
                 fontsize=9)
    return fig


def ntk_eigenvalues(model, t: torch.Tensor, n_sub: int = 256) -> np.ndarray:
    """Eigenvalues of J J^T for the residual on a subsample of points."""
    from ..pinn import residual_of

    idx = torch.linspace(0, len(t) - 1, min(n_sub, len(t))).long()
    tt = t[idx].detach().clone().requires_grad_(True)
    r = residual_of(model, tt).reshape(-1)
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for i in range(len(r)):
        g = torch.autograd.grad(r[i], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                               for gi, p in zip(g, params)]))
    J = torch.stack(rows)
    return torch.linalg.eigvalsh(J @ J.T).flip(0).clamp_min(0).detach().cpu().numpy()


def fig_ntk_spectrum(spectra: dict[int, np.ndarray]):
    fig, ax = plt.subplots(figsize=(7, 4))
    for epoch, ev in spectra.items():
        ax.semilogy(np.maximum(ev, 1e-20), lw=1, label=f"epoch {epoch}")
    ax.set_xlabel("eigenvalue index"); ax.set_ylabel("NTK eigenvalue"); ax.legend(fontsize=8)
    ax.set_title("NTK spectrum: a fast-decaying tail means high-frequency residual modes learn slowly",
                 fontsize=9)
    return fig
