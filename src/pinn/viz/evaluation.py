"""Evaluation-phase figures: where and how the trained curve is wrong."""
import matplotlib.pyplot as plt
import numpy as np


def fig_error_vs_t(t, pred, ref, residual, joints=()):
    err = np.linalg.norm(pred - ref, axis=1)
    res = np.linalg.norm(residual, axis=1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(t, np.maximum(err, 1e-16), "tab:blue", lw=1, label="|u - ref|")
    ax.semilogy(t, np.maximum(res, 1e-16), "tab:red", lw=0.8, label="|residual|")
    for j in joints:
        ax.axvline(j, color="0.7", lw=0.5)
    ax.set_xlabel("t"); ax.legend(fontsize=8)
    ax.set_title("error vs residual along t: error growing while residual stays flat is the ODE "
                 "amplifying small residuals, not the network failing", fontsize=9)
    return fig


def fig_error_growth(t, pred, ref):
    err = np.linalg.norm(pred - ref, axis=1)
    m = (t > t[0] + 0.05 * (t[-1] - t[0])) & (err > 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    if m.sum() > 2:
        p = np.polyfit(np.log(t[m]), np.log(err[m]), 1)
        ax.loglog(t[m], err[m], ".", ms=2, label="|error|")
        ax.loglog(t[m], np.exp(np.polyval(p, np.log(t[m]))), "k--", label=f"fit: t^{p[0]:.2f}")
    ax.set_xlabel("t"); ax.set_ylabel("|u - ref|"); ax.legend(fontsize=8)
    ax.set_title("error growth: exponent near 1-2 is polynomial (periodic system); a curve bending up "
                 "on log-log is exponential (chaos or a broken window)", fontsize=9)
    return fig


def fig_precision_floor(loss32, loss64):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss32, lw=0.7, label="float32")
    ax.semilogy(loss64, lw=0.7, label="float64")
    ax.set_xlabel("iteration"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
    ax.set_title("loss floor by precision: if float32 flattens where float64 keeps falling, "
                 "precision was the ceiling", fontsize=9)
    return fig
