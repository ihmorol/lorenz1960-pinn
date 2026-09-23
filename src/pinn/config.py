"""Central configuration and locked-baseline access for the FYDP-2 PINN."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[1]   # .../src
_REPO_ROOT = _SRC_ROOT.parent                     # repository root (anchors results/ and data/)
sys.path.insert(0, str(_SRC_ROOT / "baseline"))

from lorenz1960_baseline import (  # noqa: E402
    Lorenz1960Config,
    compute_error_metrics,
    lorenz1960_coefficients,
    solve_lorenz1960_scipy,
)

ACTIVATIONS = ("tanh", "relu", "sigmoid", "gelu", "swish")
IC_MODES = ("hard", "soft")


@dataclass(frozen=True)
class Config:
    k: float = 2.0
    l: float = 1.0
    initial_state: tuple[float, float, float] = (0.5, 0.75, 1.0)
    t_span: tuple[float, float] = (0.0, 1.0)
    problem: str = "lorenz1960"
    end_state: tuple[float, float, float] | None = None   # set for a two-point BVP
    n_windows: int = 1        # >1: one network per time window, chained by end state
    # Causal training (Wang, Sankaran & Perdikaris 2024, Algorithm 1): per window, run
    # Adam under each eps in turn, advancing when min_i w_i > causal_delta or after
    # causal_max_iters. Empty schedule = plain mean-squared residual.
    causal_eps_schedule: tuple[float, ...] = ()
    causal_delta: float = 0.99
    causal_max_iters: int = 0
    warm_start: bool = False
    # With n_windows > 1: reuse ONE network for every window and carry the window start
    # state instead of copying weights between per-window nets. One 4x60 net (11,283
    # params) then covers all 27 windows, where WindowedPINN would hold 27 of them.
    share_network: bool = False

    depth: int = 4
    width: int = 60
    activation: str = "tanh"

    ic: str = "hard"          # "hard" trial solution, or "soft" IC penalty (Raissi-style)
    gamma: float = 1.0        # soft-IC penalty weight (unused when ic="hard")
    ic_scale: str = "span"    # "span": g = (t - t0)/(tf - t0); "unit": g = t - t0

    epochs: int = 20000       # Adam epochs
    lbfgs_iters: int = 0   # L-BFGS fine-tuning iterations (0 disables)
    lr_start: float = 1e-3
    lr_end: float = 1e-4
    seed: int = 0

    n_collocation: int = 3000
    n_eval: int = 1001
    points_per_unit: float | None = None    # overrides n_collocation as density x window length
    eval_per_unit: float | None = None      # overrides n_eval the same way
    collocation: str = "lhs"                # "lhs" | "uniform"
    dtype: str = "float32"                  # "float32" | "float64"
    lr_decay: float | None = None           # None: linear lr_start -> lr_end; else StepLR gamma
    lr_decay_every: int = 5000

    log_every: int = 10       # epochs between gradient/loss-component diagnostics
    eval_every: int = 100     # epochs between reference-solution error evaluations
    print_every: int = 250    # epochs between console progress lines (0 = silent)

    # Epochs between full per-collocation-point snapshots. 0 disables snapshotting
    # entirely, which is the default so the run-of-record pipeline is unchanged;
    # the architecture sweep sets it to 50.
    snapshot_every: int = 0
    checkpoint_every: int = 0   # Adam epochs between resumable checkpoints (0 = off)

    results_dir: str = "src/pinn/results"
    ckpt_dir: str = "src/pinn/history"
    runs_dir: str = "runs"

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.ic not in IC_MODES:
            raise ValueError(f"ic must be one of {IC_MODES}")
        if self.ic_scale not in ("span", "unit"):
            raise ValueError("ic_scale must be 'span' or 'unit'")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.collocation not in ("lhs", "uniform"):
            raise ValueError("collocation must be 'lhs' or 'uniform'")
        span = self.t_span[1] - self.t_span[0]
        if self.points_per_unit is not None:
            object.__setattr__(self, "n_collocation", int(round(self.points_per_unit * span)))
        if self.eval_per_unit is not None:
            object.__setattr__(self, "n_eval", int(round(self.eval_per_unit * span)) + 1)

    @property
    def coefficients(self) -> np.ndarray:
        return lorenz1960_coefficients(self.k, self.l)

    @property
    def results_path(self) -> Path:
        return _resolve(self.results_dir)

    @property
    def ckpt_path(self) -> Path:
        return _resolve(self.ckpt_dir)

    @property
    def runs_path(self) -> Path:
        return _resolve(self.runs_dir)

    @property
    def spec(self):
        from .problems import PROBLEMS
        return PROBLEMS[self.problem](self.coefficients, self.initial_state, self.t_span,
                                      self.end_state, lambda t: reference_at(self, t))

    @property
    def torch_dtype(self):
        import torch
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def arch(self) -> str:
        """Directory-safe run tag: ``4x60``, plus a suffix per non-default knob."""
        tag = f"{self.depth}x{self.width}"
        if self.dtype == "float64":
            tag += "_f64"
        if self.ic_scale == "unit":
            tag += "_unit"
        if self.n_windows > 1:
            tag += f"_win{self.n_windows}"
            if self.share_network:
                tag += "_shared"
        if self.causal_eps_schedule:
            tag += "_causal"
        if self.warm_start:
            tag += "_warm"
        return tag

    @property
    def label(self) -> str:
        """Short run descriptor used in figure titles."""
        return (
            f"{self.depth}x{self.width} {self.activation}, {self.ic} IC, "
            f"{self.n_collocation} {self.collocation.upper() if self.collocation == 'lhs' else self.collocation} points"
            + (f", {self.n_windows} windows" if self.n_windows > 1 else "")
            + (", shared net" if self.share_network else "")
            + (", causal" if self.causal_eps_schedule else "")
        )


def _resolve(path: str) -> Path:
    """Anchor relative output paths to the repo root, not the caller's cwd."""
    p = Path(path)
    return p if p.is_absolute() else _REPO_ROOT / p


def _baseline(cfg: Config, n: int = 1001) -> Lorenz1960Config:
    return Lorenz1960Config(
        k=cfg.k, l=cfg.l, initial_state=cfg.initial_state, t_span=cfg.t_span, n_eval=n
    )


def reference_trajectory(cfg: Config, n: int = 1001) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth (t, [x,y,z]) from the locked SciPy baseline solver."""
    t, ys, _ = solve_lorenz1960_scipy(config=_baseline(cfg, n))
    return t, ys


def reference_at(cfg: Config, t: np.ndarray) -> np.ndarray:
    """Ground-truth states at arbitrary, possibly unsorted times.

    Integrated directly at the requested times rather than interpolated from the
    uniform grid: linear interpolation of a 1001-point trajectory carries ~1e-6
    error, the same order as the PINN error it would be used to measure.
    """
    t = np.clip(np.asarray(t, dtype=float).reshape(-1), *cfg.t_span)   # float32 grids overshoot tf
    order = np.argsort(t)
    _, ys, _ = solve_lorenz1960_scipy(config=_baseline(cfg), t_eval=t[order])
    out = np.empty_like(ys)
    out[order] = ys
    return out


__all__ = ["Config", "reference_trajectory", "reference_at", "compute_error_metrics",
           "ACTIVATIONS", "IC_MODES"]
