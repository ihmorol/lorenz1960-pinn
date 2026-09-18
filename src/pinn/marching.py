"""Sequential / time-marching PINN: solve the time window one stage at a time.

Instead of one network satisfying the ODE over the whole of ``[t0, tf]`` with a
single initial condition at ``t0``, split the window into ``n_segments`` stages.
Stage ``k`` covers ``[t_k, t_{k+1}]`` and is trained with a hard initial condition
at ``t_k`` — the state the *previous stage ended on*. Train stage 0, freeze its
endpoint, hand that value to stage 1, and march forward.

Three details are where this goes wrong, so they are explicit here:

* Inside stage ``k`` the trial solution is ``u(t) = v_k + g_k(t) N(t)`` with
  ``g_k(t) = (t - t_k)/(t_{k+1} - t_k)``, so ``u(t_k) = v_k`` exactly — the same
  hard-IC construction as the single-domain problem, applied per stage.
* ``v_k`` is the network's own output at ``t_k`` under *stage k-1's* transform,
  detached and frozen for the whole of stage ``k``'s training. It is never the
  reference solution: marching with the true endpoint state would be circular.
  The truth enters only in evaluation, as everywhere else in this project.
* Because ``v_k`` is frozen while the weights keep moving, the piecewise solution
  develops a **jump** at each boundary. :func:`boundary_jumps` measures it, and it
  decides whether marching worked: the global error cannot beat the seams you
  stitched into the answer.

``MarchingConfig.finetune_epochs > 0`` makes the anchors trainable and optimises
all stages jointly for that many epochs, letting them close their own seams.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from torch import Tensor

from .config import Config, compute_error_metrics, reference_trajectory
from .history import FLOAT_FORMAT, TrainHistory
from .pinn import PINN, pinn_loss, residual_parts
from .train import get_device, set_seed


@dataclass(frozen=True)
class MarchingConfig(Config):
    """A :class:`Config` whose window is solved as a sequence of stages.

    ``epochs`` is epochs *per stage*, so the budget is ``n_segments * epochs``.
    Matching the single-domain epoch count would not match the compute, which is
    why :attr:`total_epochs` is what belongs in any cost comparison.
    """

    n_segments: int = 8
    epochs: int = 2000
    finetune_epochs: int = 0      # 0 = strictly sequential, stages never revisited

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_segments < 1:
            raise ValueError("n_segments must be at least 1")
        if self.finetune_epochs < 0:
            raise ValueError("finetune_epochs cannot be negative")
        if self.ic == "soft" and self.finetune_epochs > 0:
            raise ValueError(
                "ic='soft' with finetune_epochs > 0 is not supported: the IC penalty "
                "would be measured against an anchor the fine-tune is free to move. "
                "Use ic='hard'."
            )

    @property
    def total_epochs(self) -> int:
        return self.n_segments * self.epochs + self.finetune_epochs

    @property
    def label(self) -> str:
        return (f"{self.depth}x{self.width} {self.activation}, {self.n_segments}-stage "
                f"marching, {self.n_collocation} LHS points")

    def edges(self) -> np.ndarray:
        return np.linspace(self.t_span[0], self.t_span[1], self.n_segments + 1)


def segment_sizes(cfg: MarchingConfig) -> np.ndarray:
    """Collocation points per stage, summing to ``n_collocation`` exactly."""
    base, extra = divmod(cfg.n_collocation, cfg.n_segments)
    return np.array([base + (k < extra) for k in range(cfg.n_segments)])


def segment_grid(cfg: MarchingConfig, k: int, device: torch.device | str) -> Tensor:
    """Latin-hypercube points inside stage ``k``.

    Seeding per stage keeps the design fixed across runs. Each stage takes a
    uniform share of the points over a uniform share of the window, so the union
    has the same density in ``t`` as the single-domain LHS — which is what makes
    the two comparable.
    """
    a, b = float(cfg.edges()[k]), float(cfg.edges()[k + 1])
    n = int(segment_sizes(cfg)[k])
    t = a + (b - a) * qmc.LatinHypercube(d=1, seed=cfg.seed + 1 + k).random(n)
    return torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)


class SegmentedPINN(PINN):
    """A :class:`PINN` whose hard IC is anchored at the current stage's start.

    The network is shared across stages; only the trial transform changes, so the
    piecewise solution comes from one set of weights — ``anchors[k]`` holds stage
    ``k``'s initial state. ``forward`` uses whichever stage is active, so a bare
    ``model(t)`` is only meaningful for ``t`` inside that stage; go through
    :func:`predict_marching`, which routes each point to its own stage.

    ``anchors`` is an ``nn.Parameter``, not a buffer: a buffer with
    ``requires_grad=True`` never reaches ``nn.Module.parameters()`` and so would
    never be updated, making the fine-tune a silent no-op. While it is frozen
    (``requires_grad=False``) Adam simply skips it.
    """

    def __init__(self, cfg: MarchingConfig) -> None:
        super().__init__(cfg)
        self.n_segments = int(cfg.n_segments)
        self.segment_index = 0
        # A buffer, so it follows .to(device) and stays out of parameters().
        self.register_buffer("edges", torch.as_tensor(cfg.edges(), dtype=torch.float32))
        anchors = torch.zeros(self.n_segments, 3)
        anchors[0] = torch.tensor(cfg.initial_state, dtype=torch.float32)
        self.anchors = torch.nn.Parameter(anchors, requires_grad=False)

    def activate(self, k: int) -> None:
        """Point the trial transform at stage ``k`` without touching its anchor."""
        self.segment_index = int(k)
        self.t0 = float(self.edges[k])
        self.tf = float(self.edges[k + 1])

    def enter_segment(self, k: int, anchor: Tensor) -> None:
        """Begin stage ``k`` with ``anchor`` as its initial condition.

        ``u0`` is kept in step so that :func:`pinn.pinn.pinn_loss` — and its
        soft-IC penalty against ``u0`` — stays correct without a second copy of
        the loss here.
        """
        self.activate(k)
        value = anchor.detach().to(self.anchors.device).reshape(3)
        with torch.no_grad():
            self.anchors[k].copy_(value)
            self.u0.copy_(value)

    def trial(self, t: Tensor, n: Tensor) -> Tensor:
        if self.ic == "hard":
            g = (t - self.t0) / (self.tf - self.t0)
            return self.anchors[self.segment_index] + g * n
        return n


def _stage_of(model: SegmentedPINN, t: np.ndarray) -> np.ndarray:
    """Stage index per point. Boundaries belong to the stage that starts there."""
    edges = model.edges.cpu().numpy()
    return np.clip(np.searchsorted(edges, t, side="right") - 1, 0, model.n_segments - 1)


def _staged(model: SegmentedPINN, t: np.ndarray, fn) -> np.ndarray:
    """Apply ``fn(model, t_k)`` to each stage's slice, writing results in place."""
    t = np.asarray(t, dtype=float).reshape(-1)
    which, out = _stage_of(model, t), np.empty((t.size, 3))
    for k in range(model.n_segments):
        mask = which == k
        if mask.any():
            model.activate(k)
            out[mask] = fn(model, t[mask])
    return out


def _forward_at(model: SegmentedPINN, t: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    tt = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        return model(tt).cpu().numpy()


def _residual_at(model: SegmentedPINN, t: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    tt = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
    return residual_parts(model, tt.requires_grad_(True)).r.detach().cpu().numpy()


def predict_marching(model: SegmentedPINN, t: np.ndarray) -> np.ndarray:
    """Assemble the global solution, routing each ``t`` to its own stage.

    Necessary because stage ``k``'s transform is only valid inside stage ``k``:
    the ramp ``g_k`` extrapolates outside it.
    """
    return _staged(model, t, _forward_at)


def residual_at_marching(model: SegmentedPINN, t: np.ndarray) -> np.ndarray:
    """ODE residual of the piecewise solution, staged point by point."""
    return _staged(model, t, _residual_at)


def _stage_anchor(model: SegmentedPINN, k: int) -> Tensor:
    """The initial state stage ``k`` starts from, for ``k >= 1``.

    The previous stage's right endpoint under the previous stage's own transform,
    where ``g = 1``, so the value is exactly ``v_{k-1} + N(t_k)``. Detached:
    gradients through the marching chain would just rebuild the single-domain
    problem with extra machinery.
    """
    model.activate(k - 1)
    with torch.no_grad():
        return model(model.edges[k].reshape(1, 1))[0].detach()


def boundary_jumps(model: SegmentedPINN) -> pd.DataFrame:
    """The discontinuity at each stage boundary, in raw state units.

    Stage ``k``'s transform extrapolated to its right end gives
    ``v_k + N(t_{k+1})``; stage ``k+1`` starts from the frozen ``v_{k+1}``, which
    was that same quantity *before* stage ``k+1`` trained. The gap is error the
    piecewise solution cannot escape, so it is the first thing to check.
    """
    rows = []
    for k in range(model.n_segments - 1):
        model.activate(k)
        with torch.no_grad():
            right = model(model.edges[k + 1].reshape(1, 1)).detach()[0]
        left = model.anchors[k + 1].detach()
        rows.append({
            "boundary": k + 1, "t": float(model.edges[k + 1]),
            "jump_x": float((right[0] - left[0]).abs()),
            "jump_y": float((right[1] - left[1]).abs()),
            "jump_z": float((right[2] - left[2]).abs()),
            "jump_max": float((right - left).abs().max()),
        })
    return pd.DataFrame(rows)


def _finetune(model: SegmentedPINN, cfg: MarchingConfig, history: TrainHistory,
              device: torch.device) -> None:
    """Joint optimisation with the interior anchors free, so stages can close seams.

    Anchor 0 is held fixed. Freeing it lets the optimiser move the initial
    condition, and since the loss is the ODE residual and nothing else, it will:
    the stage then satisfies the physics almost exactly while solving a *different*
    initial value problem, which looks like a 1e-7 residual and a 0.3 solution
    error at the same time. Only the interior boundaries are seams to be closed;
    the first anchor is the problem statement.
    """
    model.anchors.requires_grad_(True)
    model.anchors.register_hook(
        lambda g: torch.cat([torch.zeros_like(g[:1]), g[1:]]))
    adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_end)
    grids = [segment_grid(cfg, k, device) for k in range(cfg.n_segments)]
    for _ in range(cfg.finetune_epochs):
        adam.zero_grad()
        total = sum((_staged_loss(model, k, g) for k, g in enumerate(grids)),
                    torch.zeros((), device=device))
        (total / cfg.n_segments).backward()
        adam.step()
        history.loss.append(float(total.detach()) / cfg.n_segments)
    with torch.no_grad():          # undo any float drift on the pinned anchor
        model.anchors[0].copy_(torch.tensor(cfg.initial_state, dtype=model.anchors.dtype))


def _staged_loss(model: SegmentedPINN, k: int, grid: Tensor) -> Tensor:
    model.activate(k)
    return pinn_loss(model, grid.clone().requires_grad_(True))


def train_marching(
    cfg: MarchingConfig,
) -> tuple[SegmentedPINN, TrainHistory, dict[str, pd.DataFrame]]:
    """Train stage by stage through ``cfg.t_span``, then optionally fine-tune jointly.

    Returns the model, the telemetry, and the per-stage and boundary tables.
    """
    set_seed(cfg.seed)
    device = get_device()
    model = SegmentedPINN(cfg).to(device)
    history = TrainHistory()
    t_ref, ys_ref = reference_trajectory(cfg, n=1001)
    sizes = segment_sizes(cfg)
    started = time.perf_counter()
    rows: list[dict[str, float]] = []

    for k in range(cfg.n_segments):
        anchor = torch.tensor(cfg.initial_state, dtype=torch.float32) if k == 0 \
            else _stage_anchor(model, k).cpu()
        model.enter_segment(k, anchor)
        grid = segment_grid(cfg, k, device)

        adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_start)
        decay = cfg.lr_end / cfg.lr_start
        sched = torch.optim.lr_scheduler.LambdaLR(
            adam, lambda e: 1.0 + (decay - 1.0) * min(e, cfg.epochs) / max(1, cfg.epochs))

        for epoch in range(cfg.epochs):
            adam.zero_grad()
            loss = pinn_loss(model, grid.clone().requires_grad_(True))
            loss.backward()
            adam.step()
            sched.step()
            history.loss.append(float(loss.detach()))
            if cfg.eval_every and epoch % cfg.eval_every == 0:
                history.record_reference(len(history.loss) - 1, float(np.mean(
                    (predict_marching(model, t_ref) - ys_ref) ** 2)))

        if cfg.lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(
                model.parameters(), max_iter=cfg.lbfgs_iters, history_size=50,
                tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe")

            def lbfgs_closure() -> Tensor:
                lbfgs.zero_grad()
                loss = pinn_loss(model, grid.clone().requires_grad_(True))
                loss.backward()
                history.loss.append(float(loss.detach()))
                return loss

            lbfgs.step(lbfgs_closure)

        a, b = float(model.edges[k]), float(model.edges[k + 1])
        mask = (t_ref >= a) & (t_ref <= b if k == cfg.n_segments - 1 else t_ref < b)
        err = predict_marching(model, t_ref[mask]) - ys_ref[mask]
        model.activate(k)
        with torch.no_grad():
            endpoint = model(torch.tensor([[b]], device=device)).detach()[0]
        rows.append({
            "stage": k, "t_start": a, "t_end": b, "n_points": int(sizes[k]),
            "anchor_x": float(anchor[0]), "anchor_y": float(anchor[1]), "anchor_z": float(anchor[2]),
            "final_loss": history.loss[-1], "endpoint_x": float(endpoint[0]),
            "endpoint_y": float(endpoint[1]), "endpoint_z": float(endpoint[2]),
            "stage_rmse": float(np.sqrt((err ** 2).mean())),
        })

    if cfg.finetune_epochs > 0:
        _finetune(model, cfg, history, device)
    model.anchors.requires_grad_(False)

    history.adam_iters = len(history.loss)
    history.wall_clock_s = time.perf_counter() - started
    frames = {"stages": pd.DataFrame(rows), "boundary_jumps": boundary_jumps(model)}
    return model, history, frames


def marching_summary(model: SegmentedPINN, history: TrainHistory,
                     frames: dict[str, pd.DataFrame], cfg: MarchingConfig) -> pd.DataFrame:
    """One row of the numbers that mean the same thing in both schemes."""
    device = next(model.parameters()).device
    t, ref = reference_trajectory(cfg, n=1001)
    pred = predict_marching(model, t)
    err = pred - ref
    metrics = compute_error_metrics(ref, pred).set_index("state")
    jumps = frames["boundary_jumps"]

    row: dict[str, object] = {
        "scheme": "marching",
        "n_segments": cfg.n_segments,
        "epochs": len(history.loss),
        "wall_clock_s": float(history.wall_clock_s),
        "final_loss": float(history.loss[-1]) if history.loss else np.nan,
        "best_loss": float(min(history.loss)) if history.loss else np.nan,
        "max_boundary_jump": float(jumps["jump_max"].max()) if len(jumps) else 0.0,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "err_norm_max": float(np.linalg.norm(err, axis=1).max()),
        "residual_mse_dense": float(np.mean(residual_at_marching(model, t) ** 2)),
        "device": device.type,
    }
    for state in ("x", "y", "z", "combined_l2"):
        for metric in ("mae", "rmse", "max_abs_error"):
            row[f"{metric}_{state}"] = float(metrics.loc[state, metric])
    for j, name in enumerate("xyz"):
        span = float(ref[:, j].max() - ref[:, j].min())
        row[f"rel_mae_{name}"] = float(np.abs(err[:, j]).mean() / span) if span else np.nan
    return pd.DataFrame([row])


def write_marching_tables(frames: dict[str, pd.DataFrame], summary: pd.DataFrame,
                          outdir: Path | str) -> list[Path]:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in list(frames.items()) + [("marching_summary", summary)]:
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False, float_format=FLOAT_FORMAT)
        written.append(path)
    return written
