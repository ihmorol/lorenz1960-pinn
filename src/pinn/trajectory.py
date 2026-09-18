"""Per-epoch trajectory capture, and the batched-versus-iterated gradient question.

Everything here answers one question from two directions: does handing the network
all ``N_c`` collocation points in a single batched pass change what it learns,
compared with walking the points one at a time?

- :func:`record_per_epoch` trains a short run with the stock loop and reads the
  trajectory back out of the snapshot CSVs it writes, so a denser cadence can be
  had without the finished runs' 50-epoch spacing.
- :func:`load_from_breakdown` reconstructs the same thing from an already
  finished run's ``breakdown/epoch_*.csv`` files, so the 20 000-epoch sweep runs
  can be visualised without retraining at all.
- :func:`gradient_equivalence` measures the gap between the batched gradient and
  the accumulated per-point gradients.
- :func:`batch_size_comparison` trains the same model at several batch sizes on
  a matched budget of point evaluations, which is the only fair way to ask
  whether iterating buys accuracy.

None of this modifies the training loop. The two experiments build their own
optimiser steps against :mod:`pinn.pinn`, and the recording paths only read what
the existing snapshot writer already produced.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import Config
from .history import FLOAT_FORMAT, TrainHistory
from .pinn import PINN, loss_terms
from .train import get_device, make_grid, set_seed, train

STATE_NAMES = ("x", "y", "z")
PRED_COLUMNS = ("epoch", "t", "x", "y", "z", "ref_x", "ref_y", "ref_z")


@dataclass(frozen=True)
class TrajectoryData:
    """One prediction of the whole collocation batch per snapshot epoch.

    ``pred[e, i, :]`` is the network's trial solution at collocation point ``i``
    at snapshot ``e``, and ``t``/``ref`` are shared across epochs. Rows are sorted
    by ``t`` so that consecutive points can be joined into a trajectory.
    """

    epochs: np.ndarray    # (E,)   snapshot epochs, ascending
    t: np.ndarray         # (Nc,)  collocation times, ascending
    pred: np.ndarray      # (E, Nc, 3) trial solution u_T
    ref: np.ndarray       # (Nc, 3) trusted reference, same order as t
    label: str = ""
    source: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.pred.shape[0])

    @property
    def n_points(self) -> int:
        return int(self.t.size)

    def rmse(self) -> np.ndarray:
        """Per-frame RMSE against the reference, one value per snapshot epoch."""
        err = self.pred - self.ref[None, :, :]
        return np.sqrt((err ** 2).mean(axis=(1, 2)))


def _sorted(epochs, t, pred, ref, label="", source="") -> TrajectoryData:
    order = np.argsort(np.asarray(t, dtype=float))
    return TrajectoryData(
        epochs=np.asarray(epochs, dtype=int),
        t=np.asarray(t, dtype=float)[order],
        pred=np.asarray(pred, dtype=np.float32)[:, order, :],
        ref=np.asarray(ref, dtype=float)[order],
        label=label,
        source=source,
    )


def record_per_epoch(
    cfg: Config, every: int = 1, results_dir: Path | str | None = None
) -> tuple[PINN, TrainHistory, TrajectoryData]:
    """Train with the stock loop and read the trajectory back from its snapshots.

    This adds nothing to the training code: it turns on ``Config.snapshot_every``,
    runs :func:`pinn.train.train` unchanged, and parses the ``breakdown/`` CSVs
    that the existing snapshot writer produces. ``every=1`` gives one frame per
    epoch.

    The cost is disk, not compute. Every snapshot is an ``N_c``-row CSV (~1.3 MB
    at the default 3000 points), so ``every=1`` over 2000 epochs writes ~2.6 GB —
    use it for short runs. The nine runs under ``runs/`` already carry 401
    snapshots each across the full 20 000 epochs, so for those the trajectory can
    be read straight off disk with :func:`load_from_breakdown` and no training at
    all.
    """
    cfg = replace(cfg, snapshot_every=max(1, int(every)))
    if results_dir is not None:
        cfg = replace(cfg, results_dir=str(results_dir))
    model, history = train(cfg)
    data = load_from_breakdown(cfg.results_path / "breakdown", label=cfg.label)
    return model, history, data


def load_from_breakdown(
    breakdown_dir: Path | str, max_frames: int | None = None, label: str = ""
) -> TrajectoryData:
    """Rebuild a :class:`TrajectoryData` from ``breakdown/epoch_*.csv`` files.

    Reads only the eight columns the figures need, and picks the same ``t``
    ordering for the reference as for the predictions.
    """
    files = sorted(Path(breakdown_dir).glob("epoch_*.csv"))
    if not files:
        raise FileNotFoundError("no epoch_*.csv under %s" % breakdown_dir)
    if max_frames and 0 < max_frames < len(files):
        keep = np.linspace(0, len(files) - 1, max_frames).round().astype(int)
        files = [files[i] for i in np.unique(keep)]

    epochs, preds, ref = [], [], None
    for path in files:
        frame = pd.read_csv(path, usecols=list(PRED_COLUMNS))
        epochs.append(int(frame["epoch"].iloc[0]))
        preds.append(frame[list(STATE_NAMES)].to_numpy(dtype=np.float32))
        if ref is None:
            ref = frame[["ref_" + a for a in STATE_NAMES]].to_numpy(dtype=float)
        t = frame["t"].to_numpy(dtype=float)
    return _sorted(epochs, t, np.stack(preds), ref, label, str(breakdown_dir))


def save_npz(data: TrajectoryData, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, epochs=data.epochs, t=data.t, pred=data.pred, ref=data.ref,
        label=np.asarray(data.label), source=np.asarray(data.source),
    )
    return path


def load_npz(path: Path | str) -> TrajectoryData:
    with np.load(Path(path), allow_pickle=False) as z:
        return TrajectoryData(
            epochs=z["epochs"], t=z["t"], pred=z["pred"], ref=z["ref"],
            label=str(z["label"]), source=str(z["source"]),
        )


def trajectory_frame(data: TrajectoryData) -> pd.DataFrame:
    """Per-epoch scalar summary: one row per snapshot, ready for a CSV."""
    pred_last = data.pred[-1]
    return pd.DataFrame({
        "epoch": data.epochs,
        "rmse": data.rmse(),
        "max_abs_err": np.abs(data.pred - data.ref[None]).max(axis=(1, 2)),
        "pred_x_min": data.pred[:, :, 0].min(axis=1),
        "pred_x_max": data.pred[:, :, 0].max(axis=1),
        "pred_y_min": data.pred[:, :, 1].min(axis=1),
        "pred_y_max": data.pred[:, :, 1].max(axis=1),
        "pred_z_min": data.pred[:, :, 2].min(axis=1),
        "pred_z_max": data.pred[:, :, 2].max(axis=1),
        "spread_x": np.ptp(data.pred[:, :, 0], axis=1),
        "spread_y": np.ptp(data.pred[:, :, 1], axis=1),
        "spread_z": np.ptp(data.pred[:, :, 2], axis=1),
        "n_points": data.n_points,
        "is_last_frame": data.epochs == data.epochs[-1],
        "final_frame_rmse": float(np.sqrt(((pred_last - data.ref) ** 2).mean())),
    })


# --------------------------------------------------------------------------
# does batching equal iterating?
# --------------------------------------------------------------------------
def gradient_equivalence(cfg: Config | None = None, seed: int | None = None) -> dict[str, float]:
    """Compare one batched gradient against the accumulated per-point gradients.

    The training loss is a mean over the ``3 * N_c`` residual entries, and every
    entry depends on the network weights but on no other collocation point, so

        L(batch)   = (1 / N_c) * sum_i  L({point i})
        dL/dtheta  = (1 / N_c) * sum_i  dL({point i})/dtheta

    are identities, not approximations. This measures how closely float32
    arithmetic reproduces them: the two gradient vectors should agree to
    summation round-off, i.e. to roughly ``eps_f32 * sqrt(3 * N_c)``.

    Returns the two losses, the two gradient norms, and the relative
    difference between them.
    """
    cfg = cfg or Config()
    set_seed(cfg.seed if seed is None else seed)
    device = get_device()
    model = PINN(cfg).to(device)
    grid = make_grid(cfg, device)
    n = grid.shape[0]
    dtype_eps = float(torch.finfo(torch.float32).eps)

    def grads() -> torch.Tensor:
        return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()
                          if p.grad is not None])

    # --- one forward/backward over the whole batch
    model.zero_grad(set_to_none=True)
    res, ic, _ = loss_terms(model, grid.clone().requires_grad_(True))
    batched = res + model.gamma * ic if cfg.ic == "soft" else res
    batched.backward()
    g_batched = grads()
    # The batched loss is the mean over 3*N_c entries; the loop below accumulates
    # sums, so scale the batched side by N_c to compare like with like.
    sum_loss = float(res.detach()) * n + (model.gamma * float(ic.detach()) if cfg.ic == "soft" else 0.0)
    sum_grad = g_batched * n

    # --- the same thing, one collocation point at a time
    model.zero_grad(set_to_none=True)
    accumulated = 0.0
    for i in range(n):
        point = grid[i:i + 1].clone().requires_grad_(True)
        res_i = loss_terms(model, point)[0].reshape(-1).sum()
        value = float(res_i.detach())
        res_i.backward()
        accumulated += value
    if cfg.ic == "soft":
        # The IC penalty sits at a single time, so it contributes once — not once
        # per collocation point — to the sum.
        t0 = torch.full((1, 1), model.t0, dtype=grid.dtype, device=grid.device)
        ic_term = model.gamma * (model(t0) - model.u0).pow(2).mean()
        value = float(ic_term.detach())
        ic_term.backward()
        accumulated += value
    g_looped = grads()

    return {
        "n_points": float(n),
        "n_params": float(sum(p.numel() for p in model.parameters())),
        "loss_batched": float(batched.detach()),
        "loss_sum_batched": sum_loss,
        "loss_sum_looped": accumulated,
        "rel_loss_diff": abs(sum_loss - accumulated) / abs(sum_loss) if sum_loss else float("nan"),
        "grad_norm_batched": float(sum_grad.norm()),
        "grad_norm_looped": float(g_looped.norm()),
        "rel_grad_diff": float((sum_grad - g_looped).norm() / sum_grad.norm()),
        "max_abs_grad_diff": float((sum_grad - g_looped).abs().max()),
        "grad_cosine": float(torch.nn.functional.cosine_similarity(
            sum_grad[None], g_looped[None]).item()),
        "float32_eps": dtype_eps,
        "roundoff_bound": dtype_eps * float(np.sqrt(3.0 * n)),
    }


def _full_batch_loss(model: PINN, grid: torch.Tensor, cfg: Config) -> float:
    """The loss over every collocation point at once.

    This is the only metric comparable across batch sizes. A mini-batch arm's own
    training loss is measured on its current mini-batch, so an arm stepping one
    point at a time would report a one-point residual — small for reasons that
    have nothing to do with how well it has learned the trajectory.
    """
    res, ic, _ = loss_terms(model, grid.clone().requires_grad_(True))
    total = res + model.gamma * ic if cfg.ic == "soft" else res
    return float(total.detach())


def batch_size_comparison(
    cfg: Config | None = None,
    batch_sizes: tuple[int, ...] = (0, 256, 32, 1),
    point_budget: int = 3_000_000,
    max_wall_s: float = 240.0,
    n_evaluations: int = 200,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Train the same model at several batch sizes on equal point-evaluation budgets.

    Batch size ``0`` means the full collocation set in one pass — one optimiser
    step per epoch, which is what the project does. The other arms step the
    optimiser once per mini-batch, so they take many more steps for the same
    number of point evaluations. Recording the *full-batch* loss against
    cumulative point evaluations is what makes the arms comparable: the loss is
    the same quantity in every arm, and the x-axis is the compute.

    Every arm starts from the identical initialisation, so the differences are
    the optimiser schedule and nothing else.
    """
    cfg = cfg or Config()
    device = get_device()
    rows: list[dict[str, float]] = []
    timing: dict[str, float] = {}

    for size in batch_sizes:
        bs = cfg.n_collocation if size <= 0 else min(size, cfg.n_collocation)
        set_seed(cfg.seed)
        model = PINN(cfg).to(device)
        grid = make_grid(cfg, device)
        adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_start)
        order = torch.arange(cfg.n_collocation, device=device)
        epoch_seen = -1
        recorded = -1

        label = "full" if size <= 0 else str(bs)
        steps = max(1, point_budget // bs)
        stride = max(1, steps // n_evaluations)   # keep the curve to ~200 points
        start = time.perf_counter()
        evaluated = 0

        for step in range(steps):
            if bs >= cfg.n_collocation:
                # The run-of-record order, so the full-batch arm reproduces
                # exactly what train() would do.
                idx = order
            else:
                epoch = step * bs // cfg.n_collocation
                if epoch != epoch_seen:
                    epoch_seen = epoch
                    order = torch.randperm(cfg.n_collocation, device=device)
                pos = (step * bs) % cfg.n_collocation
                idx = order[pos:pos + bs]
                if idx.numel() < bs:            # wrap the tail of the epoch
                    idx = torch.cat([idx, order[:bs - idx.numel()]])

            adam.zero_grad(set_to_none=True)
            res, ic, _ = loss_terms(model, grid[idx].clone().requires_grad_(True))
            loss = res + model.gamma * ic if cfg.ic == "soft" else res
            loss.backward()
            adam.step()

            evaluated += int(idx.numel())
            if step % stride == 0:
                rows.append({"batch_size": label, "step": step,
                             "points_evaluated": evaluated,
                             "loss": _full_batch_loss(model, grid, cfg),
                             "batch_loss": float(loss.detach())})
                recorded = step

            if time.perf_counter() - start > max_wall_s:
                break

        timing[label] = time.perf_counter() - start
        if recorded != step:                       # loop ended between recordings
            rows.append({"batch_size": label, "step": step, "points_evaluated": evaluated,
                         "loss": _full_batch_loss(model, grid, cfg),
                         "batch_loss": float(loss.detach())})

    return pd.DataFrame(rows), timing


def write_experiment_tables(report: dict[str, float], frame: pd.DataFrame,
                            timing: dict[str, float], outdir: Path | str) -> dict[str, list[Path]]:
    """Persist the two experiment results as CSVs beside the run's other tables."""
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {}

    equiv = out / "batching_equivalence.csv"
    pd.DataFrame([report]).to_csv(equiv, index=False, float_format=FLOAT_FORMAT)
    written["equivalence"] = [equiv]

    cost = frame.groupby("batch_size", sort=False)["points_evaluated"].max().to_frame()
    cost["wall_clock_s"] = pd.Series(timing)
    cost["us_per_point_eval"] = cost["wall_clock_s"] / cost["points_evaluated"] * 1e6
    cost = cost.reset_index()
    path = out / "batching_cost.csv"
    cost.to_csv(path, index=False, float_format=FLOAT_FORMAT)
    written["cost"] = [path]

    curves = out / "batching_curves.csv"
    frame.to_csv(curves, index=False, float_format=FLOAT_FORMAT)
    written["curves"] = [curves]
    return written
