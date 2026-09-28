"""Optimisation telemetry recorded during training, consumed by :mod:`pinn.viz`.

Holds :class:`TrainHistory`, the sampled per-iteration optimisation record. The
per-point snapshot store and its loaders live in :mod:`pinn.functions.telemetry`
and are re-exported here so ``from pinn.history import ...`` keeps working.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .functions.telemetry import (  # noqa: F401  (re-exported public surface)
    CONVERGED_RESIDUAL,
    FLOAT_FORMAT,
    SNAPSHOT_COLUMNS,
    SnapshotWriter,
    flat_grads,
    flat_params,
    grad_norms,
    load_snapshots,
    point_history,
    residual_grid,
)


@dataclass
class TrainHistory:
    """Per-iteration optimisation record.

    ``loss`` holds every iteration (Adam, then L-BFGS); the remaining series are
    sampled every ``cfg.log_every`` / ``eval_every`` epochs so the
    instrumentation stays cheap.
    """

    loss: list[float] = field(default_factory=list)
    loss_phase: list[str] = field(default_factory=list)
    loss_window: list[int] = field(default_factory=list)
    adam_iters: int = 0
    resumed_from: int = 0
    wall_clock_s: float = 0.0
    saved_n_snapshots: int = 0

    # Live snapshot writer for the run, when ``cfg.snapshot_every`` is enabled.
    # Not part of the CSV round trip: the breakdown files are the record.
    snapshots: "SnapshotWriter | None" = field(default=None, repr=False, compare=False)
    param_epochs: list[int] = field(default_factory=list, repr=False, compare=False)
    param_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    grad_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    eps_marks: list[tuple[int, float]] = field(default_factory=list)
    stage_status: list[dict] = field(default_factory=list)
    min_w: list[float] = field(default_factory=list)
    window_marks: list[int] = field(default_factory=list)
    weight_profiles: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)

    log_epoch: list[int] = field(default_factory=list)
    logged_loss: list[float] = field(default_factory=list)
    residual_loss: list[float] = field(default_factory=list)
    ic_loss: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)
    update_norm: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    layer_grad_norms: dict[str, list[float]] = field(default_factory=dict)

    ref_epoch: list[int] = field(default_factory=list)
    ref_mse: list[float] = field(default_factory=list)
    ref_until: list[float] = field(default_factory=list)

    def record_step(
        self,
        epoch: int,
        *,
        model,
        loss: Tensor,
        residual: Tensor,
        ic: Tensor,
        lr: float,
        params_before: Tensor,
    ) -> None:
        """Snapshot gradients and the step just taken. Call after ``optimizer.step()``."""
        norm, per_layer = grad_norms(model)
        self.log_epoch.append(epoch)
        self.logged_loss.append(float(loss.detach()))
        self.residual_loss.append(float(residual.detach()))
        self.ic_loss.append(float(ic.detach()))
        self.grad_norm.append(norm)
        self.lr.append(lr)
        self.update_norm.append(float((flat_params(model) - params_before).norm()))
        for name, value in per_layer.items():
            self.layer_grad_norms.setdefault(name, []).append(value)

    def record_reference(self, epoch: int, mse: float, trained_until: float | None = None) -> None:
        self.ref_epoch.append(epoch)
        self.ref_mse.append(mse)
        self.ref_until.append(float(trained_until) if trained_until is not None else float("nan"))

    def diagnostics_frame(self) -> pd.DataFrame:
        """Sampled optimiser diagnostics, one row per logged epoch."""
        return pd.DataFrame({
            "epoch": self.log_epoch,
            "loss": self.logged_loss,
            "residual_loss": self.residual_loss,
            "ic_loss": self.ic_loss,
            "grad_norm": self.grad_norm,
            "update_norm": self.update_norm,
            "lr": self.lr,
            **{name.replace(" ", "_") + "_grad_norm": v
               for name, v in self.layer_grad_norms.items()},
        })

    def reference_frame(self) -> pd.DataFrame:
        """Error against the trusted solution, sampled on its own cadence."""
        return pd.DataFrame({"epoch": self.ref_epoch, "ref_mse": self.ref_mse,
                             "trained_until": self.ref_until or [float("nan")] * len(self.ref_epoch)})

    @classmethod
    def from_saved(cls, data_dir: Path | str, results_dir: Path | str | None = None) -> "TrainHistory":
        """Reload a finished run's telemetry, so figures can be redrawn without retraining."""
        out = Path(data_dir)
        results = Path(results_dir) if results_dir is not None else out.parent
        diag = pd.read_csv(out / "training_diagnostics.csv")
        ref = pd.read_csv(out / "reference_error.csv")
        loss_frame = pd.read_csv(out / "loss_history.csv")
        loss = loss_frame["loss"].tolist()
        layer_cols = [c for c in diag.columns if c.startswith("layer_") and c.endswith("_grad_norm")]
        causal = out / "causal.csv"
        marks = out / "marks.csv"
        stages = out / "stages.csv"
        extra = {}
        if causal.exists():
            extra["min_w"] = pd.read_csv(causal)["min_w"].tolist()
        if marks.exists():
            m = pd.read_csv(marks)
            extra["eps_marks"] = [(int(i), float(e)) for i, e in zip(m["iteration"], m["eps"]) if e == e]
            extra["window_marks"] = [int(i) for i, k in zip(m["iteration"], m["kind"]) if k == "window"]
        if stages.exists():
            extra["stage_status"] = pd.read_csv(stages).to_dict("records")
        summary_path = results / "run_summary.csv"
        summary = pd.read_csv(summary_path).iloc[0] if summary_path.exists() else None
        breakdown_count = len(list((results / "breakdown").glob("epoch_*.csv")))
        return cls(
            **extra,
            loss=loss,
            loss_phase=loss_frame["phase"].tolist() if "phase" in loss_frame else [],
            loss_window=loss_frame["window"].astype(int).tolist() if "window" in loss_frame else [],
            adam_iters=(int(summary["adam_steps"]) if summary is not None and "adam_steps" in summary
                        else (sum(loss_frame["phase"] == "adam") if "phase" in loss_frame
                              else (len(extra["min_w"]) if "min_w" in extra
                                    else int(diag["epoch"].max()) + 1))),
            wall_clock_s=float(summary["wall_clock_s"]) if summary is not None else 0.0,
            saved_n_snapshots=(breakdown_count or
                               (int(summary["n_snapshots"]) if summary is not None else 0)),
            log_epoch=diag["epoch"].tolist(),
            logged_loss=diag["loss"].tolist(),
            residual_loss=diag["residual_loss"].tolist(),
            ic_loss=diag["ic_loss"].tolist(),
            grad_norm=diag["grad_norm"].tolist(),
            update_norm=diag["update_norm"].tolist(),
            lr=diag["lr"].tolist(),
            layer_grad_norms={c.replace("_grad_norm", "").replace("_", " "): diag[c].tolist()
                              for c in layer_cols},
            ref_epoch=ref["epoch"].tolist(),
            ref_mse=ref["ref_mse"].tolist(),
            ref_until=ref["trained_until"].tolist() if "trained_until" in ref else [],
        )
