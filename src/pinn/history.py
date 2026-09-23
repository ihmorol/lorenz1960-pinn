"""Optimisation telemetry recorded during training, consumed by :mod:`pinn.viz`.

This module holds :class:`TrainHistory`, the sampled per-iteration optimisation
record. The bulk per-point breakdown store and the parameter/gradient trails it
points to live in :mod:`pinn.functions.telemetry` and are re-exported here so the
existing ``from pinn.history import ...`` call sites keep working unchanged.
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
    sampled every ``cfg.log_every`` / ``cfg.eval_every`` epochs so the
    instrumentation stays cheap.
    """

    loss: list[float] = field(default_factory=list)
    adam_iters: int = 0
    resumed_from: int = 0
    wall_clock_s: float = 0.0

    # Live snapshot writer for the run, when ``cfg.snapshot_every`` is enabled.
    # Not part of the CSV round trip: the breakdown files are the record.
    snapshots: "SnapshotWriter | None" = field(default=None, repr=False, compare=False)
    param_epochs: list[int] = field(default_factory=list, repr=False, compare=False)
    param_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    grad_trail: list[np.ndarray] = field(default_factory=list, repr=False, compare=False)
    eps_marks: list[tuple[int, float]] = field(default_factory=list)
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

    def record_reference(self, epoch: int, mse: float) -> None:
        self.ref_epoch.append(epoch)
        self.ref_mse.append(mse)

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
        return pd.DataFrame({"epoch": self.ref_epoch, "ref_mse": self.ref_mse})

    @classmethod
    def from_saved(cls, data_dir: Path | str) -> "TrainHistory":
        """Reload a finished run's telemetry, so figures can be redrawn without retraining."""
        out = Path(data_dir)
        diag = pd.read_csv(out / "training_diagnostics.csv")
        ref = pd.read_csv(out / "reference_error.csv")
        loss = pd.read_csv(out / "loss_history.csv")["loss"].tolist()
        layer_cols = [c for c in diag.columns if c.endswith("_grad_norm")]
        causal = out / "causal.csv"
        marks = out / "marks.csv"
        extra = {}
        if causal.exists():
            extra["min_w"] = pd.read_csv(causal)["min_w"].tolist()
        if marks.exists():
            m = pd.read_csv(marks)
            extra["eps_marks"] = [(int(i), float(e)) for i, e in zip(m["iteration"], m["eps"]) if e == e]
            extra["window_marks"] = [int(i) for i, k in zip(m["iteration"], m["kind"]) if k == "window"]
        return cls(
            **extra,
            loss=loss,
            adam_iters=int(diag["epoch"].max()) + 1,
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
        )
