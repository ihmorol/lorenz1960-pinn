"""Train the Lorenz-1960 PINN, evaluate against the locked baseline, save results.

Orchestration only: :func:`train` runs the loops in :mod:`pinn.functions.trainer`;
:func:`main` then persists the checkpoint, telemetry and figures via
:mod:`pinn.functions.reporting`, whose plotting calls go through :mod:`pinn.viz`.
Names from those modules are re-exported below so existing
``from pinn.train import ...`` call sites keep working unchanged.
"""
from __future__ import annotations

import time

import pandas as pd

from .config import Config  # noqa: F401  (re-exported public surface)
from .functions.reporting import (  # noqa: F401  (re-exported public surface)
    LOSS_THRESHOLDS, collect_artifacts, config_for, evaluate, load_run, predict, rebuild_figures,
    residual_at, run_summary, save_results, write_breakdown_figures,
)
from .functions.trainer import (  # noqa: F401  (re-exported public surface)
    get_device, make_grid, set_seed, train, train_windows,
)
from .history import TrainHistory
from .pinn import PINN

__all__ = [
    "train", "save_results", "write_breakdown_figures", "load_run", "rebuild_figures", "main",
    "get_device", "set_seed", "make_grid", "predict", "residual_at", "evaluate",
    "collect_artifacts", "run_summary", "config_for", "LOSS_THRESHOLDS",
]


def main(cfg: Config | None = None) -> tuple[PINN, TrainHistory, pd.DataFrame]:
    cfg = cfg or Config()
    t0 = time.perf_counter()
    model, history = train(cfg)
    metrics = save_results(model, history, cfg)
    if cfg.print_every:
        print(f"[done]  total {time.perf_counter() - t0:.1f}s", flush=True)
    print(metrics.to_string(index=False))
    return model, history, metrics


if __name__ == "__main__":
    main()
