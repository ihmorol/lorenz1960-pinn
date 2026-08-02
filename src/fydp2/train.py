"""Train the Lorenz-1960 PINN, evaluate against the locked baseline, save results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from torch import Tensor

from . import figures
from .config import Config, compute_error_metrics, reference_trajectory
from .history import TrainHistory, flat_params
from .pinn import PINN, loss_terms, pinn_loss, residual


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def make_grid(cfg: Config, device: torch.device) -> Tensor:
    # Latin hypercube sampling over [t0, tf], following Matthews & Bihlo (PinnDE).
    t0, tf = cfg.t_span
    sample = qmc.LatinHypercube(d=1, seed=cfg.seed).random(cfg.n_collocation)
    t = t0 + (tf - t0) * sample  # Linear interpolation onto [t0, tf].
    return torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)


def train(cfg: Config) -> tuple[PINN, TrainHistory]:
    set_seed(cfg.seed)
    device = get_device()
    model = PINN(cfg).to(device)
    grid = make_grid(cfg, device)
    history = TrainHistory()
    t_ref, ys_ref = reference_trajectory(cfg, n=1001)

    adam = torch.optim.Adam(model.parameters(), lr=cfg.lr_start)
    decay = cfg.lr_end / cfg.lr_start
    sched = torch.optim.lr_scheduler.LambdaLR(
        adam, lambda e: 1.0 + (decay - 1.0) * min(e, cfg.epochs) / cfg.epochs
    )

    for epoch in range(cfg.epochs):
        last = epoch == cfg.epochs - 1
        logging = epoch % cfg.log_every == 0 or last

        adam.zero_grad()
        res, ic = loss_terms(model, grid.clone().requires_grad_(True))
        loss = res + model.gamma * ic if cfg.ic == "soft" else res
        loss.backward()

        lr = adam.param_groups[0]["lr"]
        before = flat_params(model) if logging else None
        adam.step()
        sched.step()
        history.loss.append(loss.item())

        if logging:
            history.record_step(epoch, model=model, loss=loss, residual=res, ic=ic,
                                lr=lr, params_before=before)
        if epoch % cfg.eval_every == 0 or last:
            history.record_reference(epoch, float(np.mean((predict(model, t_ref) - ys_ref) ** 2)))

    history.adam_iters = len(history.loss)

    if cfg.lbfgs_iters > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), max_iter=cfg.lbfgs_iters, history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            lbfgs.zero_grad()
            loss = pinn_loss(model, grid.clone().requires_grad_(True))
            loss.backward()
            history.loss.append(loss.item())
            return loss

        lbfgs.step(closure)

    return model, history


def predict(model: PINN, t: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    tt = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
    with torch.no_grad():
        return model(tt).cpu().numpy().astype(np.float64)


def residual_at(model: PINN, t: np.ndarray) -> np.ndarray:
    """ODE residual r(t) = du/dt - f(u) evaluated at arbitrary times."""
    device = next(model.parameters()).device
    tt = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
    return residual(model, tt).detach().cpu().numpy().astype(np.float64)


def evaluate(model: PINN, cfg: Config) -> pd.DataFrame:
    t, ys = reference_trajectory(cfg, n=1001)
    return compute_error_metrics(ys, predict(model, t))


def collect_artifacts(model: PINN, history: TrainHistory, cfg: Config) -> figures.RunArtifacts:
    """Bundle a finished run into the plain-array record the figure suite reads."""
    t, ref = reference_trajectory(cfg, n=1001)
    grid = make_grid(cfg, next(model.parameters()).device).cpu().numpy().reshape(-1)
    return figures.RunArtifacts(
        t=t,
        pred=predict(model, t),
        ref=ref,
        history=history,
        coefficients=cfg.coefficients,
        collocation_t=grid,
        collocation_residual=residual_at(model, grid),
        dense_residual=residual_at(model, t),
        t_span=cfg.t_span,
        label=cfg.label,
    )


def save_results(model: PINN, history: TrainHistory, cfg: Config) -> pd.DataFrame:
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.ckpt_path / "pinn.pt")
    return figures.write_run_report(
        collect_artifacts(model, history, cfg), cfg.results_path, data_dir=cfg.ckpt_path
    )


def rebuild_figures(cfg: Config | None = None) -> pd.DataFrame:
    """Redraw every figure from a finished run's checkpoint and saved CSVs, no retraining."""
    cfg = cfg or Config()
    model = PINN(cfg).to(get_device())
    model.load_state_dict(torch.load(cfg.ckpt_path / "pinn.pt", map_location=get_device()))
    history = TrainHistory.from_saved(cfg.ckpt_path)
    return figures.write_run_report(
        collect_artifacts(model, history, cfg), cfg.results_path, data_dir=cfg.ckpt_path
    )


def main(cfg: Config | None = None) -> tuple[PINN, TrainHistory, pd.DataFrame]:
    cfg = cfg or Config()
    model, history = train(cfg)
    metrics = save_results(model, history, cfg)
    print(metrics.to_string(index=False))
    return model, history, metrics


if __name__ == "__main__":
    main()
