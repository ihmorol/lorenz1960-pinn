"""Train the Lorenz-1960 PINN, evaluate against the locked baseline, save results."""
from __future__ import annotations

import time

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

    if cfg.print_every:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[setup] device={device.type} | seed={cfg.seed} | {cfg.label}", flush=True)
        print(f"[setup] {n_params} params | {cfg.n_collocation} collocation pts on "
              f"t in [{cfg.t_span[0]}, {cfg.t_span[1]}] | ref traj {len(t_ref)} pts", flush=True)
        print(f"[setup] {cfg.epochs} Adam epochs (lr {cfg.lr_start:.1e} -> {cfg.lr_end:.1e}) "
              f"+ {cfg.lbfgs_iters} L-BFGS iters", flush=True)
    t_start = time.perf_counter()

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

        if cfg.print_every and (epoch % cfg.print_every == 0 or last):
            elapsed = time.perf_counter() - t_start
            eta = elapsed / (epoch + 1) * (cfg.epochs - epoch - 1)
            print(f"[adam]  epoch {epoch + 1:>6}/{cfg.epochs} | loss {history.loss[-1]:.4e} | "
                  f"res {res.item():.4e} | ic {ic.item():.4e} | "
                  f"|grad| {history.grad_norm[-1]:.3e} | step {history.update_norm[-1]:.3e} | "
                  f"lr {lr:.2e} | ref_mse {history.ref_mse[-1]:.4e} | "
                  f"{elapsed:6.1f}s elapsed, ~{eta:5.0f}s left", flush=True)

    history.adam_iters = len(history.loss)
    if cfg.print_every:
        print(f"[adam]  done in {time.perf_counter() - t_start:.1f}s | "
              f"final loss {history.loss[-1]:.4e} | best loss {min(history.loss):.4e}", flush=True)

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
            n = len(history.loss) - history.adam_iters
            if cfg.print_every and n % 10 == 1:
                print(f"[lbfgs] eval {n:>5} | loss {history.loss[-1]:.4e} | "
                      f"{time.perf_counter() - t_start:6.1f}s elapsed", flush=True)
            return loss

        lbfgs.step(closure)
        if cfg.print_every:
            print(f"[lbfgs] done after {len(history.loss) - history.adam_iters} evals | "
                  f"final loss {history.loss[-1]:.4e}", flush=True)

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
    if cfg.print_every:
        print(f"[save]  checkpoint + telemetry -> {cfg.ckpt_path}", flush=True)
        print(f"[save]  rendering figures -> {cfg.results_path} ...", flush=True)
    metrics = figures.write_run_report(
        collect_artifacts(model, history, cfg), cfg.results_path, data_dir=cfg.ckpt_path
    )
    if cfg.print_every:
        print(f"[save]  wrote metrics.csv + {len(list((cfg.results_path / 'figures').glob('*')))} "
              f"figure files", flush=True)
    return metrics


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
    t0 = time.perf_counter()
    model, history = train(cfg)
    metrics = save_results(model, history, cfg)
    if cfg.print_every:
        print(f"[done]  total {time.perf_counter() - t0:.1f}s", flush=True)
    print(metrics.to_string(index=False))
    return model, history, metrics


if __name__ == "__main__":
    main()
