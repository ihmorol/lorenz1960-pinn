"""Train the Lorenz-1960 PINN, evaluate against the locked baseline, save results."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from torch import Tensor

from . import figures
from .config import Config, compute_error_metrics, reference_at, reference_trajectory
from .history import FLOAT_FORMAT, SnapshotWriter, TrainHistory, flat_params, residual_grid
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

    if cfg.snapshot_every > 0:
        grid_t = grid.detach().cpu().numpy().reshape(-1)
        history.snapshots = SnapshotWriter(
            cfg.results_path / "breakdown", grid_t, reference_at(cfg, grid_t)
        )

    if cfg.print_every:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[setup] device={device.type} | seed={cfg.seed} | {cfg.label}", flush=True)
        print(f"[setup] {n_params} params | {cfg.n_collocation} collocation pts on "
              f"t in [{cfg.t_span[0]}, {cfg.t_span[1]}] | ref traj {len(t_ref)} pts", flush=True)
        print(f"[setup] {cfg.epochs} Adam epochs (lr {cfg.lr_start:.1e} -> {cfg.lr_end:.1e}) "
              f"+ {cfg.lbfgs_iters} L-BFGS iters", flush=True)
        if history.snapshots is not None:
            n_snap = cfg.epochs // cfg.snapshot_every + 1
            print(f"[setup] per-point snapshots every {cfg.snapshot_every} epochs "
                  f"(~{n_snap} files) -> {history.snapshots.dir}", flush=True)
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
        res, ic, parts = loss_terms(model, grid.clone().requires_grad_(True))
        loss = res + model.gamma * ic if cfg.ic == "soft" else res
        loss.backward()

        # Snapshot before the step, so the recorded residuals are exactly the ones
        # this epoch's loss and gradient were computed from. Costs one CSV write;
        # the tensors are already in hand.
        if history.snapshots is not None and (epoch % cfg.snapshot_every == 0 or last):
            history.snapshots.write(epoch, parts)

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
    history.wall_clock_s = time.perf_counter() - t_start
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
        history.wall_clock_s = time.perf_counter() - t_start
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


LOSS_THRESHOLDS = (1e-4, 1e-6, 1e-8)


def _epochs_to(loss: list[float], threshold: float) -> float:
    """First iteration index at which the loss dropped below ``threshold``."""
    for i, value in enumerate(loss):
        if value < threshold:
            return float(i)
    return float("nan")


def run_summary(
    model: PINN, history: TrainHistory, cfg: Config, run: figures.RunArtifacts | None = None
) -> pd.DataFrame:
    """One wide row describing a finished run: config, accuracy, physics, cost.

    This is the row the architecture sweep concatenates into ``comparison.csv``.
    Every column is computed after training; nothing here feeds back into the loss.
    """
    run = run if run is not None else collect_artifacts(model, history, cfg)
    pred, ref, t = run.pred, run.ref, run.t
    err = pred - ref
    err_norm = np.linalg.norm(err, axis=1)
    metrics = run.metrics.set_index("state")

    row: dict[str, object] = {
        # --- configuration
        "arch": cfg.arch, "depth": cfg.depth, "width": cfg.width,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "activation": cfg.activation, "ic": cfg.ic, "gamma": cfg.gamma,
        "epochs": cfg.epochs, "lbfgs_iters": cfg.lbfgs_iters, "seed": cfg.seed,
        "n_collocation": cfg.n_collocation,
        "lr_start": cfg.lr_start, "lr_end": cfg.lr_end,
        "t_start": cfg.t_span[0], "t_end": cfg.t_span[1],
    }

    # --- accuracy, per state and combined
    for state in ("x", "y", "z", "combined_l2"):
        for metric in ("mae", "rmse", "max_abs_error"):
            row[f"{metric}_{state}"] = float(metrics.loc[state, metric])

    for j, name in enumerate("xyz"):
        span = float(ref[:, j].max() - ref[:, j].min())
        ss_tot = float(((ref[:, j] - ref[:, j].mean()) ** 2).sum())
        row[f"rel_mae_{name}"] = float(np.abs(err[:, j]).mean() / span) if span else float("nan")
        row[f"r2_{name}"] = 1.0 - float((err[:, j] ** 2).sum()) / ss_tot if ss_tot else float("nan")
        # Endpoint state: where the trajectory actually landed versus the truth.
        row[f"final_{name}"] = float(pred[-1, j])
        row[f"final_ref_{name}"] = float(ref[-1, j])
        row[f"final_err_{name}"] = float(err[-1, j])

    for q in (50, 90, 99):
        row[f"err_norm_p{q}"] = float(np.percentile(err_norm, q))
    row["err_norm_max"] = float(err_norm.max())

    # --- physics: does the ODE hold away from the points we trained on?
    row["final_loss"] = float(history.loss[-1]) if history.loss else float("nan")
    row["best_loss"] = float(min(history.loss)) if history.loss else float("nan")
    coll = float(np.mean(np.asarray(run.collocation_residual) ** 2)) \
        if run.collocation_residual is not None else float("nan")
    dense = float(np.mean(np.asarray(run.dense_residual) ** 2)) \
        if run.dense_residual is not None else float("nan")
    row["residual_mse_collocation"] = coll
    row["residual_mse_dense"] = dense
    row["residual_generalisation_gap"] = dense / coll if coll else float("nan")

    # --- conserved quantities
    _, drift = figures.invariant_series(pred, run.coefficients)
    _, ref_drift = figures.invariant_series(ref, run.coefficients)
    for i in range(drift.shape[1]):
        row[f"invariant_{i + 1}_max_drift"] = float(drift[:, i].max())
        row[f"invariant_{i + 1}_max_drift_reference"] = float(ref_drift[:, i].max())

    # --- cost
    wall = float(history.wall_clock_s)
    row["wall_clock_s"] = wall
    row["ms_per_epoch"] = wall / cfg.epochs * 1e3 if cfg.epochs else float("nan")
    for threshold in LOSS_THRESHOLDS:
        row[f"epochs_to_{threshold:.0e}"] = _epochs_to(history.loss, threshold)
    row["device"] = next(model.parameters()).device.type
    # CUDA-only; NaN on CPU, where torch exposes no equivalent counter.
    row["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
    )
    row["n_snapshots"] = len(history.snapshots.epochs) if history.snapshots else 0
    return pd.DataFrame([row])


def write_breakdown_figures(cfg: Config, formats=figures.FORMATS) -> dict[str, list]:
    """Render the per-epoch collocation figures from a run's ``breakdown/`` CSVs.

    Lives here rather than in :mod:`pinn.figures` because it reads the breakdown
    through :mod:`pinn.history`, and that module imports torch; ``figures`` stays
    a pure array-in, figure-out module.
    """
    out = cfg.results_path / "figures"
    epochs, centres, values = residual_grid(cfg.results_path / "breakdown")
    summary = pd.read_csv(cfg.results_path / "point_summary.csv")
    figures.set_style()

    written = {
        "residual_evolution": figures.save_figure(
            figures.fig_residual_evolution(epochs, centres, values, cfg.label),
            out, "residual_evolution", formats),
        "residual_profiles": figures.save_figure(
            figures.fig_residual_profiles(epochs, centres, values, label=cfg.label),
            out, "residual_profiles", formats),
        "point_convergence": figures.save_figure(
            figures.fig_point_convergence(summary, cfg.label),
            out, "point_convergence", formats),
    }
    html = figures.write_residual_surface_html(
        epochs, centres, values, out / "residual_surface.html", cfg.label)
    if html is not None:
        written["residual_surface_html"] = [html]
    return written


def save_results(model: PINN, history: TrainHistory, cfg: Config) -> pd.DataFrame:
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.ckpt_path / "pinn.pt")
    if cfg.print_every:
        print(f"[save]  checkpoint + telemetry -> {cfg.ckpt_path}", flush=True)
        print(f"[save]  rendering figures -> {cfg.results_path} ...", flush=True)
    run = collect_artifacts(model, history, cfg)
    metrics = figures.write_run_report(run, cfg.results_path, data_dir=cfg.ckpt_path)

    summary = run_summary(model, history, cfg, run)
    summary.to_csv(cfg.results_path / "run_summary.csv", index=False, float_format=FLOAT_FORMAT)
    if history.snapshots is not None:
        history.snapshots.finalize(cfg.results_path)
        breakdown = write_breakdown_figures(cfg)
        if cfg.print_every:
            print(f"[save]  {len(history.snapshots.epochs)} snapshots + point_summary.csv + "
                  f"{len(breakdown)} breakdown figures"
                  f"{'' if 'residual_surface_html' in breakdown else ' (no plotly: 3-D HTML skipped)'}",
                  flush=True)
    if cfg.print_every:
        print(f"[save]  wrote metrics.csv, run_summary.csv + "
              f"{len(list((cfg.results_path / 'figures').glob('*')))} figure files", flush=True)
    return metrics


def load_run(cfg: Config | None = None) -> tuple[PINN, TrainHistory, Config]:
    """Reload a finished run's trained weights and telemetry, with no retraining."""
    cfg = cfg or Config()
    device = get_device()
    model = PINN(cfg).to(device)
    model.load_state_dict(torch.load(cfg.ckpt_path / "pinn.pt", map_location=device))
    model.eval()
    return model, TrainHistory.from_saved(cfg.ckpt_path), cfg


def rebuild_figures(cfg: Config | None = None) -> pd.DataFrame:
    """Redraw every figure from a finished run's checkpoint and saved CSVs, no retraining."""
    model, history, cfg = load_run(cfg)
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
