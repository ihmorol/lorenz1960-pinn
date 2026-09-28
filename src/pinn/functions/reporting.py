"""Finished-run reporting: metrics, the summary row, and every figure it feeds.

Collects what was previously spread across ``train.py``, ``sweep.py`` and
``viz/__init__.py``. Everything a trained model or saved run turns into after
training lives here; :mod:`pinn.viz` stays a pure array-in, figure-out leaf and
``pinn.train`` calls this module. Most names are re-exported from
``pinn.train`` so existing call sites keep working unchanged.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import viz as figures
from ..config import Config, compute_error_metrics, reference_trajectory
from ..history import FLOAT_FORMAT, TrainHistory, residual_grid
from ..pinn import PINN, build_model
from ..viz import evaluation, landscape, trajectory3d, training
from .trainer import get_device, make_grid, predict, residual_at


def evaluate(model: PINN, cfg: Config) -> pd.DataFrame:
    t, ys = reference_trajectory(cfg, n=cfg.n_eval)
    return compute_error_metrics(ys, predict(model, t))


def collect_artifacts(model: PINN, history: TrainHistory, cfg: Config) -> figures.RunArtifacts:
    """Bundle a finished run into the plain-array record the figure suite reads."""
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
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
        "arch": cfg.arch, "problem": cfg.problem, "depth": cfg.depth, "width": cfg.width,
        "n_windows": cfg.n_windows, "causal_eps_schedule": " ".join(f"{e:g}" for e in cfg.causal_eps_schedule),
        "causal_delta": cfg.causal_delta, "causal_max_iters": cfg.causal_max_iters, "warm_start": cfg.warm_start,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "activation": cfg.activation, "ic": cfg.ic, "gamma": cfg.gamma,
        "epochs": cfg.epochs, "lbfgs_iters": cfg.lbfgs_iters, "seed": cfg.seed,
        "n_collocation": cfg.n_collocation,
        "lr_start": cfg.lr_start, "lr_end": cfg.lr_end,
        "t_start": cfg.t_span[0], "t_end": cfg.t_span[1],
        "dtype": cfg.dtype, "ic_scale": cfg.ic_scale, "collocation": cfg.collocation,
        "n_eval": cfg.n_eval, "lr_decay": cfg.lr_decay if cfg.lr_decay is not None else float("nan"),
        "lr_decay_every": cfg.lr_decay_every,
        "k": cfg.k, "l": cfg.l,
        "initial_state": json.dumps(cfg.initial_state), "end_state": json.dumps(cfg.end_state),
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
    coll = float(np.mean(np.asarray(run.collocation_residual) ** 2)) \
        if run.collocation_residual is not None else float("nan")
    dense = float(np.mean(np.asarray(run.dense_residual) ** 2)) \
        if run.dense_residual is not None else float("nan")
    row["residual_mse_collocation"] = coll
    row["final_loss"] = coll
    row["final_residual_mse_full"] = coll
    row["last_logged_objective"] = float(history.loss[-1]) if history.loss else float("nan")
    comparable_history = (cfg.n_windows == 1 and not cfg.causal_eps_schedule
                          and cfg.ic == "hard" and cfg.end_state is None)
    row["best_loss"] = float(min(history.loss)) if history.loss and comparable_history else float("nan")
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
    row["adam_steps"] = history.adam_iters
    row["lbfgs_evals"] = sum(p == "lbfgs" for p in history.loss_phase) if history.loss_phase else max(
        0, len(history.loss) - history.adam_iters)
    row["capped_stages"] = (sum(not s["threshold_met"] for s in history.stage_status)
                             if history.stage_status else
                             (float("nan") if cfg.causal_eps_schedule else 0))
    row["ms_per_epoch"] = (wall / history.adam_iters * 1e3
                            if row["lbfgs_evals"] == 0 and history.adam_iters else float("nan"))
    for threshold in LOSS_THRESHOLDS:
        row[f"epochs_to_{threshold:.0e}"] = (_epochs_to(history.loss, threshold)
                                                if comparable_history and row["lbfgs_evals"] == 0
                                                else float("nan"))
    row["device"] = next(model.parameters()).device.type
    # CUDA-only; NaN on CPU, where torch exposes no equivalent counter.
    row["peak_mem_mb"] = (
        torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else float("nan")
    )
    row["n_snapshots"] = (len(history.snapshots.epochs) if history.snapshots
                          else history.saved_n_snapshots)
    return pd.DataFrame([row])


def config_for(run_dir: str | Path) -> Config:
    """Rebuild the Config a finished run was trained with, from its own summary.

    Needed to reload a checkpoint: the weights only fit a network of the same
    shape. Falls back to the defaults for runs predating ``run_summary.csv``,
    which is correct for the 4x60 run of record.
    """
    run_dir = Path(run_dir)
    default = Config()
    ckpt = default.ckpt_path if run_dir.resolve() == default.results_path.resolve() else run_dir / "history"
    paths = {"results_dir": str(run_dir), "ckpt_dir": str(ckpt)}
    manifest = ckpt / "config.json"
    if manifest.exists():
        return Config.from_record({**json.loads(manifest.read_text()), **paths})
    summary = run_dir / "run_summary.csv"
    if not summary.exists():
        return replace(Config(), **paths) if run_dir != Config().results_path else Config()

    row = pd.read_csv(summary).iloc[0]
    return replace(
        Config(), **paths,
        depth=int(row["depth"]), width=int(row["width"]), activation=str(row["activation"]),
        ic=str(row["ic"]), seed=int(row["seed"]), epochs=int(row["epochs"]),
        n_collocation=int(row["n_collocation"]),
        t_span=(float(row["t_start"]), float(row["t_end"])),
        lbfgs_iters=int(row["lbfgs_iters"]), gamma=float(row["gamma"]),
        lr_start=float(row["lr_start"]), lr_end=float(row["lr_end"]),
        dtype=str(row.get("dtype", "float32")), ic_scale=str(row.get("ic_scale", "span")),
        collocation=str(row.get("collocation", "lhs")), n_eval=int(row.get("n_eval", 1001)),
        lr_decay=None if pd.isna(row.get("lr_decay", float("nan"))) else float(row["lr_decay"]),
        lr_decay_every=int(row.get("lr_decay_every", 5000)),
        problem=str(row.get("problem", "lorenz1960")), n_windows=int(row.get("n_windows", 1)),
        causal_eps_schedule=tuple(float(e) for e in str(row.get("causal_eps_schedule", "")).split()
                                  if e not in ("", "nan")),
        causal_delta=float(row.get("causal_delta", 0.99)), causal_max_iters=int(row.get("causal_max_iters", 0)),
        warm_start=bool(row.get("warm_start", False)),
        k=float(row.get("k", 2.0)), l=float(row.get("l", 1.0)),
        initial_state=tuple(json.loads(row["initial_state"])) if "initial_state" in row else (0.5, 0.75, 1.0),
        end_state=tuple(json.loads(row["end_state"])) if "end_state" in row and pd.notna(row["end_state"])
                  and row["end_state"] != "null" else None,
    )


def load_run(cfg: Config | None = None) -> tuple[PINN, TrainHistory, Config]:
    """Reload a finished run's trained weights and telemetry, with no retraining."""
    cfg = cfg or Config()
    device = get_device()
    model = build_model(cfg).to(device=device, dtype=cfg.torch_dtype)
    try:
        state = torch.load(cfg.ckpt_path / "pinn.pt", map_location=device, weights_only=True)
    except TypeError:  # older torch
        state = torch.load(cfg.ckpt_path / "pinn.pt", map_location=device)
    model.load_state_dict(state)
    return model, TrainHistory.from_saved(cfg.ckpt_path, cfg.results_path), cfg


def rebuild_figures(cfg: Config | None = None) -> pd.DataFrame:
    """Redraw every figure from a finished run's checkpoint and saved CSVs, no retraining."""
    model, history, cfg = load_run(cfg)
    return figures.write_run_report(
        collect_artifacts(model, history, cfg), cfg.results_path, data_dir=cfg.ckpt_path
    )


def write_breakdown_figures(cfg: Config, formats=figures.FORMATS) -> dict[str, list]:
    """Render the per-epoch collocation figures from a run's ``breakdown/`` CSVs."""
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


def generate_run_extras(run_dir) -> list[Path]:
    """Training- and evaluation-phase figures beyond the standard suite, from a saved run."""
    run = Path(run_dir)
    out = run / "figures"
    out.mkdir(parents=True, exist_ok=True)
    cfg = config_for(run)
    model, history, _ = load_run(cfg)
    p = next(model.parameters())
    written: list[Path] = []

    if (run / "breakdown").exists():
        written += trajectory3d.write_all(run)
    trail_path = run / "history" / "param_trail.npz"
    if trail_path.exists():
        # A full 27-network landscape/NTK is expensive and is not the causal
        # training objective. run_landscape.py remains an explicit diagnostic.
        if cfg.n_windows == 1:
            written += landscape.write_all(run)
        trail = np.load(trail_path)
        written += figures.save_figure(training.fig_gradient_stability(trail["epochs"], trail["grads"], history.window_marks),
                                       out, "gradient_stability", ("png",))
        sizes = [(n, p.numel()) for n, p in model.named_parameters()]
        written += figures.save_figure(training.fig_gradient_histograms(trail["epochs"], trail["grads"], sizes),
                                       out, "gradient_histograms", ("png",))
        if cfg.n_windows == 1:
            grid = make_grid(cfg, next(model.parameters()).device)
            spectra = {}
            for k in np.unique(np.linspace(0, len(trail["epochs"]) - 1, 4).round().astype(int)):
                torch.nn.utils.vector_to_parameters(
                    torch.as_tensor(trail["params"][k], dtype=p.dtype, device=p.device), model.parameters())
                spectra[int(trail["epochs"][k])] = training.ntk_eigenvalues(model, grid)
            written += figures.save_figure(training.fig_ntk_spectrum(spectra), out, "ntk_spectrum", ("png",))
            model, history, _ = load_run(cfg)

    loss = np.asarray(history.loss)
    written += figures.save_figure(training.fig_loss_phases(loss, history.adam_iters, history.eps_marks,
                                                            history.window_marks, history.loss_phase),
                                   out, "loss_phases", ("png",))
    if history.min_w:
        written += figures.save_figure(training.fig_min_w(np.asarray(history.min_w), cfg.causal_delta,
                                                          history.eps_marks, history.window_marks), out, "min_w", ("png",))
        if trail_path.exists() and "weights" in np.load(trail_path):
            trail = np.load(trail_path)
            written += figures.save_figure(training.fig_causal_weights(trail["epochs"], trail["weights"]),
                                           out, "causal_weights", ("png",))
    if history.window_marks:
        written += figures.save_figure(training.fig_window_grid(loss, history.window_marks), out, "window_grid", ("png",))
    if hasattr(model, "edges"):
        written += figures.save_figure(evaluation.fig_joint_continuity(model, model.edges), out, "joint_continuity", ("png",))
    t, ref = reference_trajectory(cfg, n=cfg.n_eval)
    pred = predict(model, t)
    joints = getattr(model, "edges", [])[1:-1]
    written += figures.save_figure(evaluation.fig_error_vs_t(t, pred, ref, residual_at(model, t), joints),
                                   out, "error_vs_t", ("png",))
    written += figures.save_figure(evaluation.fig_error_growth(t, pred, ref), out, "error_growth", ("png",))
    return written


def precision_floor(run32, run64, out) -> list[Path]:
    from dataclasses import asdict

    a, b = asdict(config_for(run32)), asdict(config_for(run64))
    for record in (a, b):
        for key in ("dtype", "results_dir", "ckpt_dir", "runs_dir"):
            record.pop(key)
    if a != b:
        raise ValueError("precision comparison requires matching run configurations except dtype")
    l32 = pd.read_csv(Path(run32) / "history" / "loss_history.csv").loss.to_numpy()
    l64 = pd.read_csv(Path(run64) / "history" / "loss_history.csv").loss.to_numpy()
    return figures.save_figure(evaluation.fig_precision_floor(l32, l64), Path(out), "precision_floor", ("png",))


def save_results(model: PINN, history: TrainHistory, cfg: Config) -> pd.DataFrame:
    if not history.loss:
        raise ValueError("cannot report a run with no optimizer steps or evaluations")
    cfg.ensure_record()
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.ckpt_path / "pinn.pt")
    if history.param_trail:
        extra = {}
        if history.weight_profiles:
            width = max(len(w) for w in history.weight_profiles)
            extra["weights"] = np.stack([np.pad(w, (0, width - len(w)), constant_values=np.nan)
                                         for w in history.weight_profiles])
        np.savez_compressed(cfg.ckpt_path / "param_trail.npz", epochs=np.asarray(history.param_epochs),
                            params=np.stack(history.param_trail), grads=np.stack(history.grad_trail), **extra)
    if history.min_w:
        pd.DataFrame({"iteration": range(len(history.min_w)), "min_w": history.min_w}).to_csv(
            cfg.ckpt_path / "causal.csv", index=False)
    if history.window_marks or history.eps_marks:
        rows = [(i, e, "eps") for i, e in history.eps_marks] + \
               [(i, float("nan"), "window") for i in history.window_marks]
        pd.DataFrame(rows, columns=["iteration", "eps", "kind"]).to_csv(cfg.ckpt_path / "marks.csv", index=False)
    if history.stage_status:
        pd.DataFrame(history.stage_status).to_csv(cfg.ckpt_path / "stages.csv", index=False)
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
        generate_run_extras(cfg.results_path)
        if cfg.print_every:
            print(f"[save]  {len(history.snapshots.epochs)} snapshots + point_summary.csv + "
                  f"{len(breakdown)} breakdown figures"
                  f"{'' if 'residual_surface_html' in breakdown else ' (no plotly: 3-D HTML skipped)'}",
                  flush=True)
    if cfg.print_every:
        print(f"[save]  wrote metrics.csv, run_summary.csv + "
              f"{len(list((cfg.results_path / 'figures').glob('*')))} figure files", flush=True)
    return metrics
