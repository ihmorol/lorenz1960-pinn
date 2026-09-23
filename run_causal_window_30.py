"""Resumable, compact Colab experiment for the Lorenz-1960 interval [0, 15].

This is a long-horizon run, not a closed-orbit run. Run ``--verify-only`` on
Colab before training. Keep the repository on mounted Drive to preserve
``progress.pt`` and compact CSVs across runtime disconnects.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pinn.config import Config, reference_at
from pinn.functions.collocation import uniform_points
from pinn.functions.losses import causal_loss, mean_squared_residual
from pinn.pinn import WindowedPINN, build_model, residual_parts
from pinn.train import predict, residual_at
from pinn.viz.figures import invariant_series


ROOT = Path(__file__).resolve().parent
PREFIX_END = 13.26446
WINDOWS = 30
EFFECTIVE_POINTS = 1536
N_COLLOCATION = WINDOWS * (EFFECTIVE_POINTS - 1) + 1
DIAGNOSTIC_POINTS = 128
ADAM_CHECKPOINT_EVERY = 1000
LBFGS_CHECKPOINT_EVERY = 25
ADAM_LOG_EVERY = 100
LBFGS_LOG_EVERY = 25


def configuration(run_dir: Path) -> Config:
    return Config(
        t_span=(0.0, 15.0), n_windows=WINDOWS, n_collocation=N_COLLOCATION,
        n_eval=15001, depth=4, width=60, activation="tanh", dtype="float64",
        ic="hard", ic_scale="unit", collocation="uniform", warm_start=True,
        causal_eps_schedule=(0.01, 0.1, 1.0, 10.0), causal_delta=0.99,
        causal_max_iters=4000, lr_start=1e-3, lr_end=1e-4,
        lr_decay=0.9, lr_decay_every=1000, lbfgs_iters=750, seed=0,
        snapshot_every=0, results_dir=str(run_dir),
        ckpt_dir=str(run_dir / "history"), runs_dir=str(run_dir.parent),
    )


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    pending.replace(path)


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)


def preflight(cfg: Config, model: WindowedPINN, grid: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    assert cfg.n_collocation == 46051 and cfg.n_eval == 15001
    assert cfg.t_span == (0.0, 15.0) and cfg.n_windows == 30
    assert cfg.dtype == "float64" and cfg.ic == "hard" and cfg.ic_scale == "unit"
    assert cfg.collocation == "uniform" and cfg.warm_start
    assert cfg.causal_eps_schedule == (0.01, 0.1, 1.0, 10.0)
    assert cfg.causal_delta == 0.99 and cfg.causal_max_iters == 4000
    assert cfg.lr_decay == 0.9 and cfg.lr_decay_every == 1000 and cfg.lbfgs_iters == 750
    assert grid.dtype == torch.float64 and grid.shape == (46051, 1)
    buckets = model.window_of(grid)
    counts = torch.bincount(buckets, minlength=WINDOWS).cpu().tolist()
    assert counts == [1535] * 29 + [1536], counts
    nodes, _ = np.polynomial.legendre.leggauss(DIAGNOSTIC_POINTS)
    training, diagnostic = [], []
    for k in range(WINDOWS):
        points = grid[buckets == k]
        if k < WINDOWS - 1:
            endpoint = torch.tensor([[model.edges[k + 1]]], dtype=grid.dtype, device=grid.device)
            points = torch.cat((points, endpoint))
        assert len(points) == EFFECTIVE_POINTS, (k, len(points))
        a, b = model.edges[k:k + 2]
        times = a + (nodes + 1.0) * (b - a) / 2.0
        diagnostic_times = torch.as_tensor(times.reshape(-1, 1), dtype=grid.dtype, device=grid.device)
        assert len(times) == DIAGNOSTIC_POINTS
        assert np.min(np.abs(times[:, None] - points[:, 0].cpu().numpy()[None, :])) > 1e-10
        training.append(points)
        diagnostic.append(diagnostic_times)
    print(f"Preflight passed: {WINDOWS} windows, {EFFECTIVE_POINTS} effective training points each; "
          f"{DIAGNOSTIC_POINTS} fixed independent Gauss diagnostic times each.", flush=True)
    return training, diagnostic


def raw_mse(sub: torch.nn.Module, points: torch.Tensor) -> float:
    with torch.enable_grad():
        residual = residual_parts(sub, points.detach().clone().requires_grad_(True)).r
        return float(mean_squared_residual(residual).detach())


def causal_objective(sub: torch.nn.Module, points: torch.Tensor, eps: float) -> tuple[torch.Tensor, float]:
    parts = residual_parts(sub, points.detach().clone().requires_grad_(True))
    objective, weights = causal_loss(parts.r, points, eps)
    return objective, float(weights.min().detach())


def versions(device: torch.device) -> dict:
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        "device": str(device),
        "gpu_model": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cpu_model": platform.processor() or Path("/proc/cpuinfo").read_text().split("model name", 1)[-1].split("\n", 1)[0].strip(" :"),
    }


def new_state(cfg: Config, model: WindowedPINN, hardware: dict) -> dict:
    return {
        "config": cfg.record(), "git_commit": git_commit(), "hardware": hardware,
        "model": model.state_dict(), "window": 0, "initialized": False,
        "phase": "adam", "stage": 0, "stage_step": 0,
        "adam_state": None, "scheduler_state": None, "lbfgs_state": None,
        "lbfgs_window_iterations": 0, "adam_updates": 0,
        "lbfgs_window_start_evals": 0,
        "lbfgs_iterations": 0, "lbfgs_closure_evaluations": 0,
        "events": [], "stages": [], "windows": [], "joints": [],
        "wall_time_s": 0.0, "complete": False,
    }


def persist(cfg: Config, model: WindowedPINN, state: dict, start: float) -> None:
    state["model"] = model.state_dict()
    state["wall_time_s"] = time.perf_counter() - start
    state["torch_rng"] = torch.get_rng_state()
    state["numpy_rng"] = np.random.get_state()
    state["cuda_rng"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    target = cfg.ckpt_path / "progress.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".tmp")
    torch.save(state, pending)
    pending.replace(target)
    write_csv(cfg.results_path / "diagnostic_history.csv", state["events"],
              ["event", "window", "phase", "epsilon", "adam_updates", "lbfgs_iterations",
               "lbfgs_closure_evaluations", "objective", "raw_diagnostic_residual_mse", "min_weight"])
    write_csv(cfg.results_path / "stage_status.csv", state["stages"],
              ["window", "epsilon", "adam_updates", "min_weight", "threshold_met", "hit_cap", "event"])
    write_csv(cfg.results_path / "window_status.csv", state["windows"],
              ["window", "t_end", "endpoint_error", "endpoint_residual_left", "adam_updates",
               "lbfgs_iterations", "lbfgs_closure_evaluations", "capped_stages", "event"])
    write_csv(cfg.results_path / "joint_status.csv", state["joints"],
              ["joint_after_window", "t", "value_jump", "left_residual", "right_residual"])


def event(state: dict, k: int, phase: str, eps: float | None, objective: float,
          diagnostic_mse: float, min_weight: float | None = None) -> None:
    state["events"].append({
        "event": state["adam_updates"] + state["lbfgs_closure_evaluations"],
        "window": k, "phase": phase, "epsilon": eps,
        "adam_updates": state["adam_updates"],
        "lbfgs_iterations": state["lbfgs_iterations"],
        "lbfgs_closure_evaluations": state["lbfgs_closure_evaluations"],
        "objective": objective, "raw_diagnostic_residual_mse": diagnostic_mse,
        "min_weight": min_weight,
    })


def at_endpoint(sub: torch.nn.Module, t: float, device: torch.device) -> tuple[np.ndarray, float]:
    point = torch.tensor([[t]], dtype=torch.float64, device=device)
    with torch.no_grad():
        state = sub(point)[0].cpu().numpy()
    return state, float(np.sqrt(3.0 * raw_mse(sub, point)))


def train(cfg: Config, model: WindowedPINN, training: list[torch.Tensor],
          diagnostic: list[torch.Tensor], run_state: dict | None, hardware: dict) -> dict:
    device = next(model.parameters()).device
    state = run_state or new_state(cfg, model, hardware)
    if run_state is not None:
        if state["config"] != cfg.record() or state["git_commit"] != git_commit():
            raise ValueError("saved configuration or Git commit differs; use the original checkout/run path")
        model.load_state_dict(state["model"])
        torch.set_rng_state(state["torch_rng"].cpu())
        np.random.set_state(state["numpy_rng"])
        if state["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        if state["hardware"]["device"] != str(device) or state["hardware"]["gpu_model"] != hardware["gpu_model"]:
            raise ValueError("resume requires the same runtime device and GPU model")
    start = time.perf_counter() - state["wall_time_s"]
    if state["complete"]:
        return state
    # Reference values are read only for the post-window endpoint report.
    reference_edges = reference_at(cfg, np.asarray(model.edges))
    persist(cfg, model, state, start)
    for k in range(state["window"], WINDOWS):
        sub = model.windows[k]
        points, diag = training[k], diagnostic[k]
        if not state["initialized"]:
            if k and cfg.warm_start:
                with torch.no_grad():
                    for parameter, old in zip(sub.net.parameters(), model.windows[k - 1].net.parameters()):
                        parameter.copy_(old)
            state["initialized"] = True
            persist(cfg, model, state, start)
        if state["phase"] == "adam":
            optimizer = torch.optim.Adam(sub.parameters(), lr=cfg.lr_start)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.lr_decay_every, cfg.lr_decay)
            if state["adam_state"] is not None:
                optimizer.load_state_dict(state["adam_state"])
                scheduler.load_state_dict(state["scheduler_state"])
            for s in range(state["stage"], len(cfg.causal_eps_schedule)):
                eps = cfg.causal_eps_schedule[s]
                step = state["stage_step"] if s == state["stage"] else 0
                while step < cfg.causal_max_iters:
                    model.zero_grad(set_to_none=True)
                    objective, minimum = causal_objective(sub, points, eps)
                    if minimum > cfg.causal_delta:
                        break  # This is the exact state passed to the next stage.
                    objective.backward()
                    optimizer.step()
                    scheduler.step()
                    step += 1
                    state["stage_step"] = step
                    state["adam_updates"] += 1
                    if step == 1 or step % ADAM_LOG_EVERY == 0:
                        post_objective, post_minimum = causal_objective(sub, points, eps)
                        event(state, k, "adam", eps, float(post_objective.detach()),
                              raw_mse(sub, diag), post_minimum)
                    if step % ADAM_CHECKPOINT_EVERY == 0:
                        state["adam_state"] = optimizer.state_dict()
                        state["scheduler_state"] = scheduler.state_dict()
                        persist(cfg, model, state, start)
                final_objective, final_minimum = causal_objective(sub, points, eps)
                met = final_minimum > cfg.causal_delta
                event(state, k, "adam_stage_end", eps, float(final_objective.detach()),
                      raw_mse(sub, diag), final_minimum)
                state["stages"].append({"window": k, "epsilon": eps, "adam_updates": step,
                                        "min_weight": final_minimum, "threshold_met": met,
                                        "hit_cap": step == cfg.causal_max_iters, "event": state["events"][-1]["event"]})
                state["stage"] = s + 1
                state["stage_step"] = 0
                state["adam_state"] = optimizer.state_dict()
                state["scheduler_state"] = scheduler.state_dict()
                persist(cfg, model, state, start)
            state["phase"] = "lbfgs"
            state["lbfgs_window_iterations"] = 0
            state["lbfgs_window_start_evals"] = state["lbfgs_closure_evaluations"]
            state["lbfgs_state"] = None
            persist(cfg, model, state, start)
        optimizer = torch.optim.LBFGS(sub.parameters(), lr=1.0, max_iter=1, max_eval=10,
                                      history_size=50, tolerance_grad=1e-12,
                                      tolerance_change=1e-16, line_search_fn="strong_wolfe")
        if state["lbfgs_state"] is not None:
            optimizer.load_state_dict(state["lbfgs_state"])
        event(state, k, "lbfgs_start", None, raw_mse(sub, points), raw_mse(sub, diag))
        for _ in range(state["lbfgs_window_iterations"], cfg.lbfgs_iters):
            before = optimizer.state.get(sub.net[0].weight, {}).get("n_iter", 0)

            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                residual = residual_parts(sub, points.detach().clone().requires_grad_(True)).r
                loss = mean_squared_residual(residual)
                loss.backward()
                state["lbfgs_closure_evaluations"] += 1
                return loss

            optimizer.step(closure)
            after = optimizer.state.get(sub.net[0].weight, {}).get("n_iter", 0)
            if after <= before:
                break
            state["lbfgs_window_iterations"] += after - before
            state["lbfgs_iterations"] += after - before
            if state["lbfgs_window_iterations"] % LBFGS_LOG_EVERY == 0:
                event(state, k, "lbfgs", None, raw_mse(sub, points), raw_mse(sub, diag))
                state["lbfgs_state"] = optimizer.state_dict()
                persist(cfg, model, state, start)
        event(state, k, "window_end", None, raw_mse(sub, points), raw_mse(sub, diag))
        right = model.edges[k + 1]
        prediction, right_residual = at_endpoint(sub, right, device)
        endpoint_error = float(np.linalg.norm(prediction - reference_edges[k + 1]))
        if k:
            previous, left_residual = at_endpoint(model.windows[k - 1], model.edges[k], device)
            incoming, incoming_residual = at_endpoint(sub, model.edges[k], device)
            state["joints"].append({"joint_after_window": k - 1, "t": model.edges[k],
                                    "value_jump": float(np.linalg.norm(incoming - previous)),
                                    "left_residual": left_residual, "right_residual": incoming_residual})
        state["windows"].append({
            "window": k, "t_end": right, "endpoint_error": endpoint_error,
            "endpoint_residual_left": right_residual,
            "adam_updates": sum(row["adam_updates"] for row in state["stages"] if row["window"] == k),
            "lbfgs_iterations": state["lbfgs_window_iterations"],
            "lbfgs_closure_evaluations": (state["lbfgs_closure_evaluations"] -
                                           state["lbfgs_window_start_evals"]),
            "capped_stages": sum(row["hit_cap"] and not row["threshold_met"] for row in state["stages"]
                                 if row["window"] == k),
            "event": state["events"][-1]["event"],
        })
        torch.save(sub.state_dict(), cfg.ckpt_path / f"window_{k:02d}.pt")
        if k < WINDOWS - 1:
            model.windows[k + 1].u0.copy_(torch.as_tensor(prediction, dtype=torch.float64, device=device).reshape(1, 3))
        state.update(window=k + 1, initialized=False, phase="adam", stage=0, stage_step=0,
                     adam_state=None, scheduler_state=None, lbfgs_state=None,
                     lbfgs_window_iterations=0, lbfgs_window_start_evals=state["lbfgs_closure_evaluations"],
                     complete=k + 1 == WINDOWS)
        persist(cfg, model, state, start)
        print(f"Window {k + 1}/{WINDOWS}: endpoint error={endpoint_error:.3e}, "
              f"Adam={state['windows'][-1]['adam_updates']}, L-BFGS={state['windows'][-1]['lbfgs_iterations']}, "
              f"wall={state['wall_time_s']:.0f}s", flush=True)
    return state


def residual_chunks(model: WindowedPINN, times: np.ndarray, block: int = 1024) -> np.ndarray:
    return np.vstack([residual_at(model, times[i:i + block])
                      for i in range(0, len(times), block)])


def evaluate_and_plot(cfg: Config, model: WindowedPINN, state: dict) -> None:
    out = cfg.results_path
    # Include the exact historical boundary while keeping exactly 15,001 fixed
    # physical times and roughly 0.001 spacing across both regions.
    t = np.r_[np.linspace(0.0, PREFIX_END, 13265),
              np.linspace(PREFIX_END, 15.0, 1737)[1:]]
    ref = reference_at(cfg, t)
    assert len(t) == 15001 and t[0] == 0 and t[-1] == 15
    pred = predict(model, t)
    residual = residual_chunks(model, t)
    basis, drift = invariant_series(pred, cfg.coefficients)
    _, ref_drift = invariant_series(ref, cfg.coefficients)
    masks = {"prefix": t <= PREFIX_END, "extension": t > PREFIX_END,
             "whole": np.ones(len(t), dtype=bool)}
    rows = []
    for name, mask in masks.items():
        err = np.linalg.norm(pred[mask] - ref[mask], axis=1)
        row = {"model": "window30", "region": name, "t_min": float(t[mask][0]),
               "t_max": float(t[mask][-1]), "n_eval": int(mask.sum()),
               "rmse_combined_l2": float(np.sqrt(np.mean(err ** 2))),
               "max_state_error": float(err.max()),
               "residual_mse_dense": float(np.mean(residual[mask] ** 2))}
        for j in range(2):
            row[f"invariant_{j + 1}_max_drift"] = float(drift[mask, j].max())
            row[f"reference_invariant_{j + 1}_max_drift"] = float(ref_drift[mask, j].max())
        rows.append(row)
    # The historical checkpoint is evaluated at exactly the new run's prefix times.
    historical = ROOT / "runs/4x60_f64_unit_win27_causal_warm/history/pinn.pt"
    historical_cfg = replace(cfg, t_span=(0.0, PREFIX_END), n_windows=27,
                             n_collocation=39793, lbfgs_iters=1500)
    old = build_model(historical_cfg).to(next(model.parameters()).device, dtype=torch.float64)
    try:
        weights = torch.load(historical, map_location=next(model.parameters()).device, weights_only=True)
    except TypeError:
        weights = torch.load(historical, map_location=next(model.parameters()).device)
    old.load_state_dict(weights)
    prefix_t, prefix_ref = t[masks["prefix"]], ref[masks["prefix"]]
    old_pred = predict(old, prefix_t)
    old_residual = residual_chunks(old, prefix_t)
    _, old_drift = invariant_series(old_pred, cfg.coefficients)
    old_error = np.linalg.norm(old_pred - prefix_ref, axis=1)
    old_row = {"model": "historical27", "region": "prefix", "t_min": float(prefix_t[0]),
               "t_max": float(prefix_t[-1]), "n_eval": len(prefix_t),
               "rmse_combined_l2": float(np.sqrt(np.mean(old_error ** 2))),
               "max_state_error": float(old_error.max()),
               "residual_mse_dense": float(np.mean(old_residual ** 2))}
    for j in range(2):
        old_row[f"invariant_{j + 1}_max_drift"] = float(old_drift[:, j].max())
        old_row[f"reference_invariant_{j + 1}_max_drift"] = float(ref_drift[masks["prefix"], j].max())
    rows.append(old_row)
    pd.DataFrame(rows).to_csv(out / "region_summary.csv", index=False)
    pd.DataFrame({"t": t, "reference_x": ref[:, 0], "reference_y": ref[:, 1],
                  "reference_z": ref[:, 2], "pred_x": pred[:, 0], "pred_y": pred[:, 1],
                  "pred_z": pred[:, 2], "error_norm": np.linalg.norm(pred - ref, axis=1),
                  "residual_norm": np.linalg.norm(residual, axis=1)}).to_csv(out / "evaluation.csv", index=False)
    torch.save(model.state_dict(), cfg.ckpt_path / "pinn.pt")
    events = pd.DataFrame(state["events"])
    fig, ax = plt.subplots(figsize=(13, 5))
    for phase, label, color in [("adam", "weighted causal objective", "tab:blue"),
                                ("adam_stage_end", "weighted causal objective", "tab:blue"),
                                ("lbfgs", "plain residual L-BFGS objective", "tab:orange"),
                                ("lbfgs_start", "plain residual L-BFGS objective", "tab:orange"),
                                ("window_end", "plain residual L-BFGS objective", "tab:orange")]:
        part = events[events.phase == phase]
        ax.scatter(part.event, np.maximum(part.objective, 1e-18), s=5, color=color,
                   label=label if phase in ("adam", "lbfgs") else None)
    ax.plot(events.event, np.maximum(events.raw_diagnostic_residual_mse, 1e-18),
            color="black", lw=0.8, alpha=0.75, label="raw residual MSE: fixed independent times")
    for row in state["stages"]:
        ax.axvline(row["event"], color="0.75", lw=0.3)
    for row in state["events"]:
        if row["phase"] == "lbfgs_start":
            ax.axvline(row["event"], color="tab:orange", ls="--", lw=0.6)
    for row in state["windows"]:
        ax.axvline(row["event"], color="0.35", lw=0.6)
    ax.set_yscale("log"); ax.set_xlabel("Adam updates + L-BFGS closure evaluations")
    ax.set_ylabel("objective / residual MSE")
    ax.set_title("Active-window objective and independent raw ODE residual; pale lines: epsilon, dark: window")
    ax.legend(loc="best"); fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures/objective_vs_diagnostic.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.semilogy(t, np.maximum(np.linalg.norm(pred - ref, axis=1), 1e-16), label="state error vs DOP853")
    ax.axvline(PREFIX_END, color="black", ls="--", lw=0.8, label="historical prefix end")
    ax.set(xlabel="time", ylabel="state-error norm", title="Long-horizon trajectory error")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figures/trajectory_error.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for j, name in enumerate("xyz"):
        axes[j].plot(t, ref[:, j], color="black", lw=1.2, label="DOP853")
        axes[j].plot(t, pred[:, j], lw=0.8, label="PINN")
        axes[j].set_ylabel(name)
    axes[0].legend(); axes[-1].set_xlabel("time"); fig.tight_layout()
    fig.savefig(out / "figures/trajectory.png", dpi=180); plt.close(fig)
    joints = pd.DataFrame(state["joints"])
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in ("left_residual", "right_residual", "value_jump"):
        ax.semilogy(joints.t, np.maximum(joints[col], 1e-18), marker=".", label=col)
    ax.set(xlabel="joint time", ylabel="norm", title="One-sided joint residuals and value continuity")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figures/joints.png", dpi=180); plt.close(fig)
    metadata = {"git_commit": git_commit(), "seed": cfg.seed, **state["hardware"],
                "wall_time_s": state["wall_time_s"], "adam_updates": state["adam_updates"],
                "lbfgs_iterations": state["lbfgs_iterations"],
                "lbfgs_closure_evaluations": state["lbfgs_closure_evaluations"],
                "incomplete_stages": [row for row in state["stages"] if not row["threshold_met"]],
                "checkpoints": [str(path.relative_to(out)) for path in sorted(cfg.ckpt_path.glob("*.pt"))],
                "reference": "DOP853 rtol=1e-10 atol=1e-12; evaluation only",
                "sampling": "15001 fixed times: 13265 on [0,13.26446], 1736 on (13.26446,15]; identical prefix times for both checkpoints",
                "invariant_drift": "relative to the t=0 invariant, including the extension region"}
    write_json(out / "metadata.json", metadata)
    lines = [
        "# Lorenz-1960 long-horizon, 30-window result", "",
        "This is one bundled configuration on [0,15], not a closed-orbit run or a causal ablation.",
        "DOP853 reference values enter evaluation and endpoint reporting only; they do not enter training or stopping.",
        "", "## Final metrics", "",
        "All models in the prefix comparison are evaluated at the same 13,265 physical times.",
        "The extension contains 1,736 times after 13.26446; the whole grid has 15,001 times.",
        "Invariant drift is relative to each model's invariant at t=0, even for the extension row.", "",
        "| Model | Region | N | Combined RMSE | Max state error | Dense residual MSE | Invariant 1 drift | Invariant 2 drift |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {model} | {region} | {n_eval} | {rmse_combined_l2:.6e} | "
                     "{max_state_error:.6e} | {residual_mse_dense:.6e} | "
                     "{invariant_1_max_drift:.6e} | {invariant_2_max_drift:.6e} |".format(**row))
    lines += [
        "", "## Provenance and optimization", "",
        f"- Seed: {cfg.seed}; Git commit: `{git_commit()}`.",
        f"- Device: {state['hardware']['device']}; GPU: {state['hardware']['gpu_model']}; CPU: {state['hardware']['cpu_model']}.",
        f"- Python {state['hardware']['python']}; PyTorch {state['hardware']['torch']}; NumPy {state['hardware']['numpy']}; "
        f"SciPy {state['hardware']['scipy']}; pandas {state['hardware']['pandas']}; matplotlib {state['hardware']['matplotlib']}.",
        f"- Wall time: {state['wall_time_s']:.1f} s; Adam updates: {state['adam_updates']}; "
        f"L-BFGS optimizer iterations: {state['lbfgs_iterations']}; "
        f"L-BFGS closure evaluations: {state['lbfgs_closure_evaluations']}.",
        f"- Stages not meeting the causal threshold: {len(metadata['incomplete_stages'])}; see `stage_status.csv`.",
        "- Checkpoints: `history/progress.pt`, `history/pinn.pt`, and `history/window_00.pt` through `history/window_29.pt`.",
        "", "## Interpretation", "",
        "This run changes horizon, window count, point density, and optimization budget as one bundle. "
        "Its difference from the historical 27-window prefix is not attributable to any single change.",
        "The raw diagnostic residual uses 128 fixed Gauss times per active window, separate from its "
        "1,536 effective uniform training points. See `figures/objective_vs_diagnostic.png` and "
        "`figures/trajectory_error.png` for distinct physics and reference-error measurements.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "runs/causal-window/long-horizon-30/seed0")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    cfg = configuration(args.run_dir.resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    model = build_model(cfg).to(device=device, dtype=cfg.torch_dtype)
    grid = torch.as_tensor(uniform_points(cfg.t_span, cfg.n_collocation), dtype=cfg.torch_dtype,
                           device=device)
    training, diagnostic = preflight(cfg, model, grid)
    if args.verify_only:
        return
    hardware = versions(device)
    cfg.ckpt_path.mkdir(parents=True, exist_ok=True)
    manifest = cfg.ckpt_path / "config.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json.dumps(cfg.record())):
            raise ValueError("existing run has a different configuration; choose another run directory")
    else:
        if (cfg.ckpt_path / "progress.pt").exists():
            raise ValueError("progress exists without config.json")
        write_json(manifest, cfg.record())
    path = cfg.ckpt_path / "progress.pt"
    if path.exists():
        try:
            state = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(path, map_location=device)
    else:
        state = None
    print(json.dumps({"git_commit": git_commit(), **hardware}, indent=2), flush=True)
    state = train(cfg, model, training, diagnostic, state, hardware)
    evaluate_and_plot(cfg, model, state)


if __name__ == "__main__":
    main()
