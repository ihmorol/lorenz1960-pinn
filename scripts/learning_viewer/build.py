"""Build an offline viewer from saved artifacts, using the causal 1536-point checkpoint and histories.

Run: python scripts/learning_viewer/build.py
Check: python scripts/learning_viewer/build.py --check
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "baseline"))
from lorenz1960_baseline import (  # noqa: E402
    Lorenz1960Config, lorenz1960_coefficients, solve_lorenz1960_scipy,
)


def lists(values):
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite visualization data")
    return np.round(values, 7).tolist()


def build_data():
    import torch
    sys.path.insert(0, str(ROOT / "src"))
    from pinn.config import Config
    from pinn.pinn import build_model, residual_parts

    run = ROOT / "runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd"
    config_path = Path(__file__).with_name("source-config.json")
    record = json.loads(config_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(run / "run_summary.csv").iloc[0]
    if record["n_collocation"] != 41446 or record["n_windows"] != 27:
        raise ValueError("Expected the causal 1536-point-per-window case")
    if (int(summary.n_collocation), int(summary.n_windows)) != (41446, 27):
        raise ValueError("Config and saved results identify different runs")
    cfg = Config(**record)
    torch.set_num_threads(1)
    model = build_model(cfg).to(dtype=torch.float64)
    checkpoint = run / "history/pinn.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for k in range(26):
        edge = torch.tensor([[model.edges[k + 1]]], dtype=torch.float64)
        with torch.no_grad():
            assert torch.allclose(model.windows[k](edge), model.windows[k + 1].u0, atol=1e-12, rtol=0)
    baseline_cfg = Lorenz1960Config(k=cfg.k, l=cfg.l, initial_state=tuple(cfg.initial_state),
                                  t_span=tuple(cfg.t_span), n_eval=cfg.n_eval)
    dense_t, dense_ref, _ = solve_lorenz1960_scipy(config=baseline_cfg)
    dense_parts = residual_parts(model, torch.tensor(dense_t[:, None], dtype=torch.float64, requires_grad=True))
    dense_pred = dense_parts.u.detach().numpy()
    dense_r = dense_parts.r.detach().numpy()
    dense_error = np.linalg.norm(dense_pred - dense_ref, axis=1)
    measured_rmse = float(np.sqrt(np.mean(dense_error ** 2)))
    if not np.isclose(measured_rmse, float(summary.rmse_combined_l2), rtol=1e-4, atol=1e-9):
        raise ValueError(f"Checkpoint does not reproduce saved RMSE: {measured_rmse}")
    time = np.unique(np.r_[np.linspace(*cfg.t_span, 540), model.edges])
    _, reference, _ = solve_lorenz1960_scipy(config=baseline_cfg, t_eval=time)
    display_parts = residual_parts(model, torch.tensor(time[:, None], dtype=torch.float64, requires_grad=True))
    predicted = display_parts.u.detach().numpy()
    residual_values = np.linalg.norm(display_parts.r.detach().numpy(), axis=1)
    error_values = np.linalg.norm(predicted - reference, axis=1)
    marks = pd.read_csv(run / "history/marks.csv")
    completions = marks.loc[marks.kind == "window", "iteration"].to_numpy(dtype=int)
    assert len(completions) == 27 and np.all(np.diff(completions) > 0)
    history = pd.read_csv(run / "history/reference_error.csv")
    if not np.isfinite(history[["epoch", "ref_mse", "trained_until"]]).all().all():
        raise ValueError("Invalid recorded reference history")
    objectives = pd.read_csv(run / "history/loss_history.csv")
    stages = pd.read_csv(run / "history/stages.csv")
    assert len(objectives) == completions[-1]
    assert np.array_equal(objectives.iteration.to_numpy(), np.arange(len(objectives)))
    assert objectives.phase.isin(["adam", "lbfgs"]).all()
    assert np.isfinite(objectives.loss).all() and (objectives.loss > 0).all()
    frames = []
    for k, sub in enumerate(model.windows):
        stop = model.edges[k + 1]
        selected, dense_selected = time <= stop + 1e-12, dense_t <= stop + 1e-12
        layers = [dict(w=lists(m.weight.detach().numpy()), b=lists(m.bias.detach().numpy()))
                  for m in sub.net if isinstance(m, torch.nn.Linear)]
        learning_history = history.loc[np.isclose(history.trained_until, stop, atol=1e-8, rtol=0)]
        assert len(learning_history) > 0
        local_loss = objectives.loc[objectives.window == k]
        first = int(local_loss.iteration.iloc[0])
        adam_count = int((local_loss.phase == "adam").sum())
        stage_rows = stages.loc[stages.window == k]
        assert adam_count == int(stage_rows.adam_steps.sum())
        assert first + len(local_loss) == completions[k]
        assert (local_loss.phase.iloc[:adam_count] == "adam").all()
        assert (local_loss.phase.iloc[adam_count:] == "lbfgs").all()
        local_ref = learning_history.epoch.to_numpy() - first
        assert np.all(local_ref >= 0) and np.all(local_ref <= len(local_loss))
        window_mask = (dense_t > model.edges[k]) & (dense_t < stop)
        left, right = np.searchsorted(time, [model.edges[k], stop])
        window_t = np.r_[time[left], dense_t[window_mask], time[right]]
        window_xyz = np.vstack([predicted[left], dense_pred[window_mask], predicted[right]])
        window_ref = np.vstack([reference[left], dense_ref[window_mask], reference[right]])
        frames.append(dict(epoch=int(completions[k]), window=k + 1, start=model.edges[k], stop=stop,
                           time=lists(time[selected]), xyz=lists(predicted[selected]),
                           error=lists(error_values[selected]), residual=lists(residual_values[selected]),
                           loss=float(np.mean(dense_r[dense_selected] ** 2)),
                           rmse=float(np.sqrt(np.mean(dense_error[dense_selected] ** 2))), layers=layers,
                           history_t=lists(local_ref),
                           objectives=local_loss.loss.to_list(), adam_count=adam_count,
                           evaluation_count=len(local_loss), first_evaluation=first,
                           stages=[dict(eps=float(r.eps), stop=int(c), adam_steps=int(r.adam_steps),
                                        threshold_met=bool(r.threshold_met), min_w=float(r.min_w_final)) for r, c in
                                   zip(stage_rows.itertuples(), np.cumsum(stage_rows.adam_steps))],
                           window_time=lists(window_t), window_xyz=lists(window_xyz),
                           window_reference=lists(window_ref),
                           history_error=lists(np.sqrt(3 * learning_history.ref_mse.to_numpy()))))
    xyz = np.concatenate([reference, predicted])
    def bounds(array):
        lo, hi = array.min(axis=0), array.max(axis=0)
        padding = np.maximum((hi - lo) * .08, .02)
        return lists(np.stack([lo - padding, hi + padding], axis=1))
    cases = []
    initial_states = {
        "baseline": ((.5, .75, 1.), "Documented initial state of the causal 1536-point run."),
        "half": ((.25, .375, .5), "Half the baseline amplitude. The quadratic RHS implies u_a(t) = a u(a t)."),
        "equilibrium": ((0., 0., 1.), "On the z axis, xy, xz and yz all vanish. The solution stays constant."),
        "perturbed": ((.02, .02, 1.), "Small equal x and y perturbations from the z-axis equilibrium; z starts unchanged."),
    }
    for k in (1.5, 2., 3.):
        c = lorenz1960_coefficients(k, 1.)
        for key, (ic, explanation) in initial_states.items():
            cfg = Lorenz1960Config(k=k, l=1., initial_state=ic, t_span=(0., 10.), n_eval=601)
            t, u, _ = solve_lorenz1960_scipy(config=cfg)
            # Two conserved quadratic combinations for this bilinear system.
            invariants = np.column_stack([c[1] * u[:, 0]**2 - c[0] * u[:, 1]**2,
                                          c[2] * u[:, 0]**2 - c[0] * u[:, 2]**2])
            drift = float(np.max(abs(invariants - invariants[0])))
            if drift > 1e-7 or not np.allclose(u[0], ic):
                raise ValueError("Reference solution failed invariant/initial-state checks")
            if key == "equilibrium" and not np.allclose(u, ic, atol=1e-12):
                raise ValueError("Equilibrium moved")
            cases.append(dict(k=k, key=key, ic=lists(ic), c=lists(c), t=lists(t), xyz=lists(u),
                              explanation=explanation, drift=drift))
    # Compare half-amplitude integration with the baseline at half physical time.
    for k in (1.5, 2., 3.):
        base = next(v for v in cases if v["k"] == k and v["key"] == "baseline")
        half = next(v for v in cases if v["k"] == k and v["key"] == "half")
        assert np.allclose(np.asarray(half["xyz"])[::2], .5 * np.asarray(base["xyz"])[:301], atol=2e-7)
    return dict(time=lists(time), reference=lists(reference), frames=frames, bounds=bounds(xyz),
                cases=cases, referenceBounds=bounds(np.concatenate([v["xyz"] for v in cases])),
                source=str(run.relative_to(ROOT)), checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                totalSnapshots=int(summary.n_snapshots), totalPoints=int(summary.n_collocation),
                finalRmse=float(summary.rmse_combined_l2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate inputs and reference mathematics only")
    parser.add_argument("--output", type=Path, default=ROOT / "visualizations" / "learning-viewer")
    args = parser.parse_args()
    data = build_data()
    if args.check:
        print(f"PASS: {sum(f['evaluation_count'] for f in data['frames'])} recorded evaluations across {len(data['frames'])} windows; stage/phase counts, reference samples and checkpoint RMSE verified")
        return
    output = args.output.resolve()
    if output == ROOT or any(output == (ROOT / name).resolve() or (ROOT / name).resolve() in output.parents
                             for name in ("runs", "src", ".git")):
        raise ValueError("Output must be separate from core code and saved runs")
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, separators=(",", ":"), allow_nan=False)
    page = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")
    page = page.replace("/* VIEWER_SCRIPT */", Path(__file__).with_name("viewer.js").read_text(encoding="utf-8"))
    page = page.replace("/* PLOTLY_LIBRARY */", get_plotlyjs()).replace("/* VIEWER_DATA */", encoded)
    (output / "index.html").write_text(page, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "source": data["source"], "window_completion_counts": [f["epoch"] for f in data["frames"]], "checkpoint_sha256": data["checkpointSha256"],
        "recorded_evaluations": sum(f["evaluation_count"] for f in data["frames"]),
        "windows": [{k: f[k] for k in ("window", "start", "stop", "adam_count", "evaluation_count", "stages")} for f in data["frames"]],
        "display_points": len(data["time"]), "training_points": data["totalPoints"],
        "reference_cases": [{k: c[k] for k in ("k", "key", "ic", "c", "drift")} for c in data["cases"]],
        "limits": ["Final frozen-window prefixes reconstructed from checkpoint", "In-window parameter snapshots absent locally",
                   "No output sensitivity or activation replay", "Mathematical cases, no physical calibration"],
    }, indent=2), encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
