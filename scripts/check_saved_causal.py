import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from pinn.functions.losses import causal_weights
from pinn.pinn import residual_parts
from pinn.sweep import config_for
from pinn.train import load_run, make_grid, predict, residual_at
from pinn.config import reference_at, reference_trajectory
from pinn.viz.figures import invariant_series


def main():
    torch.set_num_threads(1)
    run = ROOT / 'runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd'
    cfg = config_for(run)
    model, history, cfg = load_run(cfg)
    t, reference = reference_trajectory(cfg, cfg.n_eval)
    pred = predict(model, t)
    err = np.linalg.norm(pred - reference, axis=1)
    grid = make_grid(cfg, next(model.parameters()).device)
    coll = residual_at(model, grid.cpu().numpy().reshape(-1))
    dense = residual_at(model, t)
    summary = pd.read_csv(run / 'run_summary.csv').iloc[0]
    metrics = {
        'rmse_combined_l2': float(np.sqrt(np.mean(err ** 2))),
        'max_abs_error_combined_l2': float(err.max()),
        'residual_mse_collocation': float(np.mean(coll ** 2)),
        'residual_mse_dense': float(np.mean(dense ** 2)),
    }
    for name, value in metrics.items():
        assert np.isclose(value, summary[name], rtol=1e-8, atol=1e-12), name
    counts = torch.bincount(model.window_of(grid), minlength=27).tolist()
    assert counts == [1535] * 26 + [1536]
    joints = []
    for k, edge in enumerate(model.edges[1:-1]):
        te = torch.tensor([[edge]], dtype=torch.float64, device=next(model.parameters()).device).requires_grad_(True)
        left = residual_parts(model.windows[k], te)
        right = residual_parts(model.windows[k + 1], te)
        joints.append({'outgoing_window': k, 'time': edge,
                       'value_jump': float((right.u - left.u).detach().norm()),
                       'derivative_jump': float((right.dudt - left.dudt).detach().norm()),
                       'left_residual_norm': float(left.r.detach().norm()),
                       'right_residual_norm': float(right.r.detach().norm())})
    assert max(j['value_jump'] for j in joints) < 1e-12
    final_weights = []
    for k, window in enumerate(model.windows):
        pts = grid[model.window_of(grid) == k]
        if k < 26:
            pts = torch.cat((pts, torch.tensor([[model.edges[k + 1]]], dtype=pts.dtype, device=pts.device)))
        r = residual_parts(window, pts.clone().requires_grad_(True)).r
        final_weights.append(float(causal_weights(r.pow(2).mean(dim=1), 10).min()))
    stages = pd.read_csv(run / 'history/stages.csv')
    _, drift = invariant_series(pred, cfg.coefficients)
    probe_t = np.random.default_rng(20261001).uniform(*cfg.t_span, 8192)
    probe_err = np.linalg.norm(predict(model, probe_t) - reference_at(cfg, probe_t), axis=1)
    probe_r = residual_at(model, probe_t)
    data = {
        "off_grid_random_probe": {"seed": 20261001, "points": 8192,
            "rmse_state_norm": float(np.sqrt(np.mean(probe_err ** 2))),
            "max_state_error_norm": float(probe_err.max()),
            "residual_mse": float(np.mean(probe_r ** 2))},
        'checkpoint_sha256': hashlib.sha256((run / 'history/pinn.pt').read_bytes()).hexdigest(),
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'scipy': scipy.__version__, 'device': str(next(model.parameters()).device)},
        'metrics': metrics,
        'invariant_relative_max_drifts': drift.max(axis=0).tolist(),
        'parameters': sum(p.numel() for p in model.parameters()),
        'global_points': len(grid), 'local_training_points': [1536] * 27,
        'adam_updates': history.adam_iters,
        'lbfgs_closure_evaluations': sum(p == 'lbfgs' for p in history.loss_phase),
        'capped_stages': stages.loc[~stages.threshold_met].to_dict(orient='records'),
        'final_epsilon10_min_weights': final_weights,
        'final_windows_below_threshold': [k for k, w in enumerate(final_weights) if w <= cfg.causal_delta],
        'joints': joints,
    }
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()
