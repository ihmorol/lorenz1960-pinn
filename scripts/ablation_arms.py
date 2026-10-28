"""Ablation arms for the windowed causal run.

    python scripts/ablation_arms.py <arm>

Arms: causal_s0, causal_s1, plain_adam1287_lbfgs, lbfgs_only, causal_adam_only.
Writes runs/ablation/<arm>/result.json; the locked results are not touched.
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pinn.config import Config, reference_trajectory  # noqa: E402
from pinn.pinn import residual_parts  # noqa: E402
from pinn.train import make_grid, predict, train  # noqa: E402

arm = sys.argv[1]
out = REPO / "runs" / "ablation" / arm
out.mkdir(parents=True, exist_ok=True)
if not torch.cuda.is_available():
    torch.set_num_threads(2)

T_LOOP = 13.26446
base = Config(t_span=(0.0, T_LOOP), n_collocation=27 * 1535 + 1, eval_per_unit=1000,
              dtype="float64", ic_scale="unit", collocation="uniform", n_windows=27,
              causal_eps_schedule=(1e-2, 1e-1, 1.0, 10.0), causal_delta=0.99,
              causal_max_iters=4000, lr_start=1e-3, lr_end=1e-4,
              lr_decay=0.9, lr_decay_every=1000, lbfgs_iters=1000,
              warm_start=True, snapshot_every=0, print_every=0,
              results_dir=str(out), ckpt_dir=str(out / "history"), runs_dir=str(out))
arms = {
    "causal_s0": base,
    "causal_s1": replace(base, seed=1),
    "plain_adam1287_lbfgs": replace(base, causal_eps_schedule=(), epochs=1287),
    "lbfgs_only": replace(base, causal_eps_schedule=(), epochs=1),
    "causal_adam_only": replace(base, lbfgs_iters=0),
}
cfg = arms[arm]
t0 = time.perf_counter()
model, hist = train(cfg)
wall = time.perf_counter() - t0
t_ref, ys = reference_trajectory(cfg, n=cfg.n_eval)
err = predict(model, t_ref) - ys
grid = make_grid(cfg, next(model.parameters()).device)
r = residual_parts(model, grid.clone().requires_grad_(True)).r.detach()
loss, phase, win = np.array(hist.loss), np.array(hist.loss_phase), np.array(hist.loss_window)
per_window = []
for k in range(27):
    m = win == k
    a, l = loss[m & (phase == "adam")], loss[m & (phase == "lbfgs")]
    per_window.append({"w": k, "adam_steps": int(len(a)), "adam_end": float(a[-1]) if len(a) else None,
                       "lbfgs_evals": int(len(l)), "lbfgs_end": float(l[-1]) if len(l) else None})
res = {"arm": arm, "wall_s": wall, "rmse_combined": float(np.sqrt((err ** 2).sum(1).mean())),
       "max_err_norm": float(np.linalg.norm(err, axis=1).max()),
       "rmse_xyz": [float(np.sqrt((err[:, j] ** 2).mean())) for j in range(3)],
       "residual_mse": float((r ** 2).mean()), "per_window": per_window,
       "adam_steps": int((phase == "adam").sum()), "lbfgs_evals": int((phase == "lbfgs").sum())}
(out / "result.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "per_window"}))
