# Causal-window repair and next configuration

This follows the [source and artifact audit](2026-09-22-two-branch-code-audit.md).
The historical Run A and Run B files are unchanged. No training was run for
this repair. Code syntax was checked, but the project dependencies are absent
from the available Python environment, so the PyTorch test suite and new
configuration have not been executed here.

## What changed

| Audit issue | Small code-level resolution | Limit |
|---|---|---|
| Incomparable `final_loss` | `run_summary()` now uses the final full-domain collocation residual for `final_loss` and `final_residual_mse_full`, and records the last optimizer objective separately. Windowed `best_loss` and threshold epochs are left undefined. | Historical CSVs retain their original meanings. |
| Incomplete or incompatible resume | `history/config.json` records every setting. `history/progress.pt` atomically stores the model, optimizer where needed, full history, random states, and next epoch/window. A mismatched configuration is rejected; completed state reloads with its history. | Old checkpoints without a manifest remain readable for evaluation but cannot silently resume. |
| Future windows in progress data | Causal progress error is measured only over the trained prefix. Breakdown rows beyond that prefix are masked, and per-point statistics count only valid snapshots. Films and interactive trajectories draw the available prefix. | Old progress figures cannot be corrected exactly because historical anchors were not saved. |
| Pre-L-BFGS point summary | When snapshotting is enabled, a final post-optimizer snapshot is written for new single-domain runs and at window ends. The comparison page evaluates the saved final checkpoints. | Existing Run A point tables and comparison image remain historical. |
| Mixed loss and step counts | Loss CSVs now carry phase and window, diagnostics distinguish raw residual from causal weighted objective, and summaries report Adam steps and L-BFGS closure evaluations. Phase plots use explicit phase tags. | A cross-window training-loss curve still has changing objectives; use the common final residual for comparisons. |
| Causal point-count scaling | The published per-time-point causal loss is retained. The proposed run fixes and records its collocation count; settings hashes prevent accidental reuse after a density or epsilon change. | There is no evidence that a one-line normalization improves this IVP. Epsilon needs retuning if the grid changes. |
| Joint slope mismatch | The final joint figure also evaluates the left and right ODE residual separately at each joint. | C0 handoff is mathematically valid for this first-order IVP. A new C1 trial form would be a separate experiment. |
| Overinterpreted plots | Figure titles now describe measurements. Historical B path heights are explicitly labeled as final-anchor global residuals; new windowed runs omit that expensive automatic diagnostic. The film no longer draws invented gradient arrows or requires a landscape that was not generated. Precision plots require otherwise matching configurations. | Existing rendered images remain historical, and the one-seed comparison cannot isolate causal weighting. |
| Stale gradient trail | Each causal Adam step clears the full model's gradients before the active window backward pass; final snapshots compute a fresh active-window residual gradient. Histograms omit inactive zero entries. | Historical gradient plots cannot be repaired without the missing parameter trail. |
| Stale causal stop test | When a stage is about to stop, and at its last allowed step, the minimum weight is recomputed after the optimizer update. A `stages.csv` records whether each stage met the threshold or reached its cap. Each nonfinal window also trains its outgoing boundary residual. | A cap remains a finite compute budget, not a convergence guarantee. |
| Reloaded cost and custom physics | Reload reads original wall time and counts from the saved summary. New runs restore the full configuration, including the exact coefficients used by the model's RHS. | Old custom runs without saved coefficients in their summaries still need their original configuration supplied. |
| Threshold and growth labels | Point plots say *first observed threshold crossing*. Error-growth plots say *descriptive fit*. | Neither proves sustained convergence or an error-growth law. |
| Missing result invariants in tests | Tests now check full history on resume, configuration mismatch, final residual semantics, and masked untrained points. | They must be run in an environment with the project requirements; no long A/B retraining is needed. |

The saved run layout is catalogued in [runs/README.md](../../runs/README.md).
Existing result paths stay intact for the reports and checkpoints. New causal
runs use `runs/causal-window/candidate-512pt/<architecture>_<settings-hash>/`;
future batch runs use `runs/batch-precision/`. `run_index.py` discovers both
flat historical and nested new runs. I removed misleading gradient-arrow code;
the standalone runners and RK4 validation
script still have documented uses, so deleting them would remove functionality.

## Proposed causal-window settings

`run_causal.py` now defines one **candidate**, not a claimed optimum:

| Setting | Candidate | Reason |
|---|---:|---|
| Time span / windows | `[0, 13.26446]` / 27 | About 0.491 time units per window, close to the 0.5-window precedent and to the successful saved run. |
| Uniform collocation | 13,825 total, about 512 per window | About one third of the saved run's point count, but twice the 256 time points used per 0.5 window in the published chaotic Lorenz experiment. The systems and networks differ, so this is a compute-conscious starting point. |
| Evaluation grid | about 1,000 per time unit | Retains the existing independent evaluation resolution. |
| Causal epsilon / threshold | `(0.01, 0.1, 1, 10)` / 0.99 | Retains the saved run's schedule and stopping rule; the changed point count changes the gate and therefore needs measurement. |
| Adam cap / L-BFGS | up to 4,000 Adam steps per epsilon stage / 1,500 L-BFGS iterations per window | Retains the budget of the successful run and reports caps explicitly. |
| Learning rate | Adam 1e-3, StepLR factor 0.9 every 1,000 steps | The old 5,000-step interval never fired in any saved window; this schedule can actually decay within a capped stage. |
| Precision / initialization | float64, hard initial condition, warm start | Retains the demonstrated configuration. |

The paper's Lorenz example uses a **different ODE** and reports manually tuned
settings, not an absolute optimum. Its 0.5 windows and 256 points provide a
reference for a candidate only: [Wang, Sankaran and Perdikaris, Appendix B/E](https://arxiv.org/pdf/2203.07404).
The old 39,793-point run is the known accurate reference. The 13,825-point
candidate could be faster or less accurate; no claim about its error or wall
time is made before training. To select a true optimum, compare a small fixed
set of point densities and budgets on identical evaluation grids, using final
whole-domain residual, trajectory error, joint residuals, and wall time. The
reference trajectory must remain evaluation-only, never part of the loss or
stopping rule.
