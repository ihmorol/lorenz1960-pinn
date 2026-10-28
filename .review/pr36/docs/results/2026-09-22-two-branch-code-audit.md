# Code and result audit: batch precision vs causal windows

**Repair status (later on September 22):** This document audits the original
branch code and saved runs. See [the causal-window repair](2026-09-22-causal-window-repair.md)
for subsequent code changes. The saved results have not been regenerated.

**Audited branches:** `feat/batch-precision-run` at `13ebe87` and `feat/causal-window-training` at `f54cac3` (the latter contains the former). **Scope:** code, the supplied CSVs, saved figures, and checkpoint file presence. No training was run. The local Python lacks PyTorch, SciPy, and pandas, so the checkpoints could not be independently executed here. CSV arithmetic below was recomputed with Python's standard library.

## Full source pass and execution boundary

I compared the branches' Python diff (17 changed or added files) and inspected all application paths: root runners; `config`, `problems`, `pinn`, `train`, `sweep`, `history`; every module under `functions/`; the SciPy baseline and independent RK4 solver; and the figure, landscape, trajectory, comparison, film, and index modules under `viz/`. The branch diff adds window and causal training in B, plus its telemetry and figures. The ODE right-hand side, hard-trial formula, time derivative, samplers, and optimizer helpers are inherited unchanged from A. A's single-domain training loop is inherited except for the new dispatch into the window path.

**Checks actually executed:** `compileall` passed for all source and root scripts; isolated `split_windows` and index-page functions passed on small inputs; and the supplied CSVs were recomputed with Python's standard library. `python3 -m pytest --version` fails because pytest is absent. PyTorch, SciPy, and pandas are also absent, so neither branch's imported test suite, checkpoint evaluation, or other dependency-bound functions could be executed in this environment. Many existing tests call `train()` even with small fixtures; they were deliberately not run under the request to avoid retraining. Thus this is full source inspection plus bounded execution, **not a claim that every function was executed**.

## What the saved data establishes

| Measure | Run A: one full-domain network | Run B: 27 sequential networks |
|---|---:|---:|
| Parameters | 11,283 | 304,641 |
| Collocation / evaluation points | 39,793 / 13,265 | 39,793 / 13,265 |
| Combined state RMSE | 2.2370605 | 8.4346039e-5 |
| Combined maximum state error | 3.3461950 | 2.3548755e-4 |
| Final whole-domain collocation residual MSE | 2.8023871e-4 | 1.4675793e-7 |
| Wall time on the recorded run | 4,528.55 s | 928.58 s |
| Recorded Adam steps | 40,000 | 29,277 |
| Recorded L-BFGS closure evaluations | 6,251 | 49,400 |

Sources: both `metrics.csv`, `run_summary.csv`, `history/loss_history.csv`, and Run B's `history/causal.csv` and `history/marks.csv`. The saved figures agree qualitatively: A approaches a nearly stationary curve near the y axis; B closely tracks the reference for this orbit. The ODE coefficients in `src/baseline/lorenz1960_baseline.py:28-47` and `src/pinn/functions/physics.py:5-10` agree: (-0.1, 1.6, -0.75). `compute_error_metrics` in the baseline uses the Euclidean state error for the combined RMSE.

These runs establish performance **for these configurations and this seed**. Run B has 27 times as many trainable parameters, a different collocation grid (uniform versus Latin hypercube), shorter subdomains, warm starts, causal weighting, and a different optimizer schedule. The comparison cannot isolate the benefit of causal weighting, windowing, or precision individually. The 78,677 entries in B's loss history are Adam steps plus L-BFGS **function evaluations**, not 78,677 optimizer steps.

The saved reference endpoint is about **1.65e-6** from the initial state at the rounded `T=13.26446`, computed from `final_ref_x/y/z` in either summary. Thus the run spans an approximately closed orbit, but the `run_batch.py:14` comment's **6e-9** return claim is inconsistent with these saved results. B's endpoint error against this reference is about **3.46e-5**; its distance from the initial state is about **3.33e-5**. These are different quantities.

## Verified defects and data interpretation risks

### Branch difference map

| Area | Batch branch | Causal-window branch |
|---|---|---|
| Model | One `PINN` over the full span | Adds `WindowedPINN`, one `PINN` per span, endpoint handoff, optional warm start |
| Adam objective | Plain mean squared residual | Per-point causal weighted residual within each window; plain residual if schedule empty |
| L-BFGS objective | Plain mean squared residual, once | Plain mean squared residual, once per window |
| Sampling | Run A selects Latin hypercube | Run B selects uniform grid; shared sampler functions unchanged |
| Training state | One Adam/scheduler followed by L-BFGS | Fresh Adam/scheduler for each window; saved window checkpoints |
| Evaluation | Shared reference, metric, and residual functions | Same functions routed through piecewise model; adds window-specific telemetry and figures |

`Config.arch` encodes window count, causal presence, and warm-start presence, but not causal schedule values, optimizer budget, collocation density, or all problem parameters. Consequently, two materially different configurations can still target the same directory.

| Core file or group inspected | Branch status | Result of logical review |
|---|---|---|
| `src/baseline/lorenz1960_baseline.py`, `lorenz1960_solver.py` | Shared | RHS and reported error formulas checked; the independent RK4 script is not called by either PINN run. |
| `src/pinn/problems.py`, `functions/physics.py`, `functions/derivative.py`, `functions/trial.py` | Shared | RHS, pointwise autograd derivative, and hard-IC algebra are consistent with the current IVP. |
| `functions/collocation.py`, `functions/optimizers.py`, `functions/reference.py` | Shared | Grid and optimizer choices explain several comparability differences; duplicate-time reference edge case noted below. |
| `functions/losses.py`, `functions/windows.py` | Losses extended; windows added in B | Causal ordering and window split inspected; exponent scaling discussed in finding 6. |
| `config.py`, `pinn.py`, `train.py` | Extended in B | Window routing, endpoint state handoff, warm start, stopping, checkpointing, and loss switch inspected. |
| `history.py`, `sweep.py` | Extended in B | Snapshot, CSV, reload, and run reuse inspected; findings 2-5 and 9-13 apply. |
| `viz/figures.py`, `training.py`, `evaluation.py`, `landscape.py` | Extended in B | Plot inputs traced to saved state; label and historical-state limitations recorded. |
| `viz/compare.py`, `trajectory3d.py`, `film/script.py`, `index.py` | Shared or extended in B | Comparison and film consume snapshot products; index affects navigation only. |
| Root `run_*.py` and `src/pinn/test_pinn.py` | A runners plus B runner/tests | Entry points traced to shared functions; tests reviewed for missing result-level assertions. |

### 1. Run B's `final_loss` is only the last window's objective (high impact on reported comparisons)

`train_windows` appends a separate objective for each window and ends after the last window's L-BFGS closure (`src/pinn/train.py:178-216`). `run_summary` assigns `history.loss[-1]` to `final_loss` (`src/pinn/train.py:329-337`). B's `final_loss` is **1.4953645e-8**, while its recomputed whole-domain residual MSE is **1.4675793e-7**. Both are real, but they answer different questions. A's `final_loss` and whole-domain MSE are both 2.8023871e-4. B's `best_loss` and `epochs_to_*` likewise mix objectives from different windows and causal stages, so they are not whole-run convergence statistics. `epochs=20000` is an unused default for B; `ms_per_epoch=wall/20000` is therefore not a measured per-step cost (`src/pinn/train.py:301,350`).

**Small fix:** keep `final_loss` as the last logged window loss only if it is renamed accordingly; report a separately recomputed `final_residual_mse_full` as the common comparison metric. Report actual Adam steps, L-BFGS closure evaluations, and wall time. Mark threshold-crossing fields unavailable for windowed runs unless the threshold is applied to one consistent full-domain metric.

### 2. Resume can reuse incompatible state and silently corrupt the history (high impact if resumed)

Run A loads the newest `adam_*.pt` based only on filename, without restoring prior loss, diagnostics, snapshots, or elapsed time (`src/pinn/train.py:67-76`). If it resumes, the optimizer state may be correct but the saved history describes only the remaining segment; if the checkpoint is at `epochs`, the later `history.loss[-1]` access fails. Run B restores `window_XX.pt` if it exists, also without matching a configuration or restoring prior history (`src/pinn/train.py:165-171,217-228`). `run_one(resume=True)` skips any run with an existing summary regardless of its configuration (`src/pinn/sweep.py:47-59`). The directory tag omits several settings, so stale files can be mistaken for the requested experiment.

On a partial resume, newly written breakdown files can coexist with older files while `point_summary.csv` accumulates only the new in-memory snapshots; B also restarts its global log index at zero, so names can collide. A fully restored B run has an empty `history.loss`, yet `main()` would still proceed to save a fresh report. The test suite exercises `train()` in this state, not `main()`/`save_results()`.

This is a **code-level risk**, not proof that either supplied run was resumed incorrectly. The supplied A history contains all 40,000 Adam steps; B has all 27 window marks and 29,277 causal steps.

**Small fix:** save the complete run configuration with every checkpoint and refuse to reuse a checkpoint or summary on mismatch. Either restore the corresponding history on resume or explicitly start a fresh output directory. Ensure the completed-checkpoint path produces a valid history and summary.

### 3. Windowed training telemetry includes untrained future windows (medium impact on figures)

Every `WindowedPINN` submodel initially has the global initial state (`src/pinn/pinn.py:83`). A later window receives its actual anchor only after the previous window finishes (`src/pinn/train.py:222-225`). Meanwhile, `record_reference` and `snapshot` evaluate the *entire* composite model (`src/pinn/train.py:155-163,200-201`). Early B values in `reference_error.csv`, the trajectory film, and time-by-training surfaces therefore include untrained future networks with wrong anchors. They describe the unfinished assembly, not the accuracy of the already trained prefix. The completed model's metrics are unaffected.

**Small fix:** for progress metrics, evaluate only through the currently trained window's right edge and record that coverage. If the full-domain unfinished state is useful for the film, label it as such. Do not use early full-domain reference error to infer how a trained window behaves.

### 4. The two point summaries refer to different optimizer states (medium impact on spatial claims)

Run A snapshots before the Adam update (`src/pinn/train.py:87-94`) and takes no final snapshot after L-BFGS (`src/pinn/train.py:127-143`). Its `point_summary.csv` "final" MSE is **2.9245742e-4**, exactly the last logged Adam loss, versus **2.8023871e-4** for the actual final model (4.36% higher). Spatial plots and per-bin tables made from that point summary describe pre-L-BFGS A. The shared comparison figure also loads each run's last `breakdown/epoch_*.csv` (`src/pinn/viz/compare.py:11-19`), so its A curve is not the final post-L-BFGS model even though its B curve is final. Run B's point-summary contribution sum matches its final full-domain MSE to rounding. B's `run_summary.csv` nevertheless says `n_snapshots=16`, while every point-summary row says `n_snapshots=17`; the supplied artifacts have a provenance mismatch.

**Small fix:** take one explicit post-optimizer full-domain snapshot in both paths, then compute the point summary and run summary from that same state. Store the checkpoint hash or run identifier with all derived artifacts. Regenerating A's point-level figures from its final checkpoint would require the project runtime; it has not been done in this audit.

### 5. Windowed loss and gradient plots mix unlike quantities (medium impact on conclusions)

During B's Adam phase, `res` is overwritten by `causal_loss` and then logged as `residual_loss` (`src/pinn/train.py:183-199`); during L-BFGS, `pinn_loss` is the plain mean residual (`src/pinn/train.py:211-216`). The loss curve therefore changes **objective definition** at every stage and at the optimizer switch. A logs the plain residual throughout. An apparent jump or fall at a boundary cannot by itself show optimizer success or failure. B's `history.adam_iters` is set to `len(history.loss)`, which already includes previous windows' L-BFGS evaluations (`src/pinn/train.py:210`); the reloader also infers it from a mixed global epoch (`src/pinn/history.py:159`). The phase plot can recover boundaries from `eps_marks`, but generic consumers of `adam_iters` can be misled.

**Small fix:** log phase, window, Adam-step count, closure-evaluation count, weighted loss, and raw residual MSE as distinct fields. Use the raw residual for cross-run plots. State clearly that an L-BFGS closure call is a function evaluation.

### 6. The causal exponent depends on point count; a one-line normalization is not a proven fix

`causal_weights` uses `exp(-eps * sum(previous per-point losses))` (`src/pinn/functions/losses.py:9-19`). With about 1,474 points per window, the meaning of `eps` and of `min_w > 0.99` changes with collocation density. At `eps=10`, passing the threshold requires the preceding losses to sum below about **0.001005**, or average below about **6.8e-7** at this density. The six stages in windows 9-11 and 23-25 reach their 4,000-Adam-step cap with final `min_w` between **0.778 and 0.905**; that is verified in `causal.csv` and `marks.csv`.

This is a scaling/design concern, not evidence of a wrong implementation of the chosen per-point causal objective. Dividing the cumulative sum by point count while retaining the same `eps` would nearly remove the causal gate: the first recorded `min_w=0.03046` would become about **0.99763**, already above the 0.99 stop threshold. It cannot be claimed to improve accuracy without another experiment. The six capped windows account for about **24%** of B's final collocation residual, so they do not dominate that metric. Windows 0, 12, 22, and 23 together account for about **70%**; only window 23 is capped.

**Strategy:** leave the saved run's definition unchanged. For a future controlled experiment, use a fixed number of time chunks with mean loss per chunk, or define an explicitly scaled time integral; retune `eps` and stopping threshold together. Compare at the same grid and budget. Do not retrofit a changed objective onto the existing result.

### 7. Joint slope mismatch is real, but C1 continuity is not required by this first-order IVP

The handoff copies the previous endpoint into the next hard trial function (`src/pinn/train.py:222-225` and `src/pinn/functions/trial.py:3-6`), giving C0 continuity at the joints. The saved joint figure shows value jumps near floating-point roundoff and finite-difference slope mismatches around 1e-3. With `u=u0+(t-t0)N(t)`, the right derivative at a window's start is `N(t0)`, so it is learned. This is an approximation limitation, **not a broken handoff**. The figure measures left-versus-right slopes; it does not show which side is responsible for each mismatch or prove that joint slopes dominate final trajectory error. Within 0.01 time units of the 26 joints, about 3.9% of B's collocation points contribute about 16.3% of its final residual MSE.

**Strategy:** first report the left and right residuals separately at every joint. A trial function fixing the initial slope to `f(u0)` is a legitimate new ansatz, but it changes optimization and must be evaluated as a separate experiment, not presented as a confirmed bug fix or guaranteed route to 1e-6 error.

### 8. Several visual interpretations overreach their inputs (documentation/figure impact)

- The `precision_floor.png` source shown in the plan compares a float32 run on `[0,10]` with Run A on `[0,13.26446]` (`docs/superpowers/plans/2026-09-21-long-horizon-precision-and-causal.md:1864`). Their grids and budgets differ as well. It does **not** isolate precision. The current saved inputs for the float32 curve are absent, so this conclusion cannot be verified further here.
- B's loss landscape samples the **unweighted full-domain** objective with the checkpoint's final window anchors (`src/pinn/viz/landscape.py:15-25,74-93`), while the historical training path optimized one window at a time under causal weights. Only parameters are saved in the parameter trail, not historical `u0` anchors (`src/pinn/train.py:155-163,395-402`). The plotted path and 2D PCA plane therefore do not reconstruct B's historical objective. A path height below the displayed plane is also possible because PCA projection discards other directions. The plot is illustrative, not proof of a monotonic funnel or escape mechanism.
- The B NTK snapshots likewise reload historical parameters into a model retaining the **final** window anchors (`src/pinn/viz/__init__.py:43-49`). Comparing that 27-network NTK with A's one-network NTK cannot establish that the two models have the same representational capacity or explain the accuracy gap.
- The equilibrium near the y axis is consistent with A's saved trajectory and the ODE, but an exact residual-zero curve satisfying the same initial condition would be the unique IVP solution. Calling A's result a "degenerate global minimizer" is unsupported. The data show a low-residual, high-error **approximate** solution, with residual concentrated early; they do not establish the global optimization landscape.

**Small fix:** revise the captions and the conclusions in `docs/results/2026-09-21-run-A-vs-run-B.md` and the newer analysis drafts to state exactly what was measured. A matched-configuration precision test or a causal-versus-noncausal window ablation is needed before making those causal claims.

### 9. Run B's saved gradient vectors contain stale gradients from completed windows (figure defect)

Each B Adam step calls `adam.zero_grad()` only on the active submodel (`src/pinn/train.py:177-189`). Completed windows retain their last `.grad` tensors, including those from L-BFGS closures. At a snapshot, `flat_grads(model)` concatenates `.grad` from **all** 27 windows (`src/pinn/train.py:155-163`; `src/pinn/history.py:38-40`). The result is neither the active window's gradient nor the gradient of the full-domain loss. B's gradient-stability curve and whole-model gradient histograms built from these vectors can therefore be wrong even though the training updates are not affected. The weight-path marker sizes instead use `record_step` norms from the active submodel (`src/pinn/viz/landscape.py:110-115`). The supplied `param_trail.npz` is absent, so the size of the gradient-trail distortion cannot be measured from the shared artifacts.

**Small fix:** clear all model gradients at the start of each windowed Adam step, or store an explicitly labeled active-window gradient vector. Do not interpret the existing B gradient-trail figures as actual full-model gradients. The `record_step` norms use only `sub` and are a different, better-defined measurement.

### 10. The causal stage condition is checked before, not after, the accepted Adam step (training logic)

`min_w` is computed from the current parameters, then `adam.step()` changes them, and only then does the loop decide to advance using the old `min_w` (`src/pinn/train.py:183-207`). Thus the recorded crossing does not prove that the weights at the start of the next stage meet `causal_delta`. Whether this mattered in the supplied run is not recoverable from the saved pre-step `causal.csv` alone.

**Small fix:** evaluate the stage condition on the parameters that will actually enter the next stage, or stop before applying the step once the pre-step state has met the condition. Keep the rule and its log on the same state.

### 11. Reloaded evaluation misreports cost, and custom problem settings are not round-tripped

`TrainHistory.from_saved()` restores loss and sampled diagnostics but leaves `wall_clock_s` and snapshot count at their defaults and reconstructs `adam_iters` from the last **mixed** global epoch (`src/pinn/history.py:139-171`). `run_eval.py:25-34` then calls `run_summary` on that history, so its displayed wall time and `ms_per_epoch` become zero, and B's Adam count is wrong if used. This does not alter the original run's saved accuracy table.

`config_for()` rebuilds a run from summary fields (`src/pinn/sweep.py:107-138`), but the summary does not save `k`, `l`, `initial_state`, or `end_state`. For these two default-physics runs this is harmless. For a custom-physics run, reloading can construct the wrong right-hand side because `PINN.problem` is made from the rebuilt configuration while the checkpoint contains only tensors; its saved `coeffs` buffer is not what `rhs()` calls (`src/pinn/pinn.py:49-57`; `src/pinn/problems.py:25-28`).

**Small fix:** persist the complete configuration and run counters beside the checkpoint; reload that record rather than guessing from a subset of summary columns. Keep original training cost fields from the saved run summary when printing a later evaluation.

### 12. Some secondary figure labels claim more than their code computes

`SnapshotWriter` stores the **first observed** snapshot with residual below 1e-4 (`src/pinn/history.py:239-240`), without checking that later snapshots remain below it. The `point_convergence` plot labels this as convergence epoch (`src/pinn/viz/figures.py:766-801`). In B it can also mark an untrained future window before that window is reached. It should be called first threshold crossing unless sustained convergence is computed. The film's yellow arrows are drawn opposite to the observed PCA path displacement (`src/pinn/viz/film/script.py:72-85`); they are not computed from a gradient, so labeling them negative-gradient arrows is unjustified. The `error_growth` log-log fit (`src/pinn/viz/evaluation.py:20-30`) is descriptive only; it does not establish a polynomial law or rule out exponential behavior from one finite trajectory.

**Small fix:** correct these labels. Only add stronger mathematical interpretations if the corresponding quantities are actually computed.

### 13. Existing tests do not protect the result-level invariants

`test_adam_checkpoint_resumes` explicitly expects a resumed six-step run to have only three loss records (`src/pinn/test_pinn.py:410-419`), and `test_windowed_resume_skips_saved_windows` explicitly expects an empty history after restoring all windows (`src/pinn/test_pinn.py:495-503`). Those tests confirm the current control flow but also normalize the incomplete-history behavior described in finding 2. The causal test checks marks and iteration bounds, not final whole-domain residual, exact handoff state, post-step stopping status, or cross-run comparability (`src/pinn/test_pinn.py:467-479`).

**Small fix:** when changing resume behavior, replace those expectations with assertions that the restored model **and** full history produce the same summary as the original run. Keep any non-training validation on saved synthetic histories or small function inputs; the long A/B training need not be repeated for this audit.

## Low-risk guards and checks

- `Config` permits a nonempty causal schedule with `causal_max_iters=0`; the causal Adam loop then executes zero steps (`src/pinn/config.py:37-39`, `src/pinn/train.py:151-179`). Validate positive iterations and positive windows/point counts.
- A zero-step causal schedule is intentionally used in `test_warm_start_copies_previous_window_weights` (`src/pinn/test_pinn.py:516-526`), so a production guard should distinguish that explicit test case instead of assuming every zero budget is accidental. More generally, `n_windows=0`, a zero-length span, or an empty window's collocation subset can fail with division by zero, an empty reduction, or NaN. Validate the configuration and actual per-window point counts before optimization.
- `PINN` initially creates `u0` and `coeffs` buffers in float32 (`src/pinn/pinn.py:52-53`). The training and reload paths call `.to(cfg.torch_dtype)`, and this run's initial state is exactly representable in float32; this does not explain either result. Initial construction can preserve `cfg.torch_dtype` for other inputs.
- The scheduler is recreated for every B window (`src/pinn/train.py:177`). With B's `StepLR` interval of 5,000 Adam steps, none of the recorded windows reaches a decay step, so the recorded Adam learning rate stays at 1e-3 despite the generic setup text advertising a move toward 1e-4 (`src/pinn/train.py:56`). This is a misleading schedule description and a configuration choice, not a code failure in the saved run.
- The three `autograd.grad` calls in `time_derivative` are a valid way to obtain the three output derivatives because each sample is processed independently. They are a possible performance target, not a demonstrated result defect. Empty time windows do not leave `torch.empty` outputs unfilled for finite times: `bucketize` assigns every queried time to one window.
- `reference_at` and generic `solve_reference` sort requested times but do not deduplicate them (`src/pinn/config.py:168-180`; `src/pinn/functions/reference.py:6-20`). SciPy rejects repeated `t_eval` times. The A/B generated grids have distinct times, so this is an API edge case rather than a defect in their results.

## Minimal order of work

1. Correct the comparison labels and report the common final whole-domain residual, actual step/evaluation counts, parameter count, and one-seed limitation. This requires no training.
2. Make checkpoint reuse configuration-safe and history-complete. Add a post-optimizer final snapshot and align summaries with it. These are correctness and provenance repairs for future runs.
3. Separate weighted training loss from raw residual telemetry, clear inactive-window gradients before snapshotting, and mask progress to the trained time prefix. Correct figure labels. Regenerate affected figures from a valid final checkpoint only where their source data suffice; historical window anchors were not saved, so do not claim exact reconstruction of early B model states.
4. Align the causal stop check with the parameter state it certifies. Treat causal rescaling, joint-slope trial functions, and causal ablations as new experiments. The supplied runs cannot determine their outcome, and this audit did not retrain either model.
