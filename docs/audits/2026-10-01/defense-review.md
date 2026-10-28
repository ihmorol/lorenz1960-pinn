# Defense review: Lorenz-1960, causal weighting, 1,536 points, 27 windows

Reviewed 1 October 2026 for the 5 October defense. Code inspected at Git commit `0c5f9c8` (abbreviated; full SHA in verification notes below). Actual manuscript inspected at `E:/University/FYDP/lorenz1960-pinn/writings/conference-paper/main.tex`, including its bibliography and writing guide. The ignored writings folder is outside this worktree. This is a review and reproducible evidence package; training code, archived results, and the manuscript were not changed.

## Verdict

**The latest checkpoint supports the reported trajectory accuracy on the specified initial-value problem. The paper needs factual corrections before presentation.** No fatal error was found in the latest default-coefficient training path. This is evidence of one successful configuration, not proof that every design choice is necessary, optimal, or reliable across seeds or problems.

The most serious issue is the paper's account of the failed experiment: it describes sequential reuse of one network across 27 windows, but the cited RMSE 2.237 belongs to the saved full-interval batch experiment. That mechanism must not be presented as an observed experiment without its own artifacts. Other issues concern causal convergence, diagnostic plots, optimizer counts, and a real reference-solver bug outside the current coefficient setting.

## 1. What was verified directly

The final `candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/history/pinn.pt` was loaded and evaluated, without retraining. Checkpoint SHA-256: `d58fbc51d2f48a8af9cd21090378f7fdb8a8ef29bce80a11911ba379532b1456`.

| Quantity | Fresh verification |
|---|---:|
| State RMSE, x | 2.74304051e-5 |
| State RMSE, y | 4.95976389e-5 |
| State RMSE, z | 4.05155236e-5 |
| Combined state RMSE | 6.96696531e-5 |
| Maximum state-error norm | 3.09323169e-4 |
| Whole-grid collocation residual MSE | 1.87391438e-7 |
| Evaluation-grid residual MSE | 1.87248116e-7 |
| Relative invariant drift, SVD basis 1 / 2 | 2.59526619e-4 / 1.67620927e-4 |
| Total trainable parameters | 304,641 |
| Adam updates | 34,765 |
| L-BFGS closure evaluations | 33,143 |
| Maximum joint value mismatch | 2.78e-16 |
| Maximum exact-autograd derivative jump | 5.61493e-3 |
| Maximum left / right joint residual norms | 5.51007e-3 / 4.92074e-3 |

The accuracy and residual metrics agree with the archived CSVs. The saved CUDA training time is approximately 881 seconds; this was not reproduced as a training-time benchmark on the current CPU. The original GPU model and exact library versions were not recorded.

The latest run improves combined RMSE by about 17.4% over historical Run B (8.435e-5), while its maximum state-error norm is about 31.3% larger. It is therefore better by average trajectory error, not uniformly better on every measure.

An additional independently sampled 8,192-time probe (seed 20261001) gave state-norm RMSE 6.93210804e-05, maximum state-error norm 3.09314961e-04, and residual MSE 1.81997110e-07. These samples were not the prescribed uniform training grid; this strengthens within-interval evidence but does not establish new-problem generalization.

## 2. Mathematical problem and reference

The implemented equations are

- dx/dt = -0.1 yz
- dy/dt = 1.6 xz
- dz/dt = -0.75 xy

with u(0)=(0.5,0.75,1.0), k=2, l=1, and t in [0,13.26446]. These are the conservative Lorenz-1960 equations, not the chaotic Lorenz-1963 butterfly system.

The selected benchmark equations and initial state are explicitly present in [PinnDE version 2, Section 4.2](https://arxiv.org/html/2408.10011v2). That section is absent from version 1. The bibliography should pin the version used, rather than leave a 2024 entry ambiguous about the 2025 revision. The equations have historical provenance in Lorenz's 1960 paper; the benchmark state is a selected experimental condition, not a universal physical constant.

Independent verification used a manually specified NumPy RHS, tighter DOP853 tolerances, an alternate Radau solver, and RK4 step refinement. The maximum reference disagreement was 3.1173e-10 in state norm. RK4 endpoint errors at 1,000 / 2,000 / 4,000 steps were 2.938e-10 / 1.832e-11 / 1.145e-12, consistent with fourth-order convergence. Thus reference integration error is far smaller than the current PINN error on this interval.

The rounded endpoint returns to the initial state within **1.6519e-6**, not 6e-9. A tighter reference and local return root give a period near **13.264461940383475**. Treat that as a numerical estimate, not a proof of an exact analytic period. Say “approximately one cycle” or “a nearly closed orbit.” Do not retrain or change the archived horizon merely to make the period wording exact.

Two independently checked quadratic invariants are 16x²+y² and 15x²−2z². Their reference drifts are below 1.5e-13 under the tighter solve. The repository reports a different but valid basis from the null space of the coefficient row. Its reported drift is already **relative**, |I(t)−I(0)|/|I(0)|; multiplying by 100 gives a percentage. Basis and normalization must be stated because invariant numbers depend on them. Conserved invariants alone do not establish correct phase or timing along an orbit.

## 3. Training flow: sound points and precise meaning

1. The interval is divided into 27 equal windows, each about 0.4912763 time units wide.
2. Each window retains an independent 1–60–60–60–60–3 tanh network: 11,283 parameters per network, 304,641 overall. Warm starting copies the previous weights; it does not share the final learned parameters among windows.
3. The global uniform grid has **41,446 distinct times**. Right-sided routing assigns 1,535 to each first 26 windows and 1,536 to the last. Adding each outgoing endpoint gives **1,536 local training times per window**, or 41,472 point-window assignments. Shared endpoints are counted twice in this last number.
4. The trial state is u0,k+(t−tk)Nk(t). The initial state is exact by construction. No missing chain-rule scale was found: input time and differentiation both use physical time.
5. Only the first anchor is the true supplied initial condition. Each later anchor is a detached **prediction** from the preceding network. Reference states are not injected into these handoffs.
6. Autograd differentiates the trial state, and the residual is its derivative minus the ODE RHS. The present MLP evaluates input rows independently, which makes the sum-based autograd derivative appropriate.
7. Sorted per-time losses are the means of three squared residual components. Causal weights use the exclusive cumulative preceding losses and are detached for gradient calculation.
8. Each window uses four epsilon stages (0.01,0.1,1,10), at most 4,000 Adam updates per stage, with StepLR factor 0.9 every 1,000 updates. Adam/scheduler state continues across epsilon stages within a window and is recreated for the next window.
9. L-BFGS then minimizes the **unweighted** local residual with a maximum of 1,000 optimizer iterations. A line search can call its closure several times per iteration.
10. The next anchor is set after refinement. Final state values are continuous; derivatives are not forced to be continuous.

Reference values enter monitoring, diagnostic snapshots, and final evaluation, but not the objective or automatic causal stopping test. Human choice of settings after examining earlier results still means this is a selected experimental configuration, not an untouched statistical holdout.

## 4. Findings that need attention

### A. Incorrect experiment narrative: fix before defense

`main.tex:107` associates RMSE 2.237 with one shared network trained sequentially over 27 windows and says earlier learning was lost. The saved experiment is a full-interval batch network. Correct the sentence to:

> The saved single-network run trained on the full interval and achieved a combined RMSE of 2.237. We subsequently evaluated a configuration with 27 independent window networks, predicted-state handoffs, warm starts, causal Adam stages, and L-BFGS refinement.

Unless separate artifacts exist for a forgetting experiment, remove that story. One failed batch run also cannot establish that all full-interval networks necessarily fail.

### B. Causal threshold was not met everywhere

There are 108 Adam epsilon stages. Two ended at their 4,000-update cap: zero-based windows 10 and 11, epsilon 10, with min weights 0.9718467 and 0.9031337. The code records failure and proceeds. This is a bounded computational policy, not a satisfied convergence condition.

After raw L-BFGS, recomputation at epsilon 10 gives final minima between 0.97199956 and 0.99998176. Window 10 remains below 0.99. The gate certifies an Adam stage endpoint, not the final model after another objective has been optimized.

At epsilon 10, min weight >0.99 bounds the preceding residual sum by approximately 0.00100503. For 1,536 points this gives an average-prefix bound near 6.55e-7. It does not mean 99% state accuracy, constrain every between-point residual, or constrain the last point directly through its weight.

### C. Final causal-weight diagnostic has a source flaw

`src/pinn/train.py:323–324` stores all-ones weight profiles for final-window snapshots, not recalculated final causal weights. This can make the heatmap misleading. It does not change the trained trajectory. The archive lacks `param_trail.npz`, so this particular heatmap's source values cannot be recovered completely. Do not use it as proof that all final gates passed. Repair by recomputing weights with the final epsilon, or label/exclude raw-L-BFGS phases.

### D. Continuous values do not guarantee continuous derivatives

The joins have state mismatch near roundoff, but the largest exact-autograd derivative jump is 0.005615. The piecewise function is continuous, not globally C1. The global residual summary evaluates shared times on the right window only. It excludes the 26 outgoing left-sided endpoint residuals included in local training. Describe it as residual MSE on the unique global grid; report separate one-sided joint checks. Neither a finite sampling average nor invariant preservation is a uniform state-error bound.

### E. Reference solver ignores custom coefficients

`src/baseline/lorenz1960_baseline.py:109–110` passes a default RHS to solve_ivp without forwarding config.k/l. A k=3,l=1 check disagrees with the correctly parameterized solution by about 0.44736 maximum component error on [0,1]. PINN reference functions also use this baseline, so coefficient sweeps would be mis-evaluated.

**The reported candidate uses default k=2,l=1 and is unaffected.** Before coefficient experiments, bind config.k/l into the reference RHS and check a nondefault case against independently specified equations. Default-only tests miss this. This review records the bug without modifying historical source or results.

### F. Paper training counts and saved-state explanation

At `main.tex:179`, replace 50-epoch snapshots with **2,000 logged evaluations plus final-window snapshots**; replace 1,535 gate points with **1,536**; replace L-BFGS epochs with optimizer iterations. At `main.tex:323`, use **33,143 closure evaluations total**, separately from maximum **1,000 iterations/window**.

The generic epochs=20000 and lr_end=1e-4 fields are inactive for causal/StepLR training. They are not actual work counts or a guaranteed terminal learning rate. At `main.tex:181`, distinguish config.json (settings) from progress.pt (resumable model, history, RNG, next-window state).

### G. Independent component benefits remain unmeasured

The single-network/windowed comparison changes parameter count, sampling, loss, warm start, and optimizer schedule simultaneously. One seed does not establish reliability. The 512/1500/1536 experiments also differ in budgets, scheduling, source history, or hardware.

The cumulative loss is unnormalized: increasing point count changes effective causal suppression and gate strictness, not only sampling. This agrees with the paper's form and is not inherently a bug. Normalizing it would be an adaptation requiring retuning. Replace claims that causal weights are necessary, plain residual is always insufficient, or more points necessarily lower error. At fixed residuals, more cumulative terms can lower the weighted objective through suppression while raw residual and state accuracy stay unchanged.

### H. Other manuscript corrections

- Line 93: replace performance of individual components with performance of their combined configuration.
- Lines 104/136: give duplicate fig:method labels separate names.
- Lines 199–200: distinguish raw network N outputs from physical trial states.
- Line 227: say separately constructed evaluation grid; the grids share at least endpoints.
- Line 297: define invariant basis and relative normalization before quoting drift.
- Line 325: a citation cannot substitute for this project's missing no-causal control.
- Line 331: replace passes every test with the measured quantities and convergence limitations.
- Throughout: name Lorenz-1960; use approximately one cycle; remove unsupported optimality, chaotic-loop and state-of-the-art claims.

## 5. Literature fidelity

[Wang, Sankaran and Perdikaris, original causal-PINN paper](https://arxiv.org/pdf/2203.07404) supports chronological residual weighting, detached weights, a weight gate, and time marching. Pointwise temporal ODE losses are a reasonable specialization; temporal bins and uniform sampling are not mandatory. This implementation ends epsilon at 10 rather than the paper's recommended 100 and adds plain-residual L-BFGS. It is an adaptation, not an exact replication. Their Lorenz benchmark uses Lorenz-1963 equations.

[PinnDE version 2](https://arxiv.org/html/2408.10011v2) Section 4.2 provides the chosen Lorenz-1960 equations and initial state over [0,1]. Its DeepONet results are not a matched comparison with this PINN. The text and appendix code use apparently different coefficient/sign conventions: cite the written equations as the benchmark definition and reconcile conventions before copying code. Version 1 lacks this Lorenz example; pin v2 in the bibliography.

[Original Lorenz reference](https://doi.org/10.1111/j.2153-3490.1960.tb01307.x): conservative invariants support a physics check, but do not certify phase accuracy or justify exactly 27 windows and 1,536 points.

## 6. Figures and reproducibility

The saved results image was inspected. Its title MSE=1.62e-9 is **component-averaged state MSE**, combined RMSE squared divided by three. It is not residual MSE=1.87e-7 or combined state MSE. The loss plot mixes windows, epsilon stages, and line-search closure calls. Transition spikes alone cannot establish forgetting or optimizer instability.

The evaluation grid has 13,265 points, fewer than the 41,446 training times. Dense is an existing diagnostic label, not denser sampling than training here. Its near-unity residual ratio concerns two grids within one interval, not generalization to new states, coefficients, or later times.

Nine-significant-digit CSV times can shift boundary queries across network routing. An audit using rounded times showed residual discrepancies up to 0.00250; regenerating the exact grid removes this and matches point residuals within 5e-12. Use config-generated grids for exact reproduction.

The archive intentionally omits breakdown snapshots and progress files. Final weights/compact telemetry support final-state verification, but all training animations cannot be regenerated. A final checkpoint is not a complete optimizer resume checkpoint. Imported Colab absolute paths also make runner reuse checks reject an unchanged manifest on another host. Use config_for(local_run) with load_run for evaluation; do not overwrite the original experiment or accidentally retrain it.

Older docs contradict current evidence. The comprehensive causal-vs-batch draft mislabels historical final-window loss, calls this a chaotic loop, overinterprets NTK evidence, and promises accuracy from untested changes. Some supervisor docs describe old short-horizon settings. Treat them as historical drafts. The older survey claim that Lorenz-1960 is absent from the literature is contradicted by PinnDE v2; do not claim first-ever use without a systematic current review. Neither NTK spectra nor a projected landscape proves the precise reason a batch model plateaued.

## 7. Questions to rehearse together

| Defense question | Supported answer |
|---|---|
| What did you contribute? | Implementation and reproducible evaluation of established techniques on this specified Lorenz-1960 trajectory, not a new causal algorithm. |
| Why 27 windows and 1,536 points? | Selected settings that produced this saved result; optimality and independent density effects were not tested. |
| Does hard IC prevent accumulated error? | Only relative to the anchor; later anchors are predictions and can be wrong. |
| Are the networks independent? | Yes; 11,283 parameters each, 304,641 total. Warm starting copies weights but does not tie them. |
| Does 0.99 mean 99% accuracy? | No. It is a prefix-residual weight threshold; two stages reached their caps without meeting it. |
| Are derivatives continuous? | No global C1 guarantee; maximum measured derivative jump about 0.005615. |
| Why use a numerical reference? | State accuracy is independent of low residual. Alternate/tighter solvers agree much more closely than PINN error. |
| Did you solve chaos? | This is conservative Lorenz-1960, not the Lorenz-1963 chaotic attractor. |
| Is it better than DOP853? | No matched numerical-solver speed/accuracy advantage has been established. |
| Did causal weighting cause the gain? | The combined configuration succeeded; a matched no-causal control is absent. |
| Does it predict beyond this interval? | Not validated; out-of-interval queries are extrapolation. |
| What remains uncertain? | Single-seed reliability, independent mechanism effects, and transfer to other initial states/parameters. |

## 8. Priorities before 5 October

1. Correct the false experiment history, optimizer/snapshot counts, duplicate labels, and unsupported necessity/ablation claims. Make slides and figures use consistent metric definitions.
2. Preserve the checkpoint and checksum. Include measured joint and convergence limitations; do not use the causal heatmap to claim final gate saturation.
3. Document the reference parameter bug and diagnostic-weight bug. Repair parameter forwarding before any coefficient sweeps. The current default-coefficient checkpoint does not need retraining to correct manuscript facts.
4. Rehearse the questions above, especially trial state versus raw output, soft gating versus accuracy, and predicted anchors versus true states.
5. If time/compute permit, prioritize a matched no-causal windowed control and repeated seeds. Hold architecture, grid and optimizer policy fixed and document stopping/budget differences. These are future experiments until actually run.

A defensible conclusion is:

> On this Lorenz-1960 initial-value problem over approximately one cycle, the saved 27-window float64 configuration achieved combined state RMSE 6.97e-5 and evaluation residual MSE 1.87e-7. Independent checkpoint evaluation reproduced these values. State handoffs are continuous, while derivatives and causal-stage convergence have measured limitations. The comparison demonstrates a combined configuration's performance, not isolated causal effects or general solver superiority.

## 9. Reproduction and open verification limits

```text
python scripts/check_saved_causal.py
python scripts/check_defense_reference.py
python -c "import torch,pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(['-q']))"
```

Machine-readable outputs: checkpoint-verification.json and reference-verification.json beside this report. Scripts assert archived metric agreement, grid counts, continuity, reference agreement, invariants and RK4 refinement. They do not retrain or alter saved results.

Full suite: **40 passed, 1 failed** out of 41 collected tests. The existing test_sweep_writes_runs_and_comparison expects three snapshots but the implementation writes four, including the pre-last-update and post-final-update snapshots. This is a snapshot contract/test mismatch, not a failure of the loaded causal trajectory. It is left visible rather than changing the test during a review. Numerical/window-focused checks also passed in the training audit. See verification-notes.md for details.

The review cannot recover missing optimizer/snapshot states, establish multi-seed reliability, prove a supremum bound between sample times, or certify unsupplied final slides. No manuscript compilation was performed: no source edits were requested in this review and the manuscript uses multiple local class, bibliography and figure files. Its duplicate labels and wording issues remain open corrections. The artifact passed final-checkpoint evaluation; that is narrower than complete scientific validation of all possible claims.
