# Journal plan: where we are, what is left, how to write it

28 October 2026. Written from a read of both repositories, every branch, the
submitted ICEEICT paper and the Overleaf export in `writings/`.

## 1. What exists today

| Phase | Where | What it contains | Status |
|---|---|---|---|
| FYDP-1 | `resources/` (`ihmorol/fydp_workspace`) | Literature corpus (`All papers/`, `references/`, `literature/`), RK4 + DOP853 baseline (`src/baseline`, verified to RMSE 2.3e-11), the plain-ANN study design (69 runs: depth x width x activation, then optimizers), early PINN prototype `fydp2/` (4x60, t in [0,1], RMSE 1.7e-5) | Plain-ANN 69-run study was designed but no result files exist. `fydp2/` was ported into `lorenz1960-pinn/src/pinn` and is superseded. |
| FYDP-2 | `lorenz1960-pinn` (`ihmorol/lorenz1960-pinn`) | Full PINN pipeline: hard-IC trial, windowing, causal weights, warm start, Adam then L-BFGS, float64, reference checks, figures | Active codebase |
| Conference | `main` (tag `iceeict2027-submitted`), from `dev/iceeict2027-revision` + `dev/iceeict2027-submission` | Final paper `writings/iceeict2027-final/`, ablation scripts (`scripts/paper_ablations.py`, `colab_suite.py`, `ablation_report.py`, `sequential_run.py`), review records | Submitted and merged |
| Journal start | `writings/journal/` (Overleaf export of 28 Oct 2026, now tracked) | UIU FYDP **report/thesis** template: Ch1 intro, Ch2 background + lit review, Ch3 design, Ch4 implementation + PINN results on [0,1], Ch5 standards/CEP (P1-P7, A1-A5, WK1-WK9), Ch6 conclusion | Inconsistent: title, abstract, Ch1, Ch3 and Ch6 still describe the plain-ANN "no physics loss" study, while Ch4 reports the PINN |

### Results already on record (conference paper, Lorenz-1960, k=2, l=1, IC (0.5, 0.75, 1.0), T = 13.26446)

| Run | Seeds | Combined RMSE vs DOP853 |
|---|---|---|
| Full-interval single network (F1) | 0 | 2.237 |
| WPINN: 27 windows, separate nets, causal, warm start | 0-4 | median 8.13e-5 (best 6.97e-5) |
| R1 no causal | 1-3 | median ~2.3x WPINN; Mann-Whitney p = 0.196 (not significant) |
| R2 no warm start | 1-3 | worse on all 3 seeds; Adam updates 34.8k -> 103k |
| R3 causal, single network | 0 | 0.275 |
| **R6 one shared network, no retention** | 1-3 | **0.350-0.648** |

### Results that exist only on local, unpushed branches (risk of loss)

- `feat/k-l-initial-sweep`: 27-cell (k, l, IC) sweep on t in [0,1]. All converge
  (RMSE 8e-6 to 1.3e-4). The system depends only on k/l; difficulty grows with
  k/l and with IC amplitude.
- `feat/multiseed-failure-runs`: 5-seed spread on 4x60 (15x outlier at seed 1);
  failure tier: long horizon (plain fails, causal-windowed 56x better), large IC
  (10,10,10) defeats both, k/l = 12 is fine.
- `feat/swappable-ode-systems`: Lorenz-1963 through the unchanged pipeline.
- `feat/paper-results-tables`, `feat/paper-ablation-runs`, `fix/reference-k-l-coeffs`
  (the k/l fix is already in `dev/iceeict2027-revision`).

Uncommitted in the main checkout: `docs/EXPERIMENT_PLAN.md`, `RESULTS_MASTER.md`,
`SIMULATION_UPGRADE_PLAN.md`, `DEFENSE_QA.md`, `docs/research/`, `poster/`.

## 2. Which repository to use

**`lorenz1960-pinn` for all new code, runs and writing.** It holds the pipeline,
the run records and the paper code. Use `resources/` as a read-only source for
the literature review, the FYDP-1 background and the baseline-solver
verification.

Consolidation (done 28 October 2026):

1. `dev/iceeict2027-revision` and `dev/iceeict2027-submission` merged into
   `main`; the paper commit is tagged `iceeict2027-submitted`.
2. Local-only branches pushed; untracked docs from the main checkout committed.
3. The Overleaf source lives in tracked `writings/journal/`. Re-export from
   Overleaf into that folder (or link Overleaf to the repo through Git) after
   each editing session.
4. Branch new work from `main`: `feat/single-network-retention`, then
   `feat/two-point-boundary-runs`.

## 3. Experiment A: one network that keeps every window's learning

R6 already shows that a single network reused across windows *without* a
retention term fails (RMSE 0.35-0.65). The open question is whether a single
network can keep earlier windows while learning later ones. That is the
backward-compatible PINN (bc-PINN) of Mattey and Ghosh (2022, CMAME 390,
114474), which the conference paper already cites and says R6 is not.

**Method.** One network N, windows trained in order with handoff as now. After
window k, store the composed predictions u_hat on a subsample of the collocation
points of windows 0..k. While training window k+1, minimise

    L = L_res(window k+1) + lambda * mean || u(t_prev) - u_hat(t_prev) ||^2

Code: `shared_network=True` already exists (`pinn.py`, `paper_ablations.py`
arm r6). Add a `retain_weight` (lambda) and a `retain_points` count to `Config`,
and the retention term inside `train_windows`.

**Arms (5 seeds each, 0-4):**

| Arm | Description |
|---|---|
| S0 | Shared network, no retention (R6 rerun on seeds 0-4) |
| S1 | Shared + retention, lambda in {1, 10} (pick on seed 0, then freeze) |
| S2 | S1 + causal schedule |
| S3 | S2 with a wider network (~300k parameters, about 4x270) so the parameter count matches WPINN |
| WPINN | Existing 5 seeds, no rerun |

**New measurement: forgetting matrix.** After finishing window k, evaluate the
error on every window j <= k: E[k, j]. Plot it as a 27x27 lower-triangular
heatmap. This turns the conference caveat ("final errors do not measure when
earlier-window accuracy changes") into a direct result. Also report the final
RMSE, Adam updates and wall time, all on the same hardware.

Cost is about 4 arms x 5 seeds x ~15-75 min, i.e. 1-2 days of Colab.

## 4. Experiment B: boundary conditions

**Get the problem right first.** The system has three first-order equations, so
it needs exactly three conditions. The current `end_state` option adds a full
terminal state *on top of* the hard initial state (six conditions). That is an
over-determined IVP with an extra terminal observation, not a BVP, and the
journal must not call it one.

**Recommended formulation, B1 (mixed two-point BVP):** give x(0) = 0.5 and
y(0) = 0.75 at the start and z(T) = z_ref(T) at the end. z(0) is unknown. The
IVP reference is the exact solution, so accuracy can be measured, and the
recovered z(0) (true value 1.0) is a clean scalar metric. Cross-check with
`scipy.integrate.solve_bvp` or DOP853 shooting.

**Arms:**

| Arm | Domain | BC enforcement |
|---|---|---|
| B-hard | Single network, T = 1 and T = 13.26 | Trial: x, y = x0, y0 + (t - t0) N; z = zT + (t - tf) N_z |
| B-soft | Single network | BC penalty with weight gamma (the existing soft-IC path) |
| B-shoot | WPINN | Learnable z0, sequential WPINN, outer secant or Newton iteration on z(T) mismatch (5-8 inner solves) |
| B-joint | All windows trained together | Interface continuity penalty plus both-end BCs (XPINN-style; gives up sequential causality) |

Start with T = 1, where single networks are known to work. Then move to the long
horizon. A single network will probably fail at T = 13.26, as in the IVP case.
That failure is itself a result. The research question is which windowed
variant carries the terminal information back.

Optional B2: a periodic orbit, u(0) = u(T) with T unknown plus a phase
condition. Use the two conserved quadratics as diagnostics. Only do this if B1
is done early.

Cost is about 2-3 weeks including code. It is new formulation work, so do it
after A.

## 5. Gaps that reviewers flagged or will flag (cheap to close)

- Parameter-matched control: a single network with ~300k parameters on the full
  interval. The archive has no R5.
- R1 and R2 have only 3 seeds. Run 5, matching WPINN.
- Wall time on one machine for every arm.
- Derivative jump at the 26 interfaces (values are continuous by construction;
  derivatives are not). Report max |du/dt(left) - du/dt(right)|.
- Generalisation: rerun a small part of the k/l x IC sweep (3-4 cells, not 27)
  at the long horizon with WPINN.

## 6. Journal structure

One evidence base, two documents:

- **FYDP final report (UIU template, the Overleaf export).** Keep the chapter
  layout and Ch5 (CEP/OBE). Rewrite the title, abstract, Ch1, Ch3 and Ch6 so
  they match the PINN work. The plain-ANN study becomes the FYDP-1 phase
  inside Ch1 and Ch3.
- **Journal article** (pick a venue early, because the length and template
  follow from it. A CMAME-style venue fits the bc-PINN and long-horizon
  literature you already cite). Proposed outline:

1. Introduction: long-horizon PINN failure; RQ1 windowing/causality, RQ2
   single-network retention, RQ3 boundary conditions, RQ4 generalisation.
2. Background and related work: Lagaris, Raissi, Wang (causal), Penwarden
   (windowing), Mattey-Ghosh (bc-PINN), Krishnapriyan (failure modes),
   Rohrhofer (residual vs. error).
3. Problem: Lorenz-1960, coefficients, k/l invariance, two invariants,
   reference solver and verification.
4. Methods: trial solutions (IC and mixed BC), windowing and handoff, causal
   weights, optimiser schedule, retention loss, BVP strategies.
5. Experimental protocol: one change per arm, 5 seeds, matched budgets, metrics
   (RMSE, max error, residual, invariant drift, interface derivative jump,
   forgetting matrix, z(0) recovery, time).
6. Results:
   1. Short horizon: PINN (and the plain-ANN data baseline, if run)
   2. Long-horizon IVP: WPINN plus ablations (conference results, extended)
   3. Single network with retention (Exp A)
   4. Boundary-value problems (Exp B)
   5. Generalisation and failure cases (k/l, IC, seeds)
7. Discussion and limitations.
8. Conclusion, plus a reproducibility appendix (configs, seeds, commit hashes).

Sections 1-4 and 6.2 can be written now from the conference material. Write
6.3-6.5 as the runs finish.

## 7. Order of work

1. Repository consolidation (section 2). Half a day.
2. Exp A code, tests and a 1-seed smoke run. 2-3 days.
3. Exp A full runs, alongside the cheap gap runs from section 5.
4. Meanwhile, rewrite report chapters 1-3 and draft journal sections 1-4.
5. Exp B: T = 1, then long horizon.
6. Results sections, figures, internal review, submission.
