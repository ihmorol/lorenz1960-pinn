# Results master — every run, every branch, one place

Generated 2026-10-07 from the committed run records. Sources:
`runs/reference_sweep/manifest.csv`, `runs/training_sweep/training_sweep.csv` (branch
`feat/k-l-initial-sweep`); `../lorenz1960-pinn-stability/runs/stability/*.csv` (branch
`feat/multiseed-failure-runs`); `runs/*/run_summary.csv` (runs of record, main).
All errors are `rmse_combined_l2` against the DOP853 reference on the eval grid.

## T1. k/l/IC grid: reference validation + training (27 cells, seed 0, 20k Adam + 1k L-BFGS)

IC legend: ic1 = (0.5, 0.75, 1.0), ic2 = (−1.0, 0.5, 0.75), ic3 = (0.25, −1.5, 1.2).
"drift" = max relative drift of the conserved quadratics over t ∈ [0, 10];
"cross" = max |RK4(h=1e-3) − DOP853|; "period" from FFT on a 10 s window
(values ≈ 10 coincide with the window length → lower bounds, not resolved periods);
"amp" = max |state| over the trajectory; train columns over t ∈ [0, 1].

| cell | k | l | k/l | IC | drift | cross | period | amp | final loss | train RMSE | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k1.5_l0.8_ic1 | 1.5 | 0.8 | 1.875 | ic1 | 2.6e-10 | 2.6e-07 | 10.002 | 1.65 | 1.0e-08 | 2.04e-05 | 348 |
| k1.5_l0.8_ic2 | 1.5 | 0.8 | 1.875 | ic2 | 1.6e-09 | 8.0e-07 | 10.002 | 1.21 | 8.6e-08 | 4.21e-05 | 335 |
| k1.5_l0.8_ic3 | 1.5 | 0.8 | 1.875 | ic3 | 2.2e-10 | 3.9e-07 | 10.002 | 1.74 | 1.5e-08 | 1.92e-05 | 305 |
| k1.5_l1_ic1 | 1.5 | 1 | 1.5 | ic1 | 2.0e-10 | 1.9e-07 | 10.002 | 1.35 | 2.4e-08 | 2.98e-05 | 309 |
| k1.5_l1_ic2 | 1.5 | 1 | 1.5 | ic2 | 8.0e-10 | 3.2e-07 | 10.002 | 1.29 | 6.6e-09 | 1.21e-05 | 302 |
| k1.5_l1_ic3 | 1.5 | 1 | 1.5 | ic3 | 2.3e-10 | 4.3e-07 | 10.002 | 1.60 | 3.7e-08 | 3.35e-05 | 311 |
| k1.5_l1.5_ic1 | 1.5 | 1.5 | 1 | ic1 | 4.7e-10 | 1.7e-07 | 10.002 | 1.00 | 1.8e-09 | 7.72e-06 | 318 |
| k1.5_l1.5_ic2 | 1.5 | 1.5 | 1 | ic2 | 4.6e-10 | 1.2e-07 | 10.002 | 1.12 | 1.9e-09 | 8.01e-06 | 308 |
| k1.5_l1.5_ic3 | 1.5 | 1.5 | 1 | ic3 | 4.6e-10 | 4.1e-07 | 10.002 | 1.52 | 5.0e-09 | 1.04e-05 | 298 |
| k2_l0.8_ic1 | 2 | 0.8 | 2.5 | ic1 | 3.3e-10 | 5.5e-07 | 10.002 | 1.62 | 2.5e-08 | 2.65e-05 | 309 |
| k2_l0.8_ic2 | 2 | 0.8 | 2.5 | ic2 | 2.6e-09 | 2.0e-06 | 5.001 | 1.19 | 7.0e-07 | 1.27e-04 | 300 |
| k2_l0.8_ic3 | 2 | 0.8 | 2.5 | ic3 | 2.8e-10 | 3.7e-07 | 10.002 | 2.17 | 1.4e-08 | 1.89e-05 | 317 |
| k2_l1_ic1 | 2 | 1 | 2 | ic1 | 3.6e-10 | 3.0e-07 | 10.002 | 1.64 | 6.7e-09 | 1.60e-05 | 310 |
| k2_l1_ic2 | 2 | 1 | 2 | ic2 | 2.1e-09 | 1.0e-06 | 5.001 | 1.20 | 1.7e-07 | 5.94e-05 | 311 |
| k2_l1_ic3 | 2 | 1 | 2 | ic3 | 1.7e-10 | 3.8e-07 | 10.002 | 1.80 | 1.4e-08 | 1.86e-05 | 301 |
| k2_l1.5_ic1 | 2 | 1.5 | 1.333 | ic1 | 2.4e-10 | 1.7e-07 | 10.002 | 1.16 | 3.5e-09 | 1.13e-05 | 300 |
| k2_l1.5_ic2 | 2 | 1.5 | 1.333 | ic2 | 4.1e-10 | 2.0e-07 | 10.002 | 1.38 | 8.8e-09 | 1.70e-05 | 264 |
| k2_l1.5_ic3 | 2 | 1.5 | 1.333 | ic3 | 2.3e-10 | 4.3e-07 | 10.002 | 1.56 | 5.4e-08 | 4.17e-05 | 262 |
| k3_l0.8_ic1 | 3 | 0.8 | 3.75 | ic1 | 7.6e-10 | 1.8e-06 | 5.001 | 1.60 | 6.0e-07 | 1.22e-04 | 264 |
| k3_l0.8_ic2 | 3 | 0.8 | 3.75 | ic2 | 2.9e-09 | 5.3e-06 | 2.501 | 1.18 | 4.4e-07 | 6.14e-05 | 260 |
| k3_l0.8_ic3 | 3 | 0.8 | 3.75 | ic3 | 3.2e-10 | 5.5e-07 | 10.002 | 2.27 | 4.8e-08 | 5.35e-05 | 265 |
| k3_l1_ic1 | 3 | 1 | 3 | ic1 | 3.4e-10 | 9.8e-07 | 10.002 | 1.61 | 1.2e-07 | 5.26e-05 | 261 |
| k3_l1_ic2 | 3 | 1 | 3 | ic2 | 3.1e-09 | 3.1e-06 | 3.334 | 1.18 | 3.8e-07 | 5.58e-05 | 267 |
| k3_l1_ic3 | 3 | 1 | 3 | ic3 | 2.6e-10 | 4.0e-07 | 10.002 | 2.27 | 3.0e-08 | 3.83e-05 | 264 |
| k3_l1.5_ic1 | 3 | 1.5 | 2 | ic1 | 3.6e-10 | 3.0e-07 | 10.002 | 1.64 | 6.7e-09 | 1.60e-05 | 265 |
| k3_l1.5_ic2 | 3 | 1.5 | 2 | ic2 | 2.1e-09 | 1.0e-06 | 5.001 | 1.20 | 1.7e-07 | 5.94e-05 | 261 |
| k3_l1.5_ic3 | 3 | 1.5 | 2 | ic3 | 1.7e-10 | 3.8e-07 | 10.002 | 1.80 | 1.4e-08 | 1.86e-05 | 265 |

All 27 cells PASS validation; all 27 trainings converge (max RMSE 1.27e-4).
Note: (k, l) and (λk, λl) are the same ODE — cells k2_l1_* and k3_l1.5_* are
byte-identical, which is the invariance confirmed empirically.

## T2. Seed stability (branch 2, default 4x60, t ∈ [0, 1], full budget)

| seed | final loss | RMSE | wall s |
|---|---|---|---|
| 0 | 6.8e-09 | 1.73e-05 | 313 |
| 1 | 2.2e-07 | 2.48e-04 | 312 |
| 2 | 2.5e-08 | 8.37e-05 | 307 |
| 3 | 1.1e-08 | 2.03e-05 | 312 |
| 4 | 5.2e-09 | 1.06e-05 | 306 |
| **mean ± std** | | **7.6e-05 ± 9.4e-05** | |

Seed 1 is a ~15x outlier → single-seed numbers on this problem are not trustworthy.

## T3. Failure tiers (branch 2, plain 4x60 vs causal-windowed 4x60_win9_causal)

Moderate tier (superseded — plain solves everything at full budget; kept as record):

| case | plain RMSE | causal RMSE |
|---|---|---|
| t ∈ [0, 5] | 2.3e-04 | 7.7e-04 |
| IC (4, 4, 4) | 8.0e-04 | 2.7e-03 |
| k=4, l=0.5 (ratio 8) | 9.1e-04 | 3.5e-03 |
| k=l=1 | 8e-06 | 5.1e-04 |

Hard tier (the real story):

| case | plain RMSE | causal RMSE | verdict |
|---|---|---|---|
| t ∈ [0, 13.26446] | **2.24** | **0.040** | plain fails; causal rescues (56x) |
| IC (10, 10, 10) | **19.6** | **235** | both fail; causal worse |
| k=6, l=0.5 (ratio 12) | 8.6e-04 | 2.1e-03 | ratio alone does not cause failure |
| k=l=1 (control) | 8e-06 | 5.1e-04 | easy for both |

Caveat: causal arm is budget-limited (causal_max_iters=2000, early advance) —
not budget-matched against plain's 20k epochs; report wall clock alongside
(plain ≈ 310 s, causal ≈ 15–414 s).

## T4. Runs of record (pre-existing, main)

| run | config | horizon | RMSE | note |
|---|---|---|---|---|
| 4x60 | 4x60, f32, 3k colloc | t ∈ [0, 1] | 1.73e-05 | run of record, unit span |
| 4x60_f64_unit | 4x60, f64, unit IC scale, 40k + 5k L-BFGS | t ∈ [0, 13.26] | **2.24** | plain training fails at full horizon |
| 4x60_f64_unit_win27_causal_warm | 27 windows, causal (0.01–10), warm start, 1500 L-BFGS | t ∈ [0, 13.26] | **8.4e-05** | candidate-1536pt, conference headline |
| arch sweep (9 cells, 3x50…5x70) | t ∈ [0, 1] | 1.0e-05 … 1.3e-04 | architecture alone varies error 13x |
| generalization-gap probe | off-grid residual gap | 1.00x both runs | no collocation overfitting |

Replication check: independent `long_horizon_13` plain run (branch 2, this week)
reproduces the recorded `4x60_f64_unit` failure almost exactly (2.239 vs 2.237).

## T5. Verification assets

- Stress suite: 31/31 pass (was 29 + strict xfail documenting the reference bug).
- Fixed on `fix/reference-k-l-coeffs`: SciPy reference ignored k/l (all pre-fix
  references for k ≠ 2 silently integrated the k=2, l=1 system).
- Reference validation: conserved quadratics (drift ≤ 3.1e-9 over 10 s),
  RK4-vs-DOP853 cross-solver (≤ 5.3e-6), FFT period, coefficient closed forms.
- New physics: coefficients invariant under (k, l) → (λk, λl) — system depends
  only on k/l (proposition + test); conserved invariant E = k⁴x² + l⁴y².

## Cross-table findings

F1. Parameter space collapses to k/l + IC (invariance; duplicate cells confirm).
F2. Accuracy degrades monotonically with k/l (median 7.7e-06 at ratio 1 → 6.1e-05
    at 3.75); near-degenerate k/l = 1 is easiest.
F3. Seed variance is large (15x spread) — multi-seed reporting is mandatory.
F4. Plain training fails at the 13.26 horizon; causal windowing rescues it
    (replicates the recorded failure independently).
F5. Extreme IC amplitude defeats both arms — causal training is not a universal fix.
F6. Moderate "failures" were undertraining artifacts — failure claims need full
    budget; the 200-epoch smoke story reversed at 20k epochs.

## What it takes to publish from here (ordered by impact)

1. **Generalize beyond one ODE.** The survey's shaped statement already commits to
   6-10 IVP problem classes (linear, oscillatory, stiff, chaotic — Lorenz-63,
   Van der Pol, Lotka-Volterra, Robertson). A single-system empirical study is a
   hard sell everywhere; the ProblemSpec/registry branch is the enabler. This is
   the one change that converts "project report" into "community contribution".
2. **Budget-matched comparisons.** Plain 20k epochs vs causal early-advance is not
   parity. Report error-vs-wall-clock convergence curves; reviewers will attack T3
   first.
3. **Multi-seed everything.** T2's 15x seed spread means every headline number
   (including the candidate run and the failure table) needs mean ± std over >= 5
   seeds. Check whether the large-IC failure is seed-universal or seed-luck.
4. **Formalize the two small theory results.** Ratio-invariance proposition with a
   3-line proof (shrinks the sweep space); conserved invariant E = k^4 x^2 + l^4 y^2
   as a reference-free metric (conservation drift + period error), aligning metrics
   with dynamical-systems theory as the survey proposes.
5. **One external baseline** from the PINN literature (e.g. Krishnapriyan
   time-marching or self-adaptive weights) so the comparison is not only internal
   arms; classical integrators stay as the oracle.
6. **Paper surgery on main.tex:** retitle from method-claim toward the benchmark/
   characterization framing; swap single-seed candidate numbers for multi-seed;
   add the failure table and the honest negative result (F5); limitations: budget
   asymmetry, non-chaotic target system, single architecture family.
7. **Artifact release:** pinned configs + seeds (already config-locked), Colab
   notebook, run index; respect the UPenn causal-weighting patent caveat in the
   license note.
8. **Venue fit:** IEEE format already matches IJCNN / ICCS (Jan 2027 deadlines);
   a NeurIPS-workshop version needs item 1 done first.
