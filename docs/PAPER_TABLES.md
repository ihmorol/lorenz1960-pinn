# Paper tables

Snapshot: 2026-10-07 08:04:16 UTC; ablations.csv had 2 rows (r1_no_causal_seed1, r1_no_causal_seed2); source: E:\University\FYDP\lorenz1960-pinn\runs\paper_ablations\ablations.csv

## T1: Failed runs and the rescue (RMSE is combined L2; wall clock in seconds)

| run | description | training points | Adam / L-BFGS | RMSE | verdict |
|---|---|---|---|---|---|
| F1 | single network, full interval | 39,793 | 40,000 / 5,000 | 2.24 | fails (plateau) |
| F2 | plateau of F1 (see runs/4x60_f64_unit/figures/error_vs_t.png) | -- | -- | -- | see figure |
| F3 | sequential single network | -- | -- | -- | PENDING (error_metrics.csv not yet written) |
| F4 | windowed-27 without causal (mean +/- std over r1_no_causal_*, n=2) | -- | 1,300 / 1,000 | 0.000182 +/- 9.52e-06 | max abs err (mean): 0.000649 |
| F5 | windowed-27 + causal + warm start (rescue) | 41,446 | 20,000 / 1,000 | 6.97e-05 | succeeds (emphasized rescue) |

## T2: Ablations (RMSE and max error are combined L2; wall clock in seconds)

| arm | what it isolates | seeds n | RMSE mean +/- std | max error mean | wall clock mean (s) |
|---|---|---|---|---|---|
| r1_no_causal | causal weighting removed (windowed + warm start kept) | 2 | 0.000182 +/- 9.52e-06 | 0.000649 | 1.06e+03 |
| r2_no_warm | warm start removed (causal + windowing kept) | PENDING | -- | -- | -- |
| r4_candidate | full method: causal weighting + warm start, windowed | PENDING | -- | -- | -- |
| r3_causal_single | causal weighting on a single network (no windowing) | PENDING | -- | -- | -- |
| r5_param_matched | parameter-matched control | PENDING | -- | -- | -- |
| note | r4_candidate seed-0 run of record (4x60_f64_unit_win27_causal_warm_f1cbeebd) | 0 | 6.97e-05 | 0.000309 | 881.1 |

## T3: Per-seed raw results

| arm | seed | arch | final_loss | rmse_combined_l2 | max_abs_error | wall_clock_s |
|---|---|---|---|---|---|---|
| r4_candidate | PENDING | -- | -- | -- | -- | -- |
| r1_no_causal | 1 | 4x60_f64_unit_win27_warm | 8.81e-07 | 0.000175 | 0.000737 | 1001.9 |
| r1_no_causal | 2 | 4x60_f64_unit_win27_warm | 1.1e-06 | 0.000189 | 0.000561 | 1117.9 |

## Validation

- PASS ablations.csv r1_no_causal_seed1 rmse_combined_l2: got 0.000175485, expected 0.000175485 (3 s.f.)
- PASS candidate run_summary rmse_combined_l2: got 6.96697e-05, expected 6.96697e-05 (3 s.f.)
- PASS full-interval run_summary rmse_combined_l2: got 2.23706, expected 2.23706 (3 s.f.)
