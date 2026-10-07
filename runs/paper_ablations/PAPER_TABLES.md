# Paper tables

Generated 2026-10-07 10:27 UTC from `runs\paper_ablations`.
Wall clock is per run; runs on different devices or under parallel jobs are not
timing-comparable. Seed 0 of r4_candidate is the candidate run of record.

## T1. Per-run results (actual optimizer work; RMSE and max error are combined L2)

| run | seed | device | params | points | Adam steps | L-BFGS evals | capped stages | RMSE | max error | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| F1_seed0 (run of record) | 0 | cuda | 11,283 | 39,793 lhs | 40,000 | -- | -- | 2.24 | 3.35 | 4529 |
| r1_no_causal_seed1 | 1 | cpu | 304,641 | 41,446 uniform | 35,100 | 33,211 | 0 | 0.000175 | 0.000737 | 1002 |
| r1_no_causal_seed2 | 2 | cpu | 304,641 | 41,446 uniform | 35,100 | 33,486 | 0 | 0.000189 | 0.000561 | 1118 |
| r1_no_causal_seed3 | 3 | cpu | 304,641 | 41,446 uniform | 35,100 | 33,385 | 0 | 0.000516 | 0.00183 | 1305 |
| r4_candidate_seed0 (run of record) | 0 | cuda | 304,641 | 41,446 uniform | 34,765 | 33,143 | 2 | 6.97e-05 | 0.000309 | 881 |
| r4_candidate_seed1 | 1 | cpu | 304,641 | 41,446 uniform | 33,564 | 33,163 | 2 | 8.13e-05 | 0.00023 | 1164 |
| r4_candidate_seed2 | 2 | cpu | 304,641 | 41,446 uniform | 36,759 | 33,082 | 1 | 0.00026 | 0.000502 | 1169 |

## T2. Arm summary (seed 0 runs of record included where they belong to the arm)

| arm | what changes | n | RMSE mean +/- std | RMSE range | mean Adam steps |
|---|---|---|---|---|---|
| r4_candidate | candidate config (27 windows, causal, warm start), other seeds | 3 | 0.000137 +/- 0.000107 | 6.97e-05 .. 0.00026 | 35,029 |
| r1_no_causal | causal weighting off; 1,300 Adam/window (35,100 total) | 3 | 0.000293 +/- 0.000193 | 0.000175 .. 0.000516 | 35,100 |
| r2_no_warm | warm start off | 0 (PENDING) | -- | -- | -- |
| r6_sequential | r1 with one network shared by all windows | 0 (PENDING) | -- | -- | -- |
| r3_causal_single | F1 + causal schedule, 4 x 10,000 Adam cap | 0 (PENDING) | -- | -- | -- |

## T3. Paired seeds: RMSE ratio arm / comparison (>1 means the arm is worse)

| pair | common seeds | ratios | arm worse in |
|---|---|---|---|
| r1_no_causal / r4_candidate | 1, 2 | 2.16, 0.726 | 1/2 |

## T4. R7 reference-free physics (E = k^4 x^2 + l^4 y^2 drift; closure ||u(T) - u(0)||)

| run | E drift pred | E drift ref | closure pred | closure ref |
|---|---|---|---|---|
| 4x60_f64_unit_win27_causal_warm_f1cbeebd | 0.000405 | 1.76e-10 | 5.64e-05 | 1.65e-06 |
| r1_no_causal_seed1 | 0.000478 | 1.76e-10 | 0.00035 | 1.65e-06 |
| r1_no_causal_seed2 | 0.000604 | 1.76e-10 | 0.000118 | 1.65e-06 |
| r1_no_causal_seed3 | 0.000856 | 1.76e-10 | 0.00134 | 1.65e-06 |
| r4_candidate_seed1 | 0.000338 | 1.76e-10 | 9.82e-05 | 1.65e-06 |
| r4_candidate_seed2 | 0.000451 | 1.76e-10 | 0.000413 | 1.65e-06 |
