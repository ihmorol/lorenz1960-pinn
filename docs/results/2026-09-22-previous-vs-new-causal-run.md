# Previous causal run versus new causal run

This report compares only these two completed causal-window runs:

- **Previous causal:** `runs/4x60_f64_unit_win27_causal_warm/`
- **New causal:** `runs/causal-window/candidate-512pt/4x60_f64_unit_win27_causal_warm_6de06c38/`

The copies of the previous causal CSVs used for the original comparison were
byte-identical to the files under `runs/`, so they are not a third experiment. All values below
come from the two `run_summary.csv`, `point_summary.csv`, loss, causal-stage, and
configuration files. No model was retrained for this comparison.

## Configuration differences

| Setting | Previous causal | New causal |
|---|---:|---:|
| Network | 4 x 60 per window | 4 x 60 per window |
| Windows / parameters | 27 / 304,641 | 27 / 304,641 |
| Time span / evaluation points | 0–13.26446 / 13,265 | 0–13.26446 / 13,265 |
| Collocation points | 39,793, about 1,474/window | 13,825, about 512/window |
| Causal epsilon schedule / delta | 0.01, 0.1, 1, 10 / 0.99 | 0.01, 0.1, 1, 10 / 0.99 |
| Adam cap / L-BFGS cap | 4,000/stage / 1,500/window | 4,000/stage / 1,500/window |
| Adam learning rate | 1e-3, StepLR 0.9 every 5,000 | 1e-3, StepLR 0.9 every 1,000 |
| Precision / sampling / warm start | float64 / uniform / yes | float64 / uniform / yes |

The new run uses 65.26% fewer collocation points. It also uses the repaired
training code, including the outgoing-window endpoint residual and the
post-update causal threshold check. The point count, learning-rate schedule,
and training implementation therefore changed together. The files support an
outcome comparison, but they do not isolate which change caused an outcome.

## Final numerical quality

For the previous run, `final_loss` is the last-window objective and is not
comparable with the corrected new `final_loss`. The common physics metric is
`residual_mse_collocation`.

| Measure | Previous causal | New causal | New relative to previous |
|---|---:|---:|---:|
| Combined MAE | 7.09353e-5 | 1.03740e-4 | 46.25% higher |
| Combined RMSE | 8.43460e-5 | 1.56946e-4 | 86.07% higher |
| Maximum state-error norm | 2.35488e-4 | 5.87768e-4 | 149.60% higher |
| Median state-error norm | 6.48866e-5 | 8.56190e-5 | 31.95% higher |
| 90th-percentile error norm | 1.38011e-4 | 2.39482e-4 | 73.52% higher |
| 99th-percentile error norm | 2.21998e-4 | 5.49111e-4 | 147.35% higher |
| Final endpoint-error norm | 3.46218e-5 | 3.09764e-4 | 8.95 times higher |
| Collocation residual MSE | 1.46758e-7 | 3.71211e-7 | 152.94% higher |
| Dense-grid residual MSE | 1.46960e-7 | 3.69964e-7 | 151.74% higher |
| Invariant 1 maximum drift | 2.51025e-4 | 2.97356e-4 | 18.46% higher |
| Invariant 2 maximum drift | 1.91864e-4 | 3.45902e-4 | 80.28% higher |

The new run is worse on every combined accuracy, common residual, and
invariant-drift measure in the saved summaries. Its collocation and dense-grid
residuals differ by only 0.34%, while the previous run differs by 0.14%; both
therefore show similar residual magnitude on their training and dense grids.

### State-by-state result

| State | Measure | Previous | New | Change |
|---|---|---:|---:|---:|
| x | MAE | 2.07497e-5 | 1.82028e-5 | 12.27% lower |
| x | RMSE | 2.83995e-5 | 2.89617e-5 | 1.98% higher |
| x | maximum absolute error | 1.04261e-4 | 9.48920e-5 | 8.99% lower |
| y | RMSE | 5.91803e-5 | 1.37553e-4 | 132.43% higher |
| y | maximum absolute error | 1.77429e-4 | 5.52884e-4 | 211.61% higher |
| z | RMSE | 5.29662e-5 | 6.98036e-5 | 31.79% higher |
| z | maximum absolute error | 1.95126e-4 | 3.25815e-4 | 66.98% higher |

The only accuracy improvements are x MAE and x maximum error. The small x
improvements are outweighed by the larger y and z degradation.

## Where the new error occurs

The per-point summaries were divided using the same 27 window boundaries.
The new run has lower window-level trajectory RMSE in 15 of 27 windows, mainly
windows 0–8 and 18–22. It is worse in 12 windows. The degradation is highly
concentrated:

| Window | Residual MSE ratio, new/previous | Trajectory RMSE ratio, new/previous |
|---:|---:|---:|
| 10 | 88.40 | 2.74 |
| 24 | 38.81 | 4.14 |
| 25 | 432.66 | 6.74 |
| 26 | 125.07 | 9.41 |

Windows 10, 24, 25, and 26 contribute 80.11% of the new run's complete
collocation residual. The maximum trajectory error occurs near `t=12.6696` in
window 25. In the previous run it occurs near `t=11.4117` in window 23.

This establishes the location of the degradation. It does not establish which
configuration or code change caused it.

## Optimization and resource tradeoffs

| Measure | Previous causal | New causal | Change |
|---|---:|---:|---:|
| Adam steps | 29,277 | 21,183 | 27.65% fewer |
| L-BFGS closure evaluations | 49,400 | 49,428 | 0.06% more |
| Total logged objective evaluations | 78,677 | 70,611 | 10.25% fewer |
| Epsilon stages reaching the 4,000-step cap | 6 of 108 | 2 of 108 | 4 fewer |
| Recorded wall time | 928.58 s | 932.39 s | 0.41% longer |
| Recorded peak CUDA allocation | 549.06 MB | 212.32 MB | 61.33% lower |

The new run's two capped stages are epsilon 10 in windows 10 and 25. Their
post-update final minimum weights are 0.98882 and 0.98942, just below the 0.99
threshold. The previous run has six 4,000-step epsilon-10 stages, in windows
9–11 and 23–25, with recorded final minimum weights from 0.77799 to 0.90536.
Because the new code certifies the threshold after the optimizer update while
the previous code did not, this is an observed stage-completion comparison,
not a controlled comparison of one stopping implementation.

Fewer points and fewer Adam steps did not reduce recorded wall time. The saved
summaries identify both devices only as `cuda`; they do not record GPU model or
runtime environment, so they cannot support a hardware-normalized speed claim.

The new result folder occupies about 445 MB: about 120 MB of breakdown files,
312 MB of checkpoint/history files, and 12 MB of figures. The available
previous folder occupies 27 MB, but its breakdown snapshots and complete
resume state are absent. This is a comparison of the delivered artifact sizes,
not a like-for-like storage benchmark.

## Evidence-based conclusion

The new run improves recorded peak GPU memory, reduces collocation points,
uses fewer Adam steps, and leaves fewer causal stages at their iteration cap.
It also produces more complete provenance and resume artifacts.

The new run does **not** improve overall numerical quality. Its combined RMSE
is 1.86 times the previous result, its maximum error is 2.50 times larger, its
whole-domain residual MSE is 2.53 times larger, and both invariant drifts are
higher. It also provides no measured wall-time reduction. For this completed
pair of runs, the previous causal result remains the more accurate solution;
the new result is the lower-memory result.
