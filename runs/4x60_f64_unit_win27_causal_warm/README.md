# SUPERSEDED run — not the run of record

This directory holds an **older, superseded** 27-window causal run
(`4x60_f64_unit_win27_causal_warm`). It is kept only so that the linked
historical reports, figures, and checkpoints remain reproducible. It is **not**
the source of the numbers reported in the paper.

## Run of record

The paper of record (`writings/conference-paper/main.tex`) uses:

```
runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/
```

Both runs share the same architecture tag `4x60_f64_unit_win27_causal_warm`
(4x60 MLP, 27 causal windows, warm start, float64, unit IC scale) but differ in
configuration and results. Check the settings hash suffix and
`history/config.json` before reusing anything here.

## Key numbers (from each run's `run_summary.csv`)

| Quantity | This run (superseded) | Run of record (`..._f1cbeebd`) |
|---|---|---|
| `rmse_combined_l2` | 8.43460385e-05 | 6.96696531e-05 |
| `wall_clock_s` | 928.57612 | 881.081312 |
| `lbfgs_iters` | 1500 | 1000 |
| `n_collocation` | 39793 | 41446 |

Tooling or humans looking for the paper's reported numbers (RMSE 6.96696531e-05,
881.08 s, 1000 L-BFGS iterations, 41,446 collocation points) should use the
`f1cbeebd` directory, not this one.

No artifacts in this directory have been modified.
