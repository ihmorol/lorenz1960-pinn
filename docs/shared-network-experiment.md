# Experiment: one shared network vs. 27 per-window networks (causal training)

Status: **the shared-network architecture does not work.** It is kept in the code
**opt-in only** (`Config.share_network` / `run_causal.py --shared`) and the default
behaviour is unchanged. Numbers below are the actual values written by the two runs;
nothing is estimated.

## Configuration changes exercised

All three requested changes are applied in `run_causal.py`:

| knob | old | new |
|---|---|---|
| `n_collocation` | 3000 | **1500** |
| `lbfgs_iters` (per window) | 1500 | **500** |
| `causal_eps_schedule` | 4 stages `0.01 0.1 1 10` | **5 stages `0.01 0.1 1 10 100`** |

Everything else is the run-of-record causal setup: 4x60 tanh network, `float64`,
`ic=hard`, `ic_scale=unit`, `collocation=uniform`, `n_windows=27`,
`causal_delta=0.99`, `causal_max_iters=4000`, `warm_start=True`, seed 0,
`t_span = (0, 13.26446)`.

## Architecture change

`WindowedPINN` holds **one network per window** (`nn.ModuleList`, 27 nets).
`SharedWindowPINN` holds **one network for all 27 windows**; each window still
starts from the previous window's end state, carried in an `edge_starts` buffer
and applied through the same hard-IC trial `u = u0 + g(t)·N(t)`.

* 4x60 network = **11,283 parameters**.
* 27 copies = **304,641 parameters** (27 x 11,283 exactly).

Both models are driven by the same window loop in `functions/trainer.py`; the loop
only branches on `isinstance(model, WindowedPINN)` — the shared net runs the 27
windows sequentially against the *same* parameter set.

## Results (identical configs, only `share_network` differs)

| | shared (`4x60_f64_unit_win27_shared_causal_warm`) | per-window (`4x60_f64_unit_win27_pernet_1500`) |
|---|---|---|
| `share_network` | True | False |
| model | `SharedWindowPINN` | `WindowedPINN` |
| `n_params` | **11,283** | **304,641** |
| `rmse_combined_l2` | **0.398124876** | **0.000178894235** |
| `mae_combined_l2` | 0.298132259 | 0.000106046666 |
| `max_abs_error_combined_l2` | **1.05726149** | **0.000531895219** |
| `rmse_x / y / z` | 0.036277 / 0.335079 / 0.211919 | 4.2654e-05 / 1.6222e-04 / 6.2199e-05 |
| `r2_x / y / z` | 0.714810 / 0.924993 / 0.920090 | 0.99999961 / 0.99999998 / 0.99999999 |
| `final_loss` | 2.28915748e-07 | 2.28915748e-07 |
| `residual_mse_collocation` | 0.639094705 | 1.94089614e-07 |
| `residual_mse_dense` | 0.639460407 | 1.94360425e-07 |
| `invariant_1_max_drift` | 1.03565693 | 0.000374095667 |
| `invariant_2_max_drift` | 1.37376463 | 0.000240221812 |
| `wall_clock_s` | 309.05 | 269.79 |
| `n_snapshots` | 32 | 32 |

**The shared network is ~2,225x worse in combined RMSE** (0.398 vs 1.789e-04).

## Why it fails: catastrophic forgetting

Per-window max abs error, read back from each run's saved model with the *same*
reference trajectory (`_cmp.py`, run before deletion):

```
pernet(304,641 params): overall_max=5.319e-04
    win0=1.72e-05  win1=8.54e-06  win2=6.40e-06 ... win24=5.29e-04 win25=4.62e-04 win26=5.10e-04
shared( 11,283 params): overall_max=1.056e+00
    win0=5.46e-01  win1=6.64e-01  win2=7.53e-01 ... win24=9.69e-02 win25=4.87e-02 win26=5.10e-04
```

* The **per-window** model reproduces *every* window to ~1e-5..5e-4.
* The **shared** model reproduces only the **last** window (`win26 = 5.10e-04`, same
  as pernet) and is wrong by O(1) in the earlier ones. Sequentially warming the one
  parameter set through 27 windows overwrites the fits of the windows already
  trained — classic catastrophic forgetting.

The identical `final_loss` in both runs is **not** a coincidence: with
`warm_start=True` the per-window run already initialises each window from the
previous window's weights, so the shared run follows the *same parameter
trajectory*. The difference is retention — the per-window model keeps 27 copies,
the shared model keeps only the final one. That is exactly why the loss matches
while the end-to-end trajectory does not.

## Decision

Per the requirement "if these changes does not break PINN ... keep the changes":

* These changes **do break** the PINN's end-to-end trajectory accuracy, therefore
  `share_network` stays **opt-in** and is **not** the default.
* Default `Config()` and `run_causal.py` (no `--shared`) are unchanged: the
  27-network `WindowedPINN` remains the working architecture.
* All 40 tests pass with the new code paths present.

## Reproduce

```bat
py -3 run_causal.py                                    :: per-window (27 nets), config changes
py -3 run_causal.py --shared                           :: ONE 11k-param net, 27 windows
py -3 run_causal.py --out 4x60_f64_unit_win27_pernet_1500
```

`--out NAME` writes to `runs/NAME` and is used so the new per-window run does not
clobber the archived `runs/4x60_f64_unit_win27_causal_warm` (a different,
39,793-collocation-point run of record).
