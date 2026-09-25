# Windowed causal PINN, 1,536 points per window

This is the selected final implementation and result.

- Horizon: `t = [0, 13.26446]`, one closed Lorenz-1960 orbit
- Model: 27 independent 4x60 tanh PINNs with warm-started weights
- Sampling: 1,536 effective uniform training points per window
- Adam: epsilon schedule `(0.01, 0.1, 1, 10)`, at most 4,000 steps per stage
- L-BFGS: at most 1,000 iterations per window
- Precision: float64
- Combined RMSE: `6.96696531e-5`
- Maximum state error: `3.09323169e-4`
- Runtime: `881.08 s` on CUDA

Windows 10 and 11 reached the Adam cap at epsilon 10 with minimum causal
weights `0.9718467` and `0.9031337`; the run is complete but those two stages
did not meet the `0.99` threshold.

Use `history/pinn.pt` for final-model reload. The committed CSVs and figures
describe this run. The source branch did not contain `history/progress.pt`,
`history/param_trail.npz`, or the per-step `breakdown/` snapshots; intermediate
resume and full training-time figure regeneration therefore remain unavailable.
