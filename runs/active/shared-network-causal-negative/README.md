# Shared-network causal negative ablation

This experiment tests one shared 4x60 network across 27 causal windows. It is a
negative result retained because it shows that sharing one network does not
replace the selected per-window formulation under this setup.

- Horizon: `t = [0, 13.26446]`
- Sampling: 1,500 uniform points total, about 55-56 per window
- Precision: float64
- Combined RMSE: `0.398124876`
- Maximum state error: `1.05726149`
- Runtime: `309.05 s` on CPU

The saw-tooth trajectory and `5.28e-2` component-averaged MSE in `results.png`
are the expected recorded failure, not a plotting defect.

`history/pinn.pt` reloads the final model. The legacy shared-network architecture
is kept locally in `evaluate.py`, so this negative ablation remains reproducible
without adding the discarded architecture to the selected implementation.

From the repository root, reproduce the recorded accuracy with:

```bash
python runs/active/shared-network-causal-negative/evaluate.py
```

The command rebuilds the original 13,265-point DOP853 reference trajectory,
evaluates the actual checkpoint, and checks the computed combined RMSE against
`run_summary.csv` with a default absolute tolerance of `1e-9`. It requires the
project's existing `numpy`, `scipy`, and `torch` dependencies.

The source branch did not contain `history/progress.pt`,
`history/param_trail.npz`, or `breakdown/` snapshots.
