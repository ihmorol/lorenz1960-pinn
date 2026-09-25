# Sequential-walk T=10 negative result

This is the differentiable trapezoid/cumulative sequential formulation. It is
not literature time marching and is retained as a negative comparison.

- Horizon: `t = [0, 10]`
- Model: 4x60 tanh PINN
- Collocation points: 3,000
- Adam steps: 20,000
- Combined RMSE: `2.05792761`
- Maximum state error: `3.27762208`
- Runtime: `171.38 s` on CPU

The `1.41` component-averaged MSE in `results.png` records failure to reproduce
the closed trajectory despite the decreasing physics loss.

The source branch contains the metrics, histories, and saved figures, but no
`history/pinn.pt`, `history/progress.pt`, parameter trail, or point snapshots.
This exact model cannot be reloaded or resumed from the committed artifacts.
