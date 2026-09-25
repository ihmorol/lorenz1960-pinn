# Full-orbit batch float64 negative result

One batch PINN trained over the full orbit. The low residual did not give an accurate trajectory.

- Recorded combined L2 RMSE: `2.23706051`
- Measurement source: `run_summary.csv`
- Key artifacts: `run_summary.csv`, `metrics.csv`, `results.png`, `figures/`, `film/final.mp4`, `history/pinn.pt`, `history/param_trail.npz`, `breakdown/`

A final checkpoint is available; no intermediate `progress.pt` is committed.
The film's yellow arrows show projected path direction, not a computed
negative gradient; the rendered film is retained as a historical visualization.
