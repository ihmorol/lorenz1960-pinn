# Historical full-orbit windowed causal run

Earlier 27-window causal result retained separately from the selected 1,536-point run.

- Recorded combined L2 RMSE: `8.43460385e-05`
- Measurement source: `run_summary.csv`
- Key artifacts: `run_summary.csv`, `metrics.csv`, `results.png`, `figures/`, `film/final.mp4`, `history/pinn.pt`, `history/param_trail.npz`, `breakdown/`

A final checkpoint is available; no intermediate `progress.pt` is committed.
The saved gradient-stability and histogram figures are historical diagnostics:
the original training loop left gradients from completed windows in its trail.
The film's yellow arrows were not computed from gradients. Neither should be
used as evidence about the optimizer; see the September 22 code audit.
