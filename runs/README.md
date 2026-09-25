# Run catalog

Each completed run keeps `history/` (checkpoint and telemetry), `figures/`,
`metrics.csv`, `run_summary.csv`, and, when snapshots were available,
`point_summary.csv` and `breakdown/`. New saved runs keep exact settings in
`history/config.json`; windowed or checkpointed runs also keep an atomic resume
point in `history/progress.pt`.
`python run_index.py` rebuilds `index.html` from completed runs, including
nested directories.

## Saved historical runs (immutable)

| Directory | Meaning | Status |
|---|---|---|
| `4x60/` | Earlier short-horizon 4x60 run | Historical result |
| `4x60_f64_unit/` | Run A, full-domain float64 | Audited; point snapshot predates final L-BFGS state |
| `4x60_f64_unit_win27_causal_warm/` | Run B, 27 causal windows | Audited; saved training telemetry has the limitations in the code audit |

The root `comparison.csv`, `figures/`, comparison images, logs, films, and
`index.html` are historical outputs. In particular, `figures/precision_floor.png`
compares different time horizons and cannot establish a precision effect.
Existing paths remain in place so the linked reports and checkpoints remain
reproducible. The fixes in this branch do **not** change those saved numbers.

## New outputs

- `causal-window/candidate-1500pt/<architecture>_<settings-hash>/`: the saved
  1500-point causal-window run. The 512-point results are on `codex/causal-512pt`.
- `causal-window/candidate-1536pt/<architecture>_<settings-hash>/`: the completed
  causal-window run configured by `run_causal.py` (1536 points per window,
  up to 1000 L-BFGS iterations per window).
- `batch-precision/<architecture>/`: future batch baseline runs, if needed.

The settings hash prevents two causal configurations from sharing a directory;
`history/config.json` is checked before any checkpoint or finished summary is
reused. This branch includes the completed 1500-point run output; the code
repair itself did not retrain the historical run. The completed 1536-point
candidate and its final checkpoint are also included.
