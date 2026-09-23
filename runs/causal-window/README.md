# Causal-window experiments

`run_causal.py` now writes the 1536-point configuration to
`candidate-1536pt/<architecture>_<settings-hash>/`. The completed 1500-point
candidate remains in `candidate-1500pt/`. Final checkpoints, configurations,
compact histories, point summaries, and figures are saved here after a run.
The large generated breakdown snapshots and resumable progress state remain
local. Compare its metrics with the unchanged historical Run B at
`../4x60_f64_unit_win27_causal_warm/`. The 512-point results live on
`codex/causal-512pt`.
