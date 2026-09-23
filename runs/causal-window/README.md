# Causal-window experiments

`run_causal.py` writes the 1500-point configuration to
`candidate-1500pt/<architecture>_<settings-hash>/`. Its final checkpoint,
configuration, compact histories, point summary, and figures are saved here.
The large generated breakdown snapshots and resumable progress state remain
local. Compare its metrics with the unchanged historical Run B at
`../4x60_f64_unit_win27_causal_warm/`. The 512-point results live on
`codex/causal-512pt`.
