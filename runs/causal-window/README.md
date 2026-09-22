# Causal-window experiments

`run_causal.py` writes each configuration to
`candidate-512pt/<architecture>_<settings-hash>/`. Compare new metrics with the
unchanged historical Run B at `../4x60_f64_unit_win27_causal_warm/`.
The candidate directory is empty until training is explicitly run.
