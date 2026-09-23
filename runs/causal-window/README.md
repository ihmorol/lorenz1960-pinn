# Causal-window experiments

`run_causal.py` writes each configuration to
`candidate-512pt/<architecture>_<settings-hash>/`. Compare new metrics with the
unchanged historical Run B at `../4x60_f64_unit_win27_causal_warm/`.
The completed candidate's final checkpoint, configuration, histories,
point summary, and figures are saved in that directory.
