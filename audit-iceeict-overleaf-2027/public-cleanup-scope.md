# Public repository cleanup scope

This repository publishes the minimum material needed to inspect and reproduce
the Lorenz-1960 PINN experiments reported by the project.

Included:

- numerical baseline and PINN training/evaluation source;
- paper ablation, physics-check, reporting, and sequential-run scripts;
- the 14 archived ablation runs, the single-network baseline, and the selected
  windowed causal run;
- final checkpoints, numerical configurations, training loss histories,
  summaries, metrics, and aggregate comparison tables;
- verification scripts and integrity manifests.

Excluded:

- manuscripts, author information, internal planning files, local worktree
  history, and private notes;
- dense per-point diagnostic tables, training films, interactive HTML output,
  logs, intermediate checkpoints, and duplicate plots;
- unrelated exploratory runs and studies.

Source changes are limited to public-facing wording in comments and docstrings.
Executable statements and numerical constants must remain equivalent to source
commit `5a14232504848fc90400c99ccf7741f39ee2c71a` (audited training code from
`babc47a`). Archived configurations may have repository-local output paths in
place of machine-specific paths; every transformation is recorded with the
original and published file hashes.
