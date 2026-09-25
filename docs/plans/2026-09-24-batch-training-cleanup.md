# Batch Training Cleanup Plan

## Goal

Make the selected 1,536-point windowed causal PINN the documented final
implementation while preserving every scientifically distinct completed run in
one self-contained folder.

## Behaviour to preserve

- Windowed causal training, warm-start transfer, checkpoint reload, evaluation,
  and visualization remain numerically unchanged.
- The committed 1,536-point checkpoint continues to reproduce its recorded
  trajectory and error metrics.
- Each retained run owns its configuration, histories, checkpoint, summaries,
  and figures; no run depends on files stored in another run folder.

## Changes

1. Make saved run manifests portable across checkout locations.
2. Update the Colab entry point and design documentation for the selected
   1,536-point/1,000-L-BFGS protocol and its capped causal stages.
3. Remove local runtime state and generated cache debris.
4. Organize active runs under `runs/active/` and unique historical experiments
   under `runs/archive/`; copy missing distinct runs from their source branches.
5. Keep only the selected 1,536-point result, the shared-network negative
   ablation, and the T=10 sequential-walk failure under `runs/active/`.
6. Add per-run README metadata so every active run can be understood independently.

## Verification

- Regression test: a completed run copied to another checkout path is reusable.
- Full Python test suite.
- Reload and re-evaluate the final 1,536-point checkpoint.
- Validate every retained run folder has its own README and artifact inventory.
- Confirm the original checkout's uncommitted files are unchanged.
