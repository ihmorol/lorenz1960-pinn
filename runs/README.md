# Run catalog

`active/` contains the three experiments used by the project. Each directory
keeps its own available checkpoint, histories, metrics, summaries, and figures.

| Directory | Role | Combined RMSE |
|---|---|---:|
| `active/windowed-causal-1536pt/` | Selected final windowed causal PINN | `6.96697e-5` |
| `active/shared-network-causal-negative/` | Shared-network negative ablation | `3.98125e-1` |
| `active/sequential-walk-t10-negative/` | Differentiable sequential-walk failure | `2.05793` |

`archive/` contains unique historical experiments, grouped by family.

`history/pinn.pt` reloads a final model when present. `history/progress.pt`
restores intermediate training, and `history/param_trail.npz` plus `breakdown/`
support training-time visualization when saved. The source branches did not
contain those three artifacts for any of the active runs. The sequential walk
also lacks a final checkpoint. The saved figures remain available, but those
missing states cannot be recovered from them.
