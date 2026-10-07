# Run the paper ablation suite on Colab

One notebook cell, using only this branch. Runtime ~2.5–3.5 h on a T4 GPU;
if the runtime disconnects, rerun the same cell — finished arms are skipped.

```python
!git clone --branch feat/colab-ablation-suite https://github.com/ihmorol/lorenz1960-pinn.git
%cd lorenz1960-pinn
!pip install -q pandas scipy matplotlib seaborn
!python scripts/colab_suite.py
```

Then download `paper_ablations_results.zip` from the file panel (folder icon on
the left) and unzip it over `runs/paper_ablations/` locally.

## What it runs (fixed problem: k=2, l=1, IC (0.5, 0.75, 1.0), t ∈ [0, 13.26446])

| Arm | Isolates | Config |
|---|---|---|
| `r1_no_causal_seed{1,2,3}` | causal weighting | windowed-27, causal off, 1,300 Adam/window (budget-matched to the candidate's recorded 34,765 updates), L-BFGS 1,000 |
| `r4_candidate_seed{1,2,3,4}` | seed spread of the headline | the candidate config, unchanged |
| `r2_no_warm_seed{1,2,3}` | warm start | windowed-27 + causal, `warm_start=False` |
| `r3_causal_single_seed{1,2}` | windowing | single network + the candidate's causal schedule, 40,000 Adam + 5,000 L-BFGS |
| `sequential_single_net` | the handoff-forgetting claim | one network trained over 27 segments, 1,300 Adam/window |

Optimizer budgets are the candidate's, unchanged — no step counts increased.
R5 (parameter-matched 4×320) was deliberately removed.

The R6 physics checks (`scripts/paper_physics_checks.py`) need the candidate
checkpoint and run locally in seconds; they are not part of the Colab suite.

## After the zip lands locally

```bash
python scripts/paper_tables.py            # in the feat/paper-results-tables worktree
```

regenerates `docs/PAPER_TABLES.md` (failed-runs table, ablation table with
mean ± std, seed table) with the PENDING rows filled in.
