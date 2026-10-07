# Run the paper ablation suite on Colab

Runtime: GPU (T4). Two cells. The second is safe to rerun after a disconnect:
finished runs are skipped and interrupted ones resume from their last checkpoint.

```python
# Cell 1: mount Drive (the auth dialog only works from a notebook cell)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2: pinned checkout, install, run
%cd /content
!test -d /content/lorenz1960-pinn || git clone https://github.com/ihmorol/lorenz1960-pinn.git /content/lorenz1960-pinn
%cd /content/lorenz1960-pinn
!git fetch -q origin && git checkout -q ed43703
!pip install -q pandas scipy matplotlib seaborn
!python /content/lorenz1960-pinn/scripts/colab_suite.py --jobs 4
```

`ed43703` is the reviewed revision; every suite invocation appends the
commit, dirty flag, package versions and GPU to `runs/paper_ablations/environment.jsonl`.

When it finishes, download `/content/lorenz1960-pinn/paper_ablations_results.zip`
(also copied to `MyDrive/lorenz1960-pinn-ablation/paper_ablations/`), unzip it
over `runs/` locally. The tables are already in `runs/paper_ablations/PAPER_TABLES.md`;
`python scripts/paper_tables.py` regenerates them and `--latex` prints booktabs tables.

## Parallelism and time

Each run is its own process on the one GPU, `--jobs` at a time (capped by
free RAM at ~2.5 GB per job; CPU threads are split between jobs). Longest
first: R3 starts immediately because it is the critical path. Recorded GPU
costs, each alone on the GPU: F1, the same network on the same 39,793 points,
took 75 min of Adam (113 ms/step, float64); the candidate took 15 min. So R3
seed 0 needs roughly 80-100 min, and the 8 remaining windowed runs share the
other three slots beside it. A 1.5-2 h total is the target, not a measurement:
the GPU is time-sliced between processes and a free Colab VM has 2 vCPUs. Each
run's console output is in `runs/paper_ablations/logs/<run>.log`. If Colab cuts
the session, rerun cell 2; at most the last sync interval (5 min) plus one
checkpoint interval is redone.

Wall clock in the tables is per run under contention, so it is not comparable
across devices or job counts. The committed R1 seeds 1-3 and R4 seeds 1-2 are
CPU runs from a local machine and are reused. To rerun everything on Colab, give
a new output root, e.g. `--out runs/paper_ablations_colab` (Drive folder follows
the root name).

A second R3 seed in a later session:
`!python /content/lorenz1960-pinn/scripts/colab_suite.py --arms r3 --seeds 1`

## Durability

- Drive must be mounted and writable on Colab, or the driver stops; `--no-drive`
  runs without persistence on purpose.
- Every finished run is synced immediately, the rest every 5 minutes. Files
  are copied to `.partial` and renamed, so an interrupted copy never replaces a
  good one, and the previous `progress.pt` is kept as `progress.prev.pt`.
- On start, Drive is restored over the clone (except runs finished only in the
  clone), and each restored `progress.pt` is loaded to validate it, falling
  back to the previous one.
- Windowed runs checkpoint at every window; R3 every 2,000 Adam steps with its
  causal stage, step, Adam and LR-scheduler state. Checkpoints resume across
  machines and checkout paths.
- Regenerable bulk (`window_*.pt`, `param_trail.npz`) never goes to Drive;
  finished runs also drop `progress.pt` and per-point snapshots there. Peak
  Drive use is about 1 GB per running job.

## What runs (k=2, l=1, IC (0.5, 0.75, 1.0), t in [0, 13.26446])

Seeds 1-3 are paired across the windowed arms; R4 adds seed 4; R3 seed 0 pairs
with F1 (seed 0). Every row of `ablations.csv` records actual Adam steps,
L-BFGS evaluations, parameter count, points and device.

| Arm | Compared with | Changed on purpose | Uncontrolled differences |
|---|---|---|---|
| `r4_candidate` | its own seeds | seed | causal stages stop adaptively, so Adam steps vary per seed (seed 0: 34,765) |
| `r1_no_causal` | r4 | causal weighting off | Adam fixed at 1,300/window = 35,100, matched to seed 0 of r4 only |
| `r2_no_warm` | r4 | warm start off | adaptive stage stopping, as r4 |
| `r6_sequential` | r1 | one network shared by all windows | 11,283 parameters vs 304,641 |
| `r3_causal_single` | F1 | causal schedule (4 stages x 10,000 Adam cap = F1's 40,000) | stages may stop early; per-point causal weights accumulate over 39,793 points instead of ~1,535, so the same epsilon is effectively stronger |

R3 vs the candidate differs in windowing, points (39,793 LHS vs 41,446 uniform),
L-BFGS (5,000 total vs 1,000 per window) and LR step (5,000 vs 1,000); only
the R3-vs-F1 comparison isolates one change. R5 (parameter-matched 4x320) was
removed by review decision.

R7 physics checks (`scripts/paper_physics_checks.py`, run by the driver at the
end) cover the candidate and every finished run: drift of E = k^4 x^2 + l^4 y^2
and orbit closure ||u(T) - u(0)||.

## Results so far (committed, CPU)

| run | RMSE (combined L2) |
|---|---|
| candidate seed 0 (GPU, run of record) | 6.97e-5 |
| r4_candidate seed 1 / 2 | 8.13e-5 / 2.60e-4 |
| r1_no_causal seed 1 / 2 / 3 | 1.75e-4 / 1.89e-4 / 5.16e-4 (mean 2.93e-4) |

Paired: causal off is worse for seed 1 (2.16x) but better for seed 2 (0.73x),
so a causal-weighting benefit is not established until seeds 3 (and more) land.
