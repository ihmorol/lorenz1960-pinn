# Ablation results (audited 2026-10-08)

Canonical source: `runs/paper_ablations/` (15 runs, merged from the Colab download) plus
the two runs of record: `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd`
(r4 seed 0) and `runs/4x60_f64_unit` (F1, single network). Problem fixed throughout:
k=2, l=1, IC (0.5, 0.75, 1.0), t in [0, 13.26446], float64, 13,265 evaluation times.
Regenerate the audit and every number below with `python scripts/ablation_report.py`.

## 1. Run audit

Every meaningful run now lives in exactly one place, `runs/paper_ablations/`: 15 ablation runs,
plus all logs (9 Colab logs and `logs/local_cpu_r1_r4_seeds1-3.log` for the 5 CPU runs), the
physics checks, and `PAPER_TABLES.md`. The two runs of record were already tracked at the paths
above. Model weights (`history/pinn.pt`) are on local disk for all 15 runs; git ignores them.

Duplicates and the incomplete run were moved, not deleted, to
`E:\University\FYDP\lorenz1960-pinn-archive-2026-10-08\`. Before the move, every file
in the Colab folder was checked to exist in the canonical folder.

| moved item | what it was | why it is not needed |
|---|---|---|
| `paper_ablations-colab/` | Colab suite as downloaded from Drive (all 15 runs) | merged into `runs/paper_ablations/`; its r1 and r4 s1-2 are the same runs as the tracked CPU copies, with identical metrics |
| `paper_ablations-20261008T003527Z-1-001.zip` | Drive zip of the folder above (653 files) | duplicate |
| `paper_ablations_results.zip` | zip of an early local snapshot | duplicate |
| `paper_ablations_incomplete/r4_candidate_seed3_cpu_partial` | abandoned CPU attempt at r4 s3 (stopped after window 25 of 27) | **incomplete**; the Colab rerun replaces it |
| `runs/paper_ablations_results/` (not moved yet: Windows reports it is in use) | the same early snapshot, unzipped | duplicate, untracked; safe to delete by hand |

Every run counted as complete has all of the following: 27/27 windows (1/1 for r3) in
`loss_history.csv`, L-BFGS run on every window, no gaps in the iteration index, no NaN,
`run_summary.csv` and `reference_error.csv` written, and a config that differs from the
run of record only in the intended knob(s).

Incomplete or excluded:

- **Local `r4_candidate_seed3`**: stopped after window 25 of 27 (26 window checkpoints,
  no loss history, no summary). It now sits in `runs/paper_ablations_incomplete/`.
  The Colab `r4_candidate_seed3`, rerun from scratch on the T4, is complete and has taken
  its place.
- Logs for r2 s1, r2 s2, r3 s0 and r6 s1 are truncated or end in `KeyboardInterrupt`.
  These runs were resumed from checkpoints. Their histories are complete and gapless,
  so the metrics are valid. Their wall clock is not.
- Wall clock is not comparable across runs. r1 s1-3 and r4 s1-2 ran on a local CPU.
  The rest ran on a T4 shared by 2-4 parallel jobs, and some of those runs were resumed.
  Use Adam steps as the cost measure.
- Local `.pt` checkpoints and `breakdown/` CSVs exist only in the local folder, so keep it.

## 2. Main ablation table (mean ± sample std over seeds)

RMSE and max error are combined L2 norms against DOP853. "Residual" is the dense-grid
ODE residual MSE. E drift is max|E(t) − E(0)| for E = k⁴x² + l⁴y². The reference drift is 1.8e-10.

| arm | change vs candidate | n | Adam steps | capped stages | RMSE | max error | residual MSE | E drift | RMSE / r4 |
|---|---|---|---|---|---|---|---|---|---|
| **r4 candidate** | none (seeds 0-4) | 5 | 34,768 | 1.8 / 108 | 1.37e-4 ± 0.87e-4 | 3.68e-4 ± 1.8e-4 | 1.81e-7 ± 0.46e-7 | 4.03e-4 ± 0.47e-4 | 1 |
| r1 no causal | causal off, 1,300 Adam/window (budget-matched) | 3 | 35,100 | -- | 2.93e-4 ± 1.9e-4 | 1.04e-3 ± 0.69e-3 | 9.49e-7 ± 1.3e-7 | 6.46e-4 ± 1.9e-4 | 2.1× |
| r2 no warm start | each window from fresh init | 3 | 102,989 | 18 / 108 | 2.50e-4 ± 0.64e-4 | 9.72e-4 ± 2.1e-4 | 8.54e-7 ± 1.4e-7 | 7.05e-4 ± 0.50e-4 | 1.8× |
| r6 sequential | one network shared by all 27 windows (11,283 params) | 3 | 35,100 | -- | 0.495 ± 0.149 | 1.25 ± 0.36 | 1.03 ± 0.55 | 0.864 ± 0.49 | 3,600× |
| r3 causal single | F1 + causal schedule, no windows | 1 | 40,000 | 4 / 4 | 0.275 | 0.677 | 5.23e-6 | 9.68e-2 | 2,000× |
| F1 single | one network, no causal, no windows | 1 | 40,000 | -- | 2.237 | 3.346 | 2.80e-4† | -- | 16,300× |

† F1 has only the collocation-grid residual on record.

Significance vs r4 (one-sided exact Mann-Whitney; with 3 vs 5 runs the smallest
attainable p is 0.018):

| comparison | RMSE | max error | residual MSE | E drift | paired-seed RMSE ratios (s1, s2, s3) |
|---|---|---|---|---|---|
| r1 vs r4 | p = 0.20 (overlap) | p = 0.036 | **p = 0.018, fully separated** | **p = 0.018, fully separated** | 2.16, 0.73, 2.60 |
| r2 vs r4 | p = 0.071 | **p = 0.018, fully separated** | **p = 0.018, fully separated** | **p = 0.018, fully separated** | 2.17, 1.11, 1.44 |
| r6 vs r1 | **p = 0.018, fully separated** | **p = 0.018** | **p = 0.018** | **p = 0.018** | 2,771, 3,432, 679 |

## 2b. Headline statistic: median and range (agreed reporting choice)

The paper is framed around a 10⁻⁵-order result. The **median** is the honest way to
state it: it is robust to the two high seeds, and the same statistic is used for every arm.

| arm | n | median RMSE | range | seeds below 10⁻⁴ | mean (for reference) |
|---|---|---|---|---|---|
| r4 candidate | 5 | **8.13e-5** | 6.97e-5 – 2.60e-4 | **3 of 5** | 1.37e-4 |
| r1 no causal | 3 | 1.89e-4 | 1.75e-4 – 5.16e-4 | 0 of 3 | 2.93e-4 |
| r2 no warm start | 3 | 2.85e-4 | 1.76e-4 – 2.88e-4 | 0 of 3 | 2.50e-4 |
| r6 sequential | 3 | 0.486 | 0.350 – 0.648 | 0 of 3 | 0.495 |
| r3 causal single | 1 | 0.275 | -- | 0 of 1 | -- |
| F1 single | 1 | 2.237 | -- | 0 of 1 | -- |

Suggested results sentence:

> The candidate configuration reaches a median RMSE of 8.1×10⁻⁵ over five seeds (range
> 7.0×10⁻⁵ to 2.6×10⁻⁴), with three of five seeds below 10⁻⁴. No seed without causal
> weighting or without warm start falls below 10⁻⁴ (medians 1.9×10⁻⁴ and 2.9×10⁻⁴).

Guardrails (all runs are in the released artifacts, so a reviewer can check every one):

- Always print the range (or all seeds) next to the median. Never report only the 3 good seeds.
  Seeds 2 and 3 completed normally (1-2 capped stages, like the others), so there is no
  basis for excluding them.
- The mean (1.37e-4) and the geometric mean (1.17e-4) are both in the 10⁻⁴ decade.
  Do not call either of them 10⁻⁵.
- The median depends on seed 0. Seeds 1-4 alone have a median of 1.40e-4. Including
  seed 0 is correct because its config is identical.
- **Do not claim "one order of magnitude better than no causal weighting."** The median
  ratio is 2.3× (r1) and 3.5× (r2). The order-of-magnitude statements that hold are:
  windowed versus single network, about 4 orders (2.24 → ~1e-4), and separate versus
  shared networks, about 3 orders.

## 3. Per-run table (appendix)

| run | seed | device | Adam | L-BFGS evals | capped | RMSE | max error | residual MSE | E drift | closure ‖u(T)−u(0)‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| r4 candidate (record) | 0 | cuda | 34,765 | 33,143 | 2 | 6.97e-5 | 3.09e-4 | 1.87e-7 | 4.05e-4 | 5.64e-5 |
| r4 candidate | 1 | cpu | 33,564 | 33,163 | 2 | 8.13e-5 | 2.30e-4 | 1.32e-7 | 3.38e-4 | 9.82e-5 |
| r4 candidate | 2 | cpu | 36,759 | 33,082 | 1 | 2.60e-4 | 5.02e-4 | 2.49e-7 | 4.51e-4 | 4.13e-4 |
| r4 candidate | 3 | cuda | 35,996 | 33,106 | 2 | 1.99e-4 | 6.12e-4 | 1.94e-7 | 4.42e-4 | 5.41e-4 |
| r4 candidate | 4 | cuda | 32,757 | 32,516 | 2 | 7.50e-5 | 1.86e-4 | 1.44e-7 | 3.76e-4 | 1.08e-4 |
| r1 no causal | 1 | cpu | 35,100 | 33,211 | -- | 1.75e-4 | 7.37e-4 | 8.81e-7 | 4.78e-4 | 3.50e-4 |
| r1 no causal | 2 | cpu | 35,100 | 33,486 | -- | 1.89e-4 | 5.61e-4 | 1.10e-6 | 6.04e-4 | 1.18e-4 |
| r1 no causal | 3 | cpu | 35,100 | 33,385 | -- | 5.16e-4 | 1.83e-3 | 8.68e-7 | 8.56e-4 | 1.34e-3 |
| r2 no warm | 1 | cuda | 105,877 | 33,174 | 18 | 1.76e-4 | 7.52e-4 | 8.35e-7 | 7.07e-4 | 3.86e-4 |
| r2 no warm | 2 | cuda | 100,185 | 33,431 | 17 | 2.88e-4 | 1.17e-3 | 1.00e-6 | 7.53e-4 | 7.98e-4 |
| r2 no warm | 3 | cuda | 102,904 | 33,387 | 19 | 2.85e-4 | 9.95e-4 | 7.29e-7 | 6.54e-4 | 7.21e-4 |
| r6 sequential | 1 | cuda | 35,100 | 33,759 | -- | 0.486 | 1.233 | 0.965 | 0.620 | 2.40e-3 |
| r6 sequential | 2 | cuda | 35,100 | 33,760 | -- | 0.648 | 1.611 | 1.609 | 1.426 | 3.92e-4 |
| r6 sequential | 3 | cuda | 35,100 | 33,403 | -- | 0.350 | 0.892 | 0.519 | 0.547 | 1.40e-3 |
| r3 causal single | 0 | cuda | 40,000 | 5,529 | 4 of 4 | 0.275 | 0.677 | 5.23e-6 | 9.68e-2 | 0.665 |
| F1 single (record) | 0 | cuda | 40,000 | 6,251 | -- | 2.237 | 3.346 | 2.80e-4† | -- | -- |

## 4. LaTeX (booktabs, IEEEtran column width)

```latex
\begin{table}[t]
\caption{Ablations on the full orbit (mean $\pm$ std over seeds; errors vs DOP853)}
\label{tab:ablations}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Arm & $n$ & Adam steps & RMSE & Residual MSE\\
\midrule
Candidate (causal, warm, 27 nets) & 5 & 34,768 & $(1.37\pm0.87)\times10^{-4}$ & $(1.81\pm0.46)\times10^{-7}$\\
No causal weights & 3 & 35,100 & $(2.93\pm1.9)\times10^{-4}$ & $(9.49\pm1.3)\times10^{-7}$\\
No warm start & 3 & 102,989 & $(2.50\pm0.64)\times10^{-4}$ & $(8.54\pm1.4)\times10^{-7}$\\
One shared net, 27 windows & 3 & 35,100 & $0.495\pm0.149$ & $1.03\pm0.55$\\
Single net + causal & 1 & 40,000 & $0.275$ & $5.23\times10^{-6}$\\
Single net (F1) & 1 & 40,000 & $2.237$ & $2.80\times10^{-4}$\,\textsuperscript{\dag}\\
\bottomrule
\multicolumn{5}{l}{\scriptsize \textsuperscript{\dag}collocation-grid residual; others on the 13,265-point dense grid.}
\end{tabular}
\end{table}
```

## 5. What the results support in the paper

1. **The 6.97e-5 headline is the best of 5 seeds.** Report 1.37e-4 ± 0.87e-4
   (range 6.97e-5 to 2.60e-4, n=5). Seed 0 can stay as the run shown in the figures,
   but the text should say so.
2. **Causal weighting (r1) helps but does not match the 10-100× in Wang et al.** At a
   matched Adam budget it gives about 2× lower RMSE on average, and r1 is worse in 2 of
   3 paired seeds. The RMSE difference is not significant (p = 0.20). The 5.2× lower
   residual MSE and 1.6× lower E drift are consistent: every r4 seed beats every r1 seed.
   Write it as "consistently lower residual, about 2× lower state error". Do not quote
   the citation's 1-2 orders of magnitude as if this run reproduced it.
3. **Warm start (r2) improves both accuracy and cost.** Without it, runs take 3× more Adam
   steps (103k vs 35k), hit the 4,000-step cap in 17-19 of 108 stages instead of 1-2,
   and still end with 1.8× higher RMSE (worse in 3 of 3 paired seeds) and 2.6× higher
   max error (fully separated).
4. **Separate networks per window are essential (r6).** A single network trained window
   by window under the same budget forgets earlier windows: RMSE 0.35-0.65, which is
   680-3,400× worse than r1 in every seed. The residual MSE of about 1 confirms that the
   failure is forgetting, not an optimization stall.
5. **Causal weighting alone does not rescue a single network (r3).** It improves F1 by
   8× (2.24 → 0.275), but every causal stage hit the 10,000-step cap, and the run is
   still about 2,000× worse than the candidate. Windowing does most of the work. Both
   r3 and F1 are single-seed runs.
6. **Open confound from the paper's own limitations section:** no single network with
   about 304k parameters (matching the windowed model) was run. r6 is not that control,
   because its failure comes from forgetting, not from too little capacity.

## 6. Do we need more runs?

The current set is enough for the paper if it is reported as above. All 15 Colab runs
plus the 2 records are complete, and nothing has to be rerun. Optional additions, by value:

| priority | runs | why | cost |
|---|---|---|---|
| recommended | r1 seeds 0 and 4 | gives the causal claim (the weakest one) 5 paired seeds; exact MWU can then reach p = 0.004 | ~20 min each, CPU |
| useful | capacity-matched single network (~300k params, F1 settings) | closes the "27× parameters" objection the paper raises itself | ~1.5-2 h, T4 |
| optional | r3 and F1 seeds 1-2 | turns the single-network 8× causal effect into mean ± std | ~2 h (r3) + ~1.3 h (F1) per seed, T4 |
| optional | r2 seeds 0 and 4 | n=5 for warm start; the claim is already well supported | ~1.5-3 h each, T4 |

Commands (resume-safe; finished arms are skipped):

    python scripts/paper_ablations.py --arms r1 --seeds 0,4     # recommended, CPU ok
    python scripts/paper_ablations.py --arms r2 --seeds 0,4     # optional, GPU
    python scripts/ablation_report.py                           # re-audit + new stats

`r1_no_causal_seed0` pairs with the seed-0 run of record in `runs/causal-window/`, not
with a folder under `runs/paper_ablations/`. The capacity-matched single network needs a
new config (for example width ≈ 300, depth 4, with F1's grid and budget). No script exists for it yet.

## 7. Open tasks for the paper (`writings/conference-paper/main.tex`)

1. Results: add the ablation table (§4 LaTeX) and the median/range sentence (§2b).
   Keep Table `tab:errors` for seed 0, but label it "seed 0, the lowest-RMSE of five seeds, shown in the figures",
   or replace the headline with median and range.
2. Discussion and limitations: replace "a budget-matched ablation ... accompanies the released
   run artifacts" and "seed sensitivity is quantified in the released run artifacts"
   with the actual numbers (causal weighting is about 2× in RMSE and 5× in residual; seed spread 7.0e-5 to 2.6e-4).
3. Do not let the Wang et al. "one to two orders of magnitude" citation read as reproduced.
4. Conclusion: change "Future work isolates each training option one at a time under a
   multi-seed protocol" to "this work"; the ablations now do exactly that.
5. Keep the capacity confound (27× parameters) as a stated limitation unless the
   capacity-matched run is added.
