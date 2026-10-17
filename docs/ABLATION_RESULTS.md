# Ablation results: Colab suite + run of record (audited 2026-10-08)

**Scope:** the statistics pool the **14 runs of the Colab suite** in `runs/paper_ablations/`
(a byte-identical copy of the Drive download, hash-checked) **plus r4 seed 0**, the paper's
run of record (`runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd`).
That gives 15 runs. Seed 0 has exactly the candidate config, differing only in seed, and was
trained in an earlier session on a GPU. It is the only other run with that exact config. The
1500-point candidate run uses a different grid and L-BFGS budget, so it is excluded. F1, the single
network in `runs/4x60_f64_unit`, is shown as reference only and never pooled.

Problem fixed throughout: k=2, l=1, IC (0.5, 0.75, 1.0), t in [0, 13.26446], float64,
13,265 evaluation times. Regenerate everything with `python scripts/ablation_report.py`.

## 1. Run audit

| arm | seeds in the Colab suite | device | status |
|---|---|---|---|
| r1 no causal | 1, 2, 3 | cpu* | complete |
| r2 no warm start | 1, 2, 3 | T4 | complete (s1, s2 resumed from checkpoint) |
| r3 causal single | 0 | T4 | complete (resumed from checkpoint) |
| r4 candidate | 1, 2, 3, 4 (+ seed 0, run of record) | s1-2 cpu*, s3-4 T4, s0 GPU (earlier session) | complete |
| r6 sequential | 1, 2, 3 | T4 | complete |

\* These 5 runs were trained on the local CPU and restored into the Colab workspace from
Drive. The Colab download records them with `device=cpu`, and they were not re-trained on Colab.

A run counts as complete when all 27 windows (1 for r3) have both Adam and L-BFGS entries,
the iteration index has no gaps or NaN, and `run_summary.csv` and `reference_error.csv` exist.
Every config was diffed against the run of record and differs only in the intended knob.
r3 differs from F1 only in the causal settings.

- Logs for r2 s1, r2 s2, r3 and r6 s1 are truncated or end in `KeyboardInterrupt`; those runs
  resumed. Their histories are gapless, so the metrics are valid. Their **wall clock is not**.
- Wall clock is not comparable across runs: some ran on the local CPU, others on a T4 shared
  by 2-4 parallel jobs. Use Adam steps as the cost measure.
- Model weights (`history/pinn.pt`) are on local disk; git ignores them.

Moved out of the repo (not deleted) to `E:\University\FYDP\lorenz1960-pinn-archive-2026-10-08\`:
the Drive download folder and zip, the early snapshot zip, the previous local copies of r1 s1-3
and r4 s1-2 with their CPU log, and the incomplete local CPU attempt at r4 s3 (stopped at
window 25 of 27). `runs/paper_ablations_results/` is an untracked duplicate that Windows
reported as in use; delete it by hand.

## 2. Main table (mean ± sample std)

RMSE and max error are combined L2 norms against DOP853. "Residual" is the dense-grid ODE
residual MSE. E drift is max|E(t) − E(0)| for E = k⁴x² + l⁴y². The reference drift is 1.8e-10.

| arm | n | Adam steps | capped stages | median RMSE | RMSE range | mean RMSE | max error | residual MSE | E drift | seeds < 1e-4 |
|---|---|---|---|---|---|---|---|---|---|---|
| **r4 candidate** (seeds 0-4) | 5 | 34,768 | 1.8 / 108 | **8.13e-5** | 6.97e-5 – 2.60e-4 | 1.37e-4 ± 0.87e-4 | 3.68e-4 ± 1.8e-4 | 1.81e-7 ± 0.46e-7 | 4.03e-4 ± 0.47e-4 | **3 / 5** |
| r1 no causal | 3 | 35,100 | -- | 1.89e-4 | 1.75e-4 – 5.16e-4 | 2.93e-4 ± 1.9e-4 | 1.04e-3 ± 0.69e-3 | 9.49e-7 ± 1.3e-7 | 6.46e-4 ± 1.9e-4 | 0 / 3 |
| r2 no warm start | 3 | 102,989 | 18 / 108 | 2.85e-4 | 1.76e-4 – 2.88e-4 | 2.50e-4 ± 0.64e-4 | 9.72e-4 ± 2.1e-4 | 8.54e-7 ± 1.4e-7 | 7.05e-4 ± 0.50e-4 | 0 / 3 |
| r6 sequential | 3 | 35,100 | -- | 0.486 | 0.350 – 0.648 | 0.495 ± 0.149 | 1.25 ± 0.36 | 1.03 ± 0.55 | 0.864 ± 0.49 | 0 / 3 |
| r3 causal single | 1 | 40,000 | 4 / 4 | 0.275 | -- | -- | 0.677 | 5.23e-6 | 9.68e-2 | 0 / 1 |
| *F1 single (reference)* | 1 | 40,000 | -- | *2.237* | -- | -- | *3.346* | *2.80e-4 (collocation grid)* | -- | 0 / 1 |

Median RMSE relative to the candidate: r1 2.3×, r2 3.5×, r6 6,000×, r3 3,400×, F1 27,500×.

Significance vs r4 (one-sided exact Mann-Whitney; with 3 vs 5 runs the smallest attainable
p is 0.018, reached only when the arms are fully separated):

| comparison | RMSE | max error | residual MSE | E drift | paired RMSE ratio (s1, s2, s3) |
|---|---|---|---|---|---|
| r1 vs r4 | 0.20 (overlap) | 0.036 | **0.018, fully separated** | **0.018, fully separated** | 2.16, 0.73, 2.60 |
| r2 vs r4 | 0.071 | **0.018** | **0.018** | **0.018** | 2.17, 1.11, 1.44 |
| r6 vs r4 | **0.018** | **0.018** | **0.018** | **0.018** | r6/r1: 2,771, 3,432, 679 |

Suggested results sentence:

> Over five seeds the candidate configuration reaches a median RMSE of 8.1×10⁻⁵ (range
> 7.0×10⁻⁵ to 2.6×10⁻⁴), with three of five seeds below 10⁻⁴. No run without causal weighting
> or without warm start falls below 10⁻⁴ (medians 1.9×10⁻⁴ and 2.9×10⁻⁴).

Reporting guardrails:
- Always give the range (or all five seeds) next to the median. Seeds 2 and 3 completed
  normally (1-2 capped stages, like the others), so there is no basis for dropping them.
- The mean (1.37e-4) and geometric mean (1.17e-4) are in the 10⁻⁴ decade. Do not call them 10⁻⁵.
- State that seed 0 comes from an earlier session with the identical config. Without it, the
  Colab-only median is 1.40e-4 (n=4).
- Do not claim an order-of-magnitude gain from causal weighting. The median ratio is 2.3×.
  The order-of-magnitude statements that hold are windowed vs single network (about 4.4 orders)
  and separate vs shared networks (about 3.8 orders).

## 3. Per-run table (appendix)

| run | seed | device | Adam | L-BFGS evals | capped | RMSE | max error | residual MSE | E drift | closure ‖u(T)−u(0)‖ |
|---|---|---|---|---|---|---|---|---|---|---|
| r4 candidate (run of record) | 0 | cuda | 34,765 | 33,143 | 2 | 6.97e-5 | 3.09e-4 | 1.87e-7 | 4.05e-4 | 5.64e-5 |
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

## 4. LaTeX (booktabs, IEEEtran column width)

```latex
\begin{table}[t]
\caption{Ablations over the full orbit (median [range] over seeds; errors vs DOP853)}
\label{tab:ablations}
\centering\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}
\toprule
Arm & $n$ & Adam steps & RMSE & Residual MSE\\
\midrule
Candidate (causal, warm, 27 nets) & 4 & 34,769 & $1.40\times10^{-4}$ [$0.75$--$2.60$] & $1.80\times10^{-7}$\\
No causal weights & 3 & 35,100 & $1.89\times10^{-4}$ [$1.75$--$5.16$] & $9.49\times10^{-7}$\\
No warm start & 3 & 102,989 & $2.85\times10^{-4}$ [$1.76$--$2.88$] & $8.54\times10^{-7}$\\
One shared net, 27 windows & 3 & 35,100 & $0.486$ [$0.350$--$0.648$] & $1.03$\\
Single net + causal & 1 & 40,000 & $0.275$ & $5.23\times10^{-6}$\\
\midrule
Single net (F1, run of record) & 1 & 40,000 & $2.237$ & $2.80\times10^{-4}$\,\textsuperscript{\dag}\\
\bottomrule
\multicolumn{5}{l}{\scriptsize Ranges in units of $10^{-4}$ for the first three rows. \textsuperscript{\dag}collocation-grid residual.}
\end{tabular}
\end{table}
```

## 5. Findings

1. **Headline:** a median RMSE of 8.1×10⁻⁵ over five seeds, with 3 of 5 seeds below 1e-4 and
   none of the no-causal or no-warm-start runs below it. Seed 0 (6.97e-5) is the best of the five.
2. **Windowing is the main effect.** RMSE goes from 2.24 (F1) to 8.1e-5, about 4.4 orders of
   magnitude. Causal weighting on a single network (r3) gives only 8× (0.275), and all of its
   causal stages hit the 10,000-step cap.
3. **Separate networks per window are essential.** r6 is 680-3,400× worse than r1 in every
   seed. A residual MSE of about 1 means earlier windows are forgotten; this is not an
   optimizer stall.
4. **Warm start improves accuracy and cost.** Without it, runs take 3× more Adam steps, hit
   the step cap in 17-19 of 108 stages (vs 1-2), and are worse in 3 of 3 paired seeds.
   Max error, residual and E drift are fully separated.
5. **Causal weighting gives 2.3× in median RMSE and 5.2× in residual MSE.** The RMSE gain is
   not significant (p = 0.20; r1 is better in paired seed 2). The residual gain is fully
   separated. That is far from the 10-100× in Wang et al.
6. **Open confound:** no single network with about 304k parameters (matching the windowed
   model) has been run. r6 does not cover this.

## 6. More runs (all resume-safe on Colab; finished arms are skipped)

| priority | command | why | cost (T4) |
|---|---|---|---|
| recommended | `python scripts/paper_ablations.py --arms r1 --seeds 0,4` | gives the causal claim, the weakest one, 5 paired seeds (exact MWU can reach p = 0.004) | ~20-60 min each |
| useful | new config: single network, about 300k params, with F1's grid and budget | closes the "27× parameters" objection | ~1.5-2 h |
| optional | `--arms r3 --seeds 1,2`, plus F1 seeds 1-2 | turns the single-network causal effect into mean ± std | ~2 h + ~1.3 h per seed |
| optional | `--arms r2 --seeds 0,4` | n=5 for warm start; the claim is already well supported | ~1.5-3 h each |

After new runs, copy them into `runs/paper_ablations/` and rerun `python scripts/ablation_report.py`.

## 7. Open tasks for the paper (`writings/conference-paper/main.tex`)

1. Results: add `tab:ablations` (§4) and the median/range sentence (§2). Keep `tab:errors`
   labelled as "seed 0, the lowest-RMSE of five seeds, shown in the figures".
2. Discussion and limitations: replace "a budget-matched ablation ... accompanies the released
   run artifacts" and "seed sensitivity is quantified in the released run artifacts" with the
   numbers from §2.
3. Do not let the Wang et al. "one to two orders of magnitude" citation read as reproduced.
4. Conclusion: move "future work isolates each training option ... multi-seed protocol"
   into the present tense; the ablations now do exactly that.
5. Keep the capacity confound (27× parameters) as a stated limitation unless that run is added.
