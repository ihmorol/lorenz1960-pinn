# Evidence map for the conference draft

This file is an internal claim ledger. The prose deliverable is `../main.tex`.

| ID | Source and level | Supports | Cannot support | Used in | Risk |
|---|---|---|---|---|---|
| E1 | `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/history/config.json`, L1 | Exact main-run configuration | Actual optimizer counts or accuracy | Method, protocol | `epochs` and `lr_end` are inactive for this mode |
| E2 | Same run's `metrics.csv` and `run_summary.csv`, L1 | Evaluation-grid errors, residuals, invariant drifts, recorded cost | Causal attribution, seed variability, normalized GPU throughput | Abstract, results | One seed; device model absent |
| E3 | Same run's `history/stages.csv`, L1 | Stage counts, thresholds, caps | Proof that capped stages converged | Method, results | Cap is not the threshold |
| E4 | Same run's `breakdown/epoch_067908.csv`, L1 | Final collocation-point state, reference, signed errors, residuals and the plotted trajectory figure | The exact 13,265-point evaluation-grid metrics | Fig. 2 | Final snapshot is on training points |
| E5 | `src/pinn/pinn.py`, `src/pinn/functions/losses.py`, `src/pinn/functions/collocation.py`, `src/pinn/train.py`, L1 | Trial function, independent networks, handoff, loss, point assignment, optimizer order | Comparative effectiveness of a component | Method | Code inspection, not an ablation |
| E6 | `src/pinn/viz/figures.py` (`invariant_series`), L1 | The evaluator's orthonormal null-space invariant construction | The incorrect draft forms $x^2+y^2$ and $x^2+z^2$ | Problem, protocol | Basis is implementation-specific |
| E7 | `runs/4x60_f64_unit/run_summary.csv` and `metrics.csv`, L1 | Saved full-interval result and configuration | Controlled single-factor comparison | Protocol, results | Different settings and unknown GPU model |
| E8 | `docs/results/2026-09-22-two-branch-code-audit.md` and `2026-09-23-three-causal-run-comparison.md`, L1 user-provided audits | Interpretation boundaries and known artifact semantics | Replacement for the saved metrics | Results, discussion | Some notes concern older code or runs |
| E9 | Lorenz, DOI `10.1111/j.2153-3490.1960.tb01307.x`, verified Crossref metadata, L3 | Publication identity and background attribution | Claims about this branch's coefficients or numerical performance | Introduction, problem | Cite at attribution level |
| E10 | Raissi et al., DOI `10.1016/j.jcp.2018.10.045`, verified Crossref metadata, L3 | PINN publication identity | Specific performance here | Introduction, related work | Cite at concept level |
| E11 | Lagaris et al., DOI `10.1109/72.712178`, verified Crossref metadata, L3 | Neural trial-function prior work | Implementation-specific details of this branch | Introduction, related work | Cite at concept level |
| E12 | Wang et al., DOI `10.1016/j.cma.2024.116813`, publisher abstract and Crossref metadata, L2 | Prior causal PINN weighting direction | Causal efficacy of this combined branch | Introduction, related work | Do not present the borrowed weighting as new |
| E13 | Penwarden et al., arXiv `2302.14227` abstract and JCP DOI `10.1016/j.jcp.2023.112464`, L2 | Causal sweeping and temporal decomposition prior-work direction | Performance or mechanism for this ODE | Related work | Do not treat windowing as new in general |
| E14 | SciPy project citation page and DOI `10.1038/s41592-019-0686-2`, L3 | SciPy publication identity | Accuracy guarantee of DOP853 in this run | Protocol | Solver tolerances come from branch files |

## Contribution-to-evidence trace

| Claim in paper | Method location | Result location | Evidence boundary |
|---|---|---|---|
| Windowed PINN with hard handoff and causal stages is implemented | Sec. III and Algorithm 1 | Tables I–II | E1, E3, E5; no component-level efficacy claim |
| The saved model closely follows the numerical reference on the chosen orbit | Sec. IV metrics | Table II, Fig. 2 | E2, E4; one seed, one orbit |
| The full-interval configuration performs poorly here | Sec. IV comparison protocol | Table III | E7; differences are not isolated |
