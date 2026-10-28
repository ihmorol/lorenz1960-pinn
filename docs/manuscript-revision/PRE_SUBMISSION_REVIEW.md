# Pre-submission review: windowed causal PINN paper

Reviewed 8 October 2026 against `writings/conference-paper/main.tex` at commit `8323afe` (6 pages, IEEEtran conference). Paradigm: STEM / computational. Target venue not stated, so venue fit was not checked. Novelty and citation-completeness retrieval was not run; the Related Work findings below come from the paper and its own `references.bib`.

## Summary

- CRITICAL: 1
- MAJOR: 9
- MINOR: 13
- Top three fixes first:
  1. Re-export Fig. 1 (`platue-results.png`) as a vector PDF.
  2. Align the claims with the ablation evidence. The title leads with "Causal Training", but the no-causal ablation overlaps the candidate (Mann–Whitney p = 0.196 on RMSE, ablation-report.txt). Also fix the false "precision" difference in Run Comparison.
  3. Add an explicit contributions statement to the Introduction and expand Related Work using the eight uncited entries already in `references.bib`.

## Dimension 1: Macro logic

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Title "Sequential Causal Training of Windowed PINNs" and abstract lead with causal weighting, but Section V-C says the no-causal median is only "about 2.3 times larger" and "for seed 2 the no-causal run has lower RMSE". The report gives MWU p = 0.196 on RMSE (n = 5 vs 3). The ablations most strongly support *separate window networks* (0.350–0.648 vs ~1e-4) and warm start. | MAJOR (attribution unverified for the titled mechanism) | Either retitle around windowing (e.g. "Sequential Training of Windowed PINNs with Causal Weighting…") or keep the title and state in the abstract that the causal gain is not significant at this seed count. Report the p-value or say "three seeds" next to the 2.3× figure. |
| 2 | Run Comparison: "which differ in causal weighting \cite{wang2024causal}, windowing \cite{penwarden2023} and warm starts as well as precision and even L-BFGS." Both runs are float64 (`verified-ablation-values.csv`, F1 row `dtype=float64`), and both use L-BFGS (5,000 vs 1,000 iterations). | MAJOR (factual error) | Replace "as well as precision and even L-BFGS" with "as well as the collocation grid, the Adam schedule and the L-BFGS budget". |
| 3 | The Introduction ends with "We also ran ablation and failure cases, where we removed…" but never states the paper's contributions. The deleted sentence "This paper provides an implementation and its saved results" was the only contribution line. | MAJOR | Add 2–3 sentences, e.g. (1) a windowed PINN with hard handoff and causal stages for one Lorenz-1960 orbit; (2) a seed-0 error of 6.97e-5 and a five-seed median of 8.13e-5 against DOP853; (3) an ablation suite showing which components matter. Map each to Sections III, V-A and V-C. |
| 4 | Related Work cites only four works ("raissi2019", "lagaris1998", "wang2024causal", "penwarden2023"). `references.bib` already holds `krishnapriyan2021`, `wang2022ntk`, `wang2021gradient`, `mattey2022bcpinn`, `rohrhofer2023`, `hao2024pinnacle`, `mcgreivy2024`, `rathore2024`, none of them cited. Reviewers will call the background thin. | MAJOR | Add one short paragraph citing the failure-mode works (krishnapriyan2021, wang2022ntk, wang2021gradient, rohrhofer2023) and the time-marching/backward-compatible PINN (mattey2022bcpinn), which is the closest prior work to the window handoff. Check each claim against the source before citing. |
| 5 | "The Lorenz system general form \cite{matthews2024pinnde} is:" cites a 2024 software arXiv preprint for the 1960 equations. The same source is cited for the initial state. | MAJOR | Cite `lorenz1960` for Eq. (1). Keep matthews2024pinnde only for the initial state if that is where it came from. |
| 6 | Network Structure: "$T=13.26446$ is one closed loop" and Related Work: "this particular closed orbit", while Section II-A says "not an exact period". | MINOR | Use "nearly closed orbit" in both places (the revision plan's rule). |
| 7 | "We tried to keep the split window closer to 0.5, as small windows are better \cite{wang2024causal}." The cited causal-training paper does not establish a 0.5 width. | MINOR | Drop the citation or say "we chose a width near 0.5". |
| 8 | "The DOP853 reference drifted less than $4\times10^{-10}$ for perspective." The quantity that drifted (an invariant? which one?) is never defined. | MINOR | Name the invariant and whether drift is absolute or relative, or remove the sentence. |

## Dimension 2: Writing details

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Over-long paragraphs: the Introduction is one ~20-line paragraph; Training Procedure ("The causal loss defined above … squared residual components.") and the saved-state paragraph ("The code saves the parameters of each window…") are each 12+ lines. | MAJOR | Split the Introduction at "To overcome this ordering" and at "We use both concepts". Split Training Procedure at "When all four stages are completed". |
| 2 | "When the saved minimum weight is above 0.99 or if there is an Adam update then the code will recalculate the causal weights at all collocation points" is hard to follow and partly repeats the next sentence. | MINOR | Say once that weights are recomputed after every Adam update, then give the 0.99 rule. |
| 3 | "Each window is approximately $0.491$ time units wide." repeats the display line $\Delta t\approx0.4912763$ directly above. | MINOR | Delete the sentence (also saves a line). |
| 4 | Table IV ("One network" vs "27 windows") and the "Full interval" row of Table I both report 2.237. | MINOR | Fine to keep, but mention once in V-B that Table I's full-interval row is the same run. |

## Dimension 3: English grammar

| # | Quoted text | Rule (grammar-rules.md) | Severity | Fix |
|---|---|---|---|---|
| 1 | "As it failed by stucking in a platue so we tried to split" | §7 Chinglish ("As… so"), spelling | MAJOR (density, see note) | "As it got stuck in a plateau, we split the time interval…" |
| 2 | "The 27 networks don't share parameters but the windows has shared boundary." | §2 subject-verb agreement; contraction | MINOR | "do not share … but neighbouring windows share a boundary." |
| 3 | "a separate network for every windows" | §2 number agreement | MINOR | "for every window" |
| 4 | "In first experiment, we used one network" | §1 articles | MINOR | "In the first experiment" |
| 5 | "We use both concepts, we split the time period in equal 27 windows" | §4 comma splice | MINOR | "We use both concepts. We split the time period into 27 equal windows" |
| 6 | "The actual training goal and window handoff is then given" | §2 agreement | MINOR | "are then given" |
| 7 | "the error show small peaks in the errors particularly at $t\simeq5$" | §2 agreement, repetition | MINOR | "the error shows small peaks, particularly at $t\simeq5$" |
| 8 | "we divide the interval in total 27 windows" (abstract) | §7 | MINOR | "we divide the interval into 27 windows" |
| 9 | "Each of our 27 window PINN passes each of the tests" (Conclusion) | §7 / unclear referent | MINOR | "Our 27-window PINN passes every check on…" |
| 10 | "The completed windows predicted endpoint" | §1 possessive | MINOR | "The completed window's predicted endpoint" |
| 11 | "to ensure our starting state precisely at the point we want to implement it ." | §7, stray space | MINOR | "to fix the starting state exactly" |
| 12 | "This results in significant increase in loss" | §1 articles | MINOR | "This results in a significant increase in error" |

Note: the 12 items are individually minor, but together they will draw a "needs language editing" comment, so the group counts as one MAJOR item. Spelling also mixes British and American ("modelling", "generalise" vs "behavior").

## Dimension 4: LaTeX format

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Author block lists student IDs ("0112230489", …), and the first five authors have no university line. If the venue is double-blind, names must go entirely. | MAJOR | Remove IDs; give every author "Dept. of CSE, United International University, Dhaka, Bangladesh". Check the venue's blind-review rule. |
| 2 | Citations and references lack the non-breaking tilde: "previous works\cite{wang2024causal}" (no space at all), "hard trial solution  \cite{lagaris1998}" (double space), "Table \ref{tab:config}", "Figure \ref{fig:pipeline}". | MINOR | Use `works~\cite{…}`, `Table~\ref{…}`, `Fig.~\ref{…}` throughout. |
| 3 | `eq:ode` and `eq:rmse` are numbered but never referenced. | MINOR | Reference them, or make them unnumbered. |
| 4 | Table caption "windowed-run configuration" starts lower-case; ablation caption uses the internal label "F1 is a separate reference", which is never defined in the text. | MINOR | Capitalise; replace "F1" with "the full-interval run". |
| 5 | Page limit: 6 pages including references, compliant. Note that the last column is full, so any addition needs an equal cut. | Info | |

## Dimension 5: Figure quality

| # | Finding | Severity | Suggested fix |
|---|---|---|---|
| 1 | Fig. 1 is `figures/platue-results.png`, a 4458×1163 raster. `plateau_training.pdf` in the same folder is a different plot (loss only), not a vector copy. | CRITICAL (raster in final draft) | Re-export the three-panel plot as PDF from its plotting script and rename it `plateau-results.pdf`. Increase font size; at column width the tick labels render at roughly 2–3 pt. |
| 2 | Captions do not state a finding: "Training result of a single network , entire time at once" (also stray space before comma), "Complete implementation pipeline", "PINN solution vs reference with signed errors." | MAJOR | Lead with the result, e.g. "Single network on [0, T]: the prediction leaves the reference after t≈2 and the loss plateaus near 3e-4 (state RMSE 2.237)." and "Seed-0 windowed PINN against DOP853; signed errors stay below 3.1e-4." |
| 3 | Fig. 2 (`method.pdf`) labels "REPEAT FOR EACH WINDOW k = 1 … 27", while all equations and Algorithm 1 use k = 0…26. | MINOR | Edit the figure to 0…26, or add "windows are numbered 1–27 in the figure" to the caption. |
| 4 | Fig. 1 middle panel is titled "MSE: 1.67e+00", a metric the text never uses (the text reports RMSE 2.237; 2.237²/3 ≈ 1.67). | MINOR | Retitle the panel "RMSE 2.237" to match the text. |

## Banned-vocabulary and em-dash scan

Full scan of `main.tex` (all lines, not sampled):
- Em dashes as sentence connectors: none.
- Listed AI-tone words: none. "significant" appears once (line 138), outside the list and acceptable.

## Final score (1-10)

5. One CRITICAL item and nine MAJOR items. All are text-level or figure re-exports; none needs new experiments.

## Submission recommendation

Needs 1-2 days more work. Fix the raster figure, the causal-attribution framing, the "precision" error and the contributions statement first; the grammar pass and captions can follow in a single sitting.
