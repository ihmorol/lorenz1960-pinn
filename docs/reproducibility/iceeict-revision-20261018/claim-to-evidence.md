# Claim-to-evidence and change record

Scope: new writings/iceeict2027-submission only. No training or new experiment was performed. Prior folders and both supplied ZIPs are preserved.

| Issue finding | Correction in the manuscript | Evidence |
| --- | --- | --- |
| C1 / m1 | Retain the preview's 33-word Results roadmap deletion and removal of stale IEEEtriggeratref; no typography compression. | Baseline manifest and final PDF/build verification. |
| M1 | Reference comparison states DOP853 only. No RK4 benchmark claim. | Saved run configuration/reference grid and archived evaluations. |
| M2 | Reported seed-0 snapshot interval is 2,000 updates; stage cap is recorded as capped, not trajectory failure. | runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/history/config.json: snapshot_every=2000. |
| M3 | Replace necessity/forgetting/universal-success wording with measured final-error and warm-start training-cost findings. Positive scoped conclusion; future matched configurations and other states/coefficient sets. | All 16 saved summaries; matched-seed no-warm-start comparisons; no intermediate-time retention measurements asserted. |
| M4 | Define both invariants using the two columns of the evaluator's orthonormal null-space basis of a^T and give maximum drift divided by each initial magnitude. | src/pinn/viz/figures.py:562-580 and src/pinn/train.py:470-475; seed-0 summary bounds; evidence-check.json includes basis, initial values and saved drifts. |
| M5 | State exploratory exact one-sided Mann-Whitney test, larger no-causal errors, seed groups, U=11 and p=0.196; retain small-group/shared-seed caution. | scripts/ablation_report.py:55-67; archived summaries reproduce p=0.19642857142857142. F1 is not pooled. |
| M6 | Correct Revanth Mattey, final bc-PINN title, Rohrhofer author list, exact titles and metadata. Explain retention constraint absent from our shared-network control. | reference-audit.md: all 13 cited entries checked against primary records. |
| M7 | Figure decision/measurements are recorded separately. No typography pass claimed merely from font embedding. | figure-audit/ and final PDF typography measurements. |
| m2 | Explicitly identify candidate seeds 0-4; no-causal/no-warm/shared 1-3; causal-single and full-interval seed 0. | All 16 run records and unchanged ablation table. |
| m3 / m4 | Repair abstract fragment, agreement, tense, Introduction grammar and local comma splice. Keep technical motivation concise. | Source diff; all nine numbered equations, display math and algorithm unchanged. |
| m5 | Define PINN/RMSE at first body use; LHS before the comparison table; full L-BFGS name at its first body occurrence. | Revised source and final PDF. |
| m6 | Local nonbreaking citation spacing. | Source diff and reference-resolution build checks. |
| m7 | Give concise, supported figure captions; retain index-convention note for the pipeline. | Final source/PDF; no figure data claim invented. |

## Preserved results and methods

All three inline result/configuration tables, ablation-table.tex, nine numbered equations, display-math blocks and algorithmic environment were compared directly with the preview and remain unchanged. Only prose and captions changed. Both IEEEtran class/style files are byte-identical to the baseline. The data are the paper's 16 completed runs only; later stability studies are excluded.

The conclusion's stronger presentation comes from leading with demonstrated accuracy, the saved full-interval comparison and matched-seed warm-start results. It does not assert a globally optimal architecture, isolated causal benefit, general solver superiority or experimentally measured forgetting.

## Author input, recorded without private identifiers

On 18 October 2026, the submitting author confirmed authorship/order agreement, supervisor and author approval to revise, no prior submission/no planned parallel submission, and the non-intellectual LLM-use history. Latest reported screening results are 7% similarity and 0% on an AI detector; earlier 0% similarity wording was superseded. Reports were not independently inspected, and the revised PDF was not rescanned. Approval of this newly compiled PDF remains a later author step, as do CMT/presenter checks and exact cutoff confirmation.
