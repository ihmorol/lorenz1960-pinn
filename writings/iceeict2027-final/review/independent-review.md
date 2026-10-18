# Independent final submission check

18 October 2026. Read-only review of the regenerated figures and final manuscript. Paper source commit: `9a9e5ab56338c0e5c4c7b8d1a754179612853268`. PDF SHA-256: `9494f54f1ec97d4a3bee13c866bb03aeec4b747d7478bdbca2be2a3a50f80de9`.

## Verdict

PASS for the technical file and figure revision reviewed here. Findings: **0 critical, 0 major, 0 minor**. The preceding figure typography finding M7 is resolved. This is a file-readiness verdict, not an acceptance or indexing prediction.

## Direct checks

| Check | Result and evidence |
| --- | --- |
| Every final page viewed | All six rendered pages inspected individually. Their pixels match fresh PyMuPDF renders of the audited PDF at 108 dpi exactly. No clipped content, overlaps, missing glyphs or misplaced floats found. |
| Figure visual inspection | Pipeline, full-interval result and windowed solution/error figures viewed individually and in the manuscript. Axis labels, ticks, scale notation, legends and arrowheads are readable; none are clipped. |
| Figure fonts | All three vector figures use embedded Times New Roman. Smallest Times New Roman text in the placed manuscript is 8.5 pt, exceeding the 8 pt target. No downscaling below the intended label size. |
| A4 and page limit | Six pages, each 595.28 x 841.89 pt. References, figures and tables included. |
| PDF size | 2,697,666 bytes, below 3,000,000 bytes. |
| Font embedding | All 26 PDF font programs embedded; zero Type 3 fonts. |
| Build diagnostics | No overfull boxes, unresolved citations/references or LaTeX errors in the final log. |
| Numeric figure provenance | Both arrays contain all 13,265 strictly increasing evaluation times and finite three-state predictions/reference values. Array hashes match the manifest; both plots use an identical reference and time grid. Saved loss history hash also matches. |
| Numeric agreement | Recomputed combined RMSE: full interval 2.2370605122285716, windowed 0.00006966965305660092. These agree with archived 2.23706051 and 0.0000696696531 to recorded precision. |
| Plot meaning | Full-interval solution/reference and signed component errors show large trajectory error despite small final residual. Windowed figure separates the overlapping solutions from signed errors explicitly scaled by 10^-4. Curves retain all evaluation points. |
| Pipeline semantics | Four hidden layers of 60 tanh units; 27 windows indexed 0 through 26; hard initial state; 1,536 collocation points; four causal stages; detached causal weights; verified minimum-weight threshold or 4,000-update cap; unweighted L-BFGS; endpoint and parameter handoff are represented faithfully. |
| Protocol wording correction | The revised sentence describes certification before accepting a stage transition, consistent with src/pinn/train.py:310-317. It no longer asserts post-update certification after every Adam update. Figure caption indexing agrees with the algorithm. |
| Preserved science | All nine numbered equation environments, the algorithmic environment and three inline tabular environments match the preceding submission source exactly. ablation-table.tex, references.bib, IEEEtran.cls and IEEEtran.bst are byte-identical. |
| New training | None. This revision uses archived checkpoint inference and saved evaluation/loss data; no training results or statistical claims were introduced. |

## Submission requirement carry-forward

The previous 24-row requirements audit remains the policy/author-attestation record, with these current file updates: S03 now measures 2,697,666 bytes; S06-S07 now cover 26 embedded fonts; S09 passes the figure typography checks instead of retaining the original failure. S01-S07, S09, S15, S18 and S19 remain technically satisfied on this bounded inspection. No identifying author block or repository link appears in the rendered anonymous paper.

S08, S10, S12-S13 and S16 remain author-confirmed. S11 and S17 require approval of this exact final PDF/output. S14 remains the author's prior reported 7% similarity result; this revised file was not rescanned. S20-S23 concern presenter, no-show policy and account obligations; S24 remains conditional on conference scope/quality decisions.

## Limits

No new literature/novelty search, model training, similarity/AI-detector screening, author-consent authentication, CMT account inspection or upload was performed. Existing scientific claims and previous reference corrections were preserved rather than reopened. Final author review and submission-system checks remain outside this technical figure/layout audit.
