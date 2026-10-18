# Independent pre-submission review

Date: 18 October 2026
Target: ICEEICT 2027 initial anonymous submission
Paradigm: bounded numerical study of a physics-informed neural network for an ordinary differential equation
Source commit: `a85ccb66be4bbce5631f80653fbcfaeebeea9d09`
Source SHA-256: `2777a67000eee743c275dd074398ccaa0822a2f5cf83189003464f178a7e023d`
PDF SHA-256: `8d31c0068bb7ac6a80bd54287936f85f487c31251a28a1a0904a71a731f896ed`

## Verdict

**PARTIAL: technically coherent and six-page ready, but not fully template-compliant.**

The manuscript has no remaining critical scientific, claim-accuracy, grammar, bibliography, anonymity, pagination, font-embedding or build defect found in this audit. One known major format defect remains: the labels inside all three vector figures are below the supplied template's 8-point Times New Roman requirement. The author-directed fallback retains the visually authoritative figures because the recovered font-only candidate changed their appearance and still did not achieve 8-point labels.

This is a viable submission candidate if the authors knowingly accept that format risk. It must not be described as fully compliant. The exact final PDF also still needs author approval, and the CMT/presenter/deadline checks remain deferred.

### Summary

- CRITICAL: 0
- MAJOR: 1
- MINOR: 0
- Final score: **7/10**
- Submission recommendation: **Needs final author checks and explicit acceptance of the figure-format risk before submission.**

## Direct evidence

- PDF: six A4 pages, 498,928 bytes, PDF 1.5.
- Fonts: 31 embedded fonts, zero Type 3 fonts, zero unembedded fonts.
- Metadata/anonymity: empty title/author/subject/keyword metadata; no author name, affiliation, email, identifying repository link or `@` string detected in the PDF.
- Build: four-pass `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` record produced six pages with no undefined citation/reference, overfull box or build error. One underfull acknowledgment line is visible only as a benign log warning.
- Visual inspection: all six pages rendered at 144 dpi. No clipping, overlap, missing glyph, misplaced float or unreadable body/table/equation text was found.
- Figures: vector PDFs with embedded, non-Type-3 fonts. Minimum printed label sizes are 4.153 pt for `method.pdf`, 2.663 pt for `plateau-results.pdf`, and 3.828 pt for `solution_vs_reference.pdf`.
- Claim evidence: 16 archived paper runs only; no training or new experiment was performed during revision.
- Statistics: exact one-sided Mann-Whitney U test, no-causal seeds 1--3 versus WPINN seeds 0--4, `alternative="greater"`, `method="exact"`, gives `U=11` and `p=0.1964285714`. The manuscript labels this exploratory and does not claim significance or equivalence.
- Invariants: both quadratic invariants are defined from the two-column orthonormal null-space basis of `a^T`; each maximum drift is normalized by its nonzero initial magnitude. The saved maximum pairwise drifts match the manuscript bounds.
- References: all 13 cited entries resolve; cited metadata and supporting descriptions were checked against primary records. The Mattey/Ghosh and Rohrhofer author records are corrected. Two uncited BibTeX entries are not printed.
- Full banned-vocabulary and Unicode em-dash scan of active `main.tex` and `ablation-table.tex`: no matches. Comments and bibliography titles were excluded from this prose-style scan.
- Equations: nine numbered equations, contiguous; every numbered equation is referenced in the body.

## Reviewer-style findings

| # | Dimension | Finding | Severity | Required handling |
| --- | --- | --- | --- | --- |
| F1 | LaTeX and figure quality | Figure labels such as "Problem setup" and "Solution vs reference" are rendered at approximately 2.66--4.15 pt and use Segoe UI or DejaVu Serif rather than 8-point Times New Roman. | MAJOR | Keep the existing figures only under the author's explicit fallback and record S09 as failed. Do not claim strict template compliance. A compliant replacement requires redesigned figure geometry, not a font-family substitution. |

No critical or minor finding remains after the final local grammar and equation-reference corrections. The scientific claims are bounded to the saved orbit and recorded configurations. The conclusion does not claim an isolated causal-weighting benefit, global optimality, general solver superiority or measured forgetting.

## Five-dimension review

### Dimension 1: macro logic

The Introduction states the long-horizon PINN problem, the windowed training structure, the recorded comparison and three evidence-backed contributions. Method and Results sections map to those contributions. The full-interval comparison is explicitly descriptive because parameter count, sampling and optimizer schedules differ. Ablations support separate window networks and warm-start training cost for this orbit, while the text explicitly declines a consistent causal-weighting accuracy claim.

No blocking macro-logic break was found.

### Dimension 2: writing details

The abstract contains the problem, method, numerical results and comparison limits. Section previews and captions are concise. Repetition was reduced without removing methods, numerical values or limitations. The final six-page layout remains readable.

No actionable writing-detail finding remains.

### Dimension 3: English grammar

The final pass corrected the remaining article/word-order issue, ambiguous antecedent, nonrestrictive `which` punctuation, compound modifiers and awkward related-work/section-preview sentences. These corrections follow G1, G4 and G5 from the review grammar rules. Tense is consistent enough for the paper's method and experimental narrative.

No actionable grammar finding remains.

### Dimension 4: LaTeX formatting

The IEEEtran conference A4 two-column structure, bibliography style, equation references, nonbreaking citation/reference spacing, six-page limit and embedded-font requirements pass. Figure typography is the single exception recorded as F1. One underfull acknowledgment line does not create visible damage and is not a submission blocker.

### Dimension 5: figure quality

All three figures are vector, unclipped, correctly captioned and referenced before appearance. Their data, curves, colors, panel order and geometry remain unchanged. The recovered method-figure source reproduces the authoritative figure pixel-for-pixel; the tested Times candidate changed label baselines and still remained below 8 pt. F1 therefore remains open by deliberate fallback.

## Submission requirements S01--S24

| ID | Requirement | Verdict | Evidence or remaining action |
| --- | --- | --- | --- |
| S01 | IEEE double-column template | PASS | `IEEEtran` conference A4, two columns; class/style unchanged. |
| S02 | PDF format | PASS | Final `main.pdf` opens and all pages rendered. |
| S03 | File size at most 3 MB | PASS | 498,928 bytes. |
| S04 | A4 paper | PASS | All pages are 595.276 by 841.89 PDF points. |
| S05 | Maximum six pages including all content | PASS | Exactly six pages including figures, tables and references. |
| S06 | No Type 3 fonts | PASS | Zero Type 3 fonts. |
| S07 | All fonts embedded | PASS | All 31 fonts embedded. |
| S08 | Every author meets authorship criteria | AUTHOR-CONFIRMED | Submitting author confirms contribution/authorship alignment; private evidence not independently inspected. |
| S09 | Strict IEEE typesetting and bibliography | FAIL | Bibliography and body issues pass; figure labels remain below 8 pt and are not Times New Roman. |
| S10 | Agreement on author sequence | AUTHOR-CONFIRMED | Author and supervisor alignment reported. |
| S11 | All authors approve submission | PENDING FINAL FILE | General approval and revision authorization are confirmed; exact PDF hash still needs final author review. |
| S12 | Original work, not previously published or submitted | AUTHOR-CONFIRMED | No prior submission reported. |
| S13 | No simultaneous submission without permission | AUTHOR-CONFIRMED | No planned parallel submission reported. |
| S14 | Similarity strictly below 30 percent | AUTHOR-REPORTED PRIOR PASS | Author reports 7 percent similarity. Report not inspected and revised PDF not rescanned. The reported 0 percent AI-detector result is not treated as a venue requirement or proof of authorship. |
| S15 | LLM use acknowledged | PASS | Acknowledgment limits use to formatting, typesetting, language editing and proofreading. |
| S16 | LLM use solely non-intellectual | AUTHOR-CONFIRMED | History matches the acknowledgment; no research result was generated in this revision. |
| S17 | LLM output reviewed and approved | PARTIAL | Prior review history confirmed; exact final PDF approval remains pending. |
| S18 | Authors accept responsibility | PASS | Explicit responsibility statement is present. |
| S19 | Double-blind initial submission | PASS ON INSPECTION | Anonymous text and metadata; no identifying repository link. External online matching is outside local proof. |
| S20 | Qualified presenter if accepted | DEFERRED | Future author obligation; no current PDF defect. |
| S21 | No-show exclusion from Xplore | POLICY CONSEQUENCE | Recorded for later compliance. |
| S22 | No-show fee non-refund | POLICY CONSEQUENCE | Recorded; no fee transaction checked. |
| S23 | Correct-email CMT account for every coauthor | DEFERRED | Account creation was reported earlier; fields were not authenticated in this audit. |
| S24 | Xplore inclusion subject to scope and quality | CONDITIONAL OUTCOME | Cannot be guaranteed before conference review and processing. |

## Scientific criteria Q01--Q19

Scale: 0 Poor, 1 Average, 2 Good, 3 Excellent. Ratings are reviewer-style judgments, not an acceptance score or venue cutoff.

| ID | Criterion | Rating | Reason |
| --- | --- | ---: | --- |
| Q01 | Rationale and motivation | 2 | Clear long-horizon PINN problem and numerical-reference rationale; intentionally bounded to one orbit. |
| Q02 | Venue fit | 2 | Physics-informed learning and numerical computation fit the conference's broad ICT/engineering scope; track selection was not authenticated in CMT. |
| Q03 | Contribution importance | 1 | Useful controlled case study, but importance beyond the studied orbit is untested. |
| Q04 | Innovation | 1 | The paper combines established hard constraints, causal weights, windowing and warm start; it does not claim invention of those components. |
| Q05 | Potential advance | 1 | Results motivate broader matched studies, but present evidence is deliberately narrow. |
| Q06 | Clarity of writing | 2 | Definitions, configuration distinctions and claim limits are explicit after local editing. |
| Q07 | Organization | 2 | Contributions map to method/results sections; related work and experimental setup are placed logically. |
| Q08 | References | 2 | All 13 cited records and supporting descriptions were primary-source checked; exhaustive literature coverage was not certified. |
| Q09 | IEEE style | 1 | Main template, equations, citations and bibliography pass; figure-label typography fails. |
| Q10 | Method strategy | 2 | Coherent sequential design with hard handoff and staged optimization; unmatched controls limit attribution. |
| Q11 | Reproducibility information | 2 | Architecture, grids, optimizer schedule, seeds, evaluation and counts are detailed; the historical dirty source diff was not retained. |
| Q12 | Methodological rigor | 2 | All 16 completed runs are reported and failure/cap semantics are separated; sample sizes remain small. |
| Q13 | Analysis | 2 | Error, residual, invariant, training-cost and ablation analyses are present with correctly bounded statistics. |
| Q14 | Result collection and reporting | 2 | Five WPINN seeds and all saved ablations are reported without success-based selection; only one initial-value problem is studied. |
| Q15 | Reasoning and concision | 2 | Conclusions track the evidence and acknowledge configuration differences without excessive qualification. |
| Q16 | Tables and figures | 1 | Tables are clear and figures are scientifically readable, but internal label typography is noncompliant. |
| Q17 | Supplementary material | N/A | No supplement is required for the initial paper; the anonymous PDF intentionally omits the public-repository link. |
| Q18 | Discussion and interpretation | 2 | The paper distinguishes final-error evidence from causal or forgetting claims and interprets the nonsignificant exploratory test correctly. |
| Q19 | Limitations | 2 | Unmatched settings, small groups, single orbit and future generalization are stated without weakening verified results. |

Total: **31/54 over 18 scored criteria**, mean **1.72/3**. The strongest improvement is evidence discipline and reproducibility reporting. Innovation and potential advance remain evidence-limited and cannot be upgraded through wording alone.

## Issue #39 finding closure

| Finding | Status | Final evidence |
| --- | --- | --- |
| C1 | CLOSED | Final paper is six pages without margin/body-font compression. |
| M1 | CLOSED | DOP853 is the only claimed numerical reference; RK4 claim removed. |
| M2 | CLOSED | Seed-0 snapshot interval is 2,000 updates; caps are not called failed jobs. |
| M3 | CLOSED | Necessity, measured forgetting and universal-success claims replaced by scoped final-error/training-cost statements. |
| M4 | CLOSED | Both invariant definitions and normalization are stated and match the evaluator. |
| M5 | CLOSED | Exact one-sided test, groups, direction, U, p and limitations are given. |
| M6 | CLOSED | Mattey/Ghosh metadata and bc-PINN distinction corrected; all cited entries checked. |
| M7 | OPEN, AUTHOR-ACCEPTED FALLBACK | Figure labels remain 2.66--4.15 pt and non-Times. |
| m1 | CLOSED | Redundant Results roadmap and stale forced reference break remain removed. |
| m2 | CLOSED | Seed groups and all 16 completed runs are stated accurately. |
| m3 | CLOSED | Abstract grammar, agreement and tense corrected. |
| m4 | CLOSED | Remaining Introduction/Related Work grammar and clarity issues corrected in commit `a85ccb66...`. |
| m5 | CLOSED | PINN, RMSE, LHS and L-BFGS are expanded before subsequent body use. |
| m6 | CLOSED | Citation/reference spacing is locally consistent. |
| m7 | CLOSED | Captions state supported findings or function without inventing data claims. |

## Evidence gaps and residual risks

1. The exact final PDF has not yet been approved by every author. Approval must bind to PDF SHA-256 `8d31c0068bb7ac6a80bd54287936f85f487c31251a28a1a0904a71a731f896ed`.
2. The reported 7 percent similarity and 0 percent AI-detector results were not independently inspected, and the revised PDF was not rescanned. Only the similarity threshold is part of the supplied submission requirements.
3. CMT coauthor fields, presenter arrangements and the exact deadline cutoff/time zone were not authenticated here.
4. External deanonymization through web matching, exhaustive novelty coverage and complete retraction-database screening were not certified.
5. Figure typography is a known explicit format risk. Embedded fonts and vector format do not cure the size/family failure.
6. Scientific generalization remains limited by one orbit, small seed groups, unmatched comparison configurations and the unavailable historical uncommitted source diff. The manuscript now states these limits appropriately.

No experiment or training run was repeated. This review is valid only for the source commit and PDF hash named above.
