# Revision strategy and task ledger

18 October 2026. Issue: https://github.com/ihmorol/lorenz1960-pinn/issues/39

## Current instructions and acceptance criteria

- Work in writings/iceeict2027-submission only; preserve both prior manuscript folders and supplied ZIPs. Baseline copy committed before edits. No training or new experiments.
- Deliver complete source folder, PDF, readable text export, source diff, before/after figures, build logs, claim-to-evidence map and independent point-by-point audit. Final PDF must be at most SIX A4 pages including everything, with IEEE default dimensions/fonts.
- Preserve simple team voice. Correct specific wording, avoid broad rewrites, and keep contribution positive but limited to demonstrated architecture/training outcomes. Motivation stays technical: training PINNs to approximate this ODE trajectory. No new inverse-problem or general-solver claims.
- New author instruction supersedes embedding-only restriction: try appropriate fonts using recovered SVG/HTML figure source; compare renders. Replace only when appearance and geometry remain acceptable. Missing compact generators may be reconstructed only with exact visual comparison. If not feasible, retain original, report unresolved typography and do not call it compliant. No drastic float/table redesign; reserve layout changes for a genuine six-page blockage.
- Author reports agreement of all authors/supervisors on authorship/order, permission for this revision, no prior submission and no planned parallel submission; LLM-use declarations confirmed. These attestations do not mean the newly generated PDF has already been individually approved.
- Latest correction: author reports 7% similarity and 0% on an AI detector, superseding the earlier 0% similarity statement. Reports were not supplied/independently inspected; no AI score is treated as evidence of authorship or a conference requirement. Screening must not be represented as fresh on the revised text.
- CMT/presenter/exact deadline and final-file approval may be checked later as user requested. Retain those as deferred, not fabricated passes.

## Execution order and ownership

1. Coordinator: baseline/hash preservation, task ledger, main.tex changes, builds and integration.
2. Reference lane: primary-source metadata audit and references.bib only; report related-work wording to coordinator.
3. Figure lane: recovered SVG/HTML source, minimally changed font candidates and visual evidence only. Do not edit main.tex or original figures.
4. Coordinator: concise invariant/test definitions, protocol/seed fixes, bounded positive conclusions, grammar/acronyms/captions; preserve all result tables/equations/algorithm logic.
5. Compile and inspect each integration; resolve page overflow by removing repetition first, never shrinking margins/body fonts.
6. Independent read-only reviewer: final six-page PDF/source versus all 24 requirements, 19 criteria and 15 findings. Coordinator addresses concrete new findings, rebuilds and binds final report to exact source/PDF hashes.
7. Commit focused changes; update issue #39 with actual statuses/evidence and deferred obligations. Package a clean archive and final text/PDF/folder.

## Strategy for every issue task

The following strategies are inherited from issue #39. Latest instructions above resolve P0.2/P0.3 and author-attestation portions of P7; invariants are traced to src/pinn/viz/figures.py:562 and src/pinn/train.py:470. Checkbox status is updated only with evidence. Original baseline tasks are not silently reinterpreted as author approval of the new PDF.

## Phase 0: freeze scope and author decisions

- [x] P0.1 Coordinator: confirm working folder/full SHA, hash ZIPs/PDF and preserve the original. C1 and m1 are resolved in the preview, not in the seven-page copy. Deliver the starting manifest and diff.
- [x] P0.2 Authors: decide whether labels may become 8-point Times New Roman at final printed size, allowing minimum spacing/geometry adjustment only if necessary. Earlier permission is exact geometry and embedding only. D may inspect/propose now; implementation remains gated. Unresolved conflict keeps M7 open.
- [x] P0.3 Authors: supply the actual motivation, research question and defensible contribution; identify any ALREADY EXISTING RK4 or training-time retention evidence. Do not commission runs. Without such evidence, remove or narrow unsupported claims.
- [x] P0.4 All lanes: read both full audits below; record new evidence separately rather than silently replacing the dated baseline.

## Phase 1: reconcile prose with archived evidence

Owner A. Numerical results stay frozen unless an evidenced transcription/calculation error needs an explicitly documented correction.

- [x] P1.1 M1: remove 'and RK4' if no matching archived independent cross-check exists. Acceptance: all reference-solver descriptions agree with actual DOP853 evaluations; legacy RK4 implementation alone is insufficient.
- [x] P1.2 M2: correct 'snapshot every 50 updates' using reported seed-0 snapshot_every=2000. Distinguish logging/checkpoint intervals. Acceptance: exact config field cited; do not generalize one run's setting without checking others.
- [x] P1.3 M4: trace BOTH reported invariant drifts to producing code and saved records; define invariant pair and denominator/normalization. Older scripts/paper_physics_checks.py computes only 16*y[:,0]^2+y[:,1]^2, normalized by its initial absolute value. It does not prove the second metric. Acceptance: both definitions match existing calculations, or flag unsupported text/value for author correction/removal. Never invent a plausible second invariant and describe it as historically measured.
- [x] P1.4 M5: document scripts/ablation_report.py's exact Mann-Whitney call: alternative='greater', method='exact', no-causal seeds 1-3 versus candidate seeds 0-4. Explain group order, larger-error direction, small samples and overlapping-seed independence/pairing limits. Retain p=0.196 only with matching inputs/rounding. Do not claim prespecification without a record; label exploratory if unknown. No favorable-test selection or claims of equivalence from nonsignificance. Keep F1 a separate single-run comparator.
- [x] P1.5 m2: correct blanket seeds 1-3 wording; causal-single uses seed 0 and candidate uses seeds 0-4. Acceptance: table counts and all run records agree.
- [x] P1.6 Check units/counts: reported seed-0 candidate has 34,765 Adam updates, 33,143 L-BFGS closure evaluations and two capped stages. Closure evaluations are not optimizer iterations. A completed high-error run is not an incomplete job; stage-cap attainment alone is not trajectory failure. There is no parameter-matched R5 control in this archive.
- [x] P1.7 Deliver claim-to-evidence mapping: manuscript phrase, artifact/config field, proposed replacement and reason. Acceptance: verified result numbers unchanged; historical source uncertainty disclosed.

## Phase 2: local framing and language corrections

Owner B, dependent on A for factual claims. Preserve the authors' simple vocabulary and sentence structure where correct.

- [x] P2.1 M3: replace necessity claims with final-accuracy comparisons under recorded configurations. Replace 'previous learning is lost or degraded' with observed final performance unless existing time-resolved evidence establishes forgetting. Replace 'passes every check' with the specific verified checks. Acceptance: no causality/generalization beyond the actual comparisons; shared-network control is not asserted parameter-matched or equivalent to bc-PINN.
- [x] P2.2 Clarify the scientific question and contribution in existing Introduction/Abstract space: what this controlled, single-orbit comparison measures and what it cannot establish. Use author-supplied motivation. Do not present established windowing/causal training as a newly invented method or imply untested deployment value.
- [x] P2.3 m3/m4: repair the abstract fragment, subject/verb agreement, tense, articles, spelling and comma splice identified in the audit. Acceptance: local before/after edits with unchanged technical meaning, except separately justified M3 narrowing. Avoid a full stylistic rewrite.
- [x] P2.4 Consolidate short limitations: one orbit/initial state/coefficient set, small seed samples, unmatched settings/controls, final-error limits on forgetting inference and incomplete historical source provenance. Reuse existing prose rather than adding a long new section. Future work is explicitly untested.
- [x] P2.5 m7: propose short specific captions only where they clarify an already-supported finding. Captions are editorial advice, not an invented venue rule; avoid extra paragraphs and unsupported claims.

## Phase 3: reference accuracy

Owner C, related-work coordination with B.

- [x] P3.1 M6: correct reference [12] to Revanth Mattey and Susanta Ghosh using https://arxiv.org/abs/2106.07606 and publisher metadata. Check year/volume/article/DOI alignment; the journal article is 2022, volume 390, article 114474, DOI 10.1016/j.cma.2021.114474.
- [x] P3.2 M6: state that bc-PINN preserves earlier-segment solutions while retraining the same network. Explain the paper's R6 lacks that retention constraint. Acceptance: readers cannot mistake the control for a faithful bc-PINN reproduction.
- [x] P3.3 Verify metadata and supporting claims for every cited entry against primary records, prioritizing the closest Wang/Penwarden/Mattey works. Record checked source/status. Fix errors locally; no citation padding or unsupported exhaustive-novelty claim. Rebuild numbered IEEE bibliography; no uncited/undefined keys.

## Phase 4: figure typography, subject to P0.2

Owner D. Existing fonts are embedded, but printed labels are about 2.67-4.17 points in DejaVu/Segoe families. The supplied template calls for 8-point Times New Roman labels. Embedding alone is insufficient.

- [ ] P4.1 Locate each figure's generator, saved data and exact source revision. Pipeline source exists; compact plot generators may be missing. Identify any reconstruction explicitly; never label reconstructed code as original historical source.
- [x] P4.2 Produce a minimal font-only comparison proposal with final column-width measurements. Preserve curves, values, colors, line styles, panel order and diagram meaning. Any necessary spacing/geometry change requires P0.2 authorization. If constraints cannot coexist, report the conflict rather than silently shrink text or certify compliance.
- [ ] P4.3 After authorization, render vector PDFs from existing data, use the required final-scale label font/size, embed fonts without Type 3 and inspect for overlap/clipping. Acceptance: original/new figure comparison and a per-figure table of generator/data hashes, placed dimensions, minimum printed font size/family and embedding status. No new experimental data.
- [x] P4.4 Send caption proposals to coordinator. Avoid silent redraws or numeric changes; include source and rendered-PDF diffs.

## Phase 5: integrate and rebuild without squeezing typography

Coordinator, after A/B/C and approved D work.

- [x] P5.1 m5: expand PINN and RMSE at first body use even if defined in Abstract; expand LHS at its first body occurrence. Acceptance: reader encounters definitions before subsequent uses.
- [x] P5.2 m6: use nonbreaking citation spacing where appropriate. Keep this a local typesetting improvement, not an asserted desk-rejection rule.
- [x] P5.3 Preserve IEEEtran conference A4 double-column defaults, margins, base font sizes, caption placement, equation numbering and bibliography style. Do not introduce negative spacing, compressed fonts, margin reductions or unreadable figure labels to force six pages.
- [x] P5.4 Build pdflatex, bibtex, pdflatex, pdflatex (or the equivalent project compiler). Inspect complete logs and all pages. Acceptance: six pages including everything, no undefined references/citations, no overfull boxes or clipped content, correct figure/table references and readable equations/tables.
- [x] P5.5 Recheck page count after every integrated substantive pass. Retain the tested roadmap deletion and removed stale trigger. If new accurate definitions/fonts push beyond six pages, first remove duplicated narration locally; preserve essential methods, results and limitations. Any larger content cut needs author decision, not typography compression.
- [x] P5.6 Deliver a concise change log grouped into factual, grammar, bibliography and typography changes, plus full source diff, figure before/after comparison, build log and PDF hash. No changes to raw ZIPs or archived runs.

## Phase 6: independent final audit

Owner E, read-only on the integrated full SHA; coordinator resolves findings and rechecks affected gates.

- [x] P6.1 Reassess all 24 submission rows and 19 scientific criteria below from the actual reviewer-visible PDF. State Pass/Fail/Conditional/Unverified/N/A with evidence; no blanket percentage, acceptance prediction or automatic upgrade to Excellent.
- [x] P6.2 Inspect every page at final printed scale and enlarged view. Check A4, <=6 pages, <=3 MB, all fonts embedded, no Type 3, 8-point Times New Roman figure labels, tables/captions, equations, citations and first-use acronyms.
- [x] P6.3 Recheck anonymous text, author/affiliation/contact identifiers, acknowledgments, self-citation wording, PDF metadata and identifying repository links. External online matching cannot be certified by a local scan. Public code remains published but unlinked from the anonymous PDF.
- [x] P6.4 Reconcile every M/m finding against a patch and evidence. C1/m1 are already fixed in the preview but must remain fixed in the final build. Confirm frozen result values and honest statements of source/reproduction limits.
- [x] P6.5 Record final ratings with reasons and remaining evidence ceilings. Close mechanical errors only with fresh proof. Acknowledged limited novelty or generalization is not disguised as an excellent result.

## Phase 7: author and submission readiness

Authors/coordinator; keep sensitive evidence outside the public issue.

- [ ] P7.1 Confirm contributions/authorship, author order, originality/publication history, no simultaneous submission without required TPC permission and final approval by every author. Record public status/date and a private evidence reference, not signatures or personal details.
- [ ] P7.2 Confirm full LLM-use history matches only formatting/language/proofreading; obtain careful review and approval of the final output and responsibility declaration. The acknowledgment is present but does not prove everyone approved it. Do not fabricate approval text.
- [ ] P7.3 Obtain similarity screening strictly below 30%; check problematic overlaps and citations with authors. Do not chase a score by synonym substitution or remove necessary attribution. No report is currently available.
- [ ] P7.4 Confirm every coauthor's correct-email CMT account privately, submission metadata/title/abstract/track, and a qualified presenter if accepted. Acknowledge no-show exclusion/no-refund policies and conditional Xplore inclusion; these are obligations/outcomes, not current PDF tests.
- [ ] P7.5 Recheck official initial-submission instructions and deadline before upload. Dated audit records 25 October 2026; exact cutoff/time zone is not established. Resolve through official CMT/organizer information without assuming midnight. Any later policy difference must be recorded.
- [ ] P7.6 Freeze the approved source commit and PDF hash; verify the file approved by authors is exactly the upload candidate. Initial-submission checks do not certify later camera-ready/PDF eXpress/copyright obligations. This task stops at a documented readiness verdict; CMT upload is separate.

## Final task outcomes

| Task | Status | Evidence or remaining dependency |
| --- | --- | --- |
| P0.1 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P0.2 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P0.3 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P0.4 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P1.1 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P1.2 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P1.3 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P1.4 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P1.5 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P1.6 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P1.7 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P2.1 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P2.2 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P2.3 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P2.4 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P2.5 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P3.1 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P3.2 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P3.3 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P4.1 | Partial | Exact SVG/HTML pipeline source recovered and pixel-identical; exact compact generators not recovered. Originals retained by author fallback. |
| P4.2 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P4.3 | Retained; requirement unresolved | Times font candidate rendered/tested and rejected; final 8-point font requirement remains unmet. |
| P4.4 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P5.1 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P5.2 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P5.3 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P5.4 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P5.5 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P5.6 | Complete | See claim-to-evidence.md, changes.md and final-verification.json. |
| P6.1 | Complete | Independent final review at a85ccb66; see independent-review.md, exact final PDF hash and all requirement/criterion verdicts. |
| P6.2 | Complete | Independent final review at a85ccb66; see independent-review.md, exact final PDF hash and all requirement/criterion verdicts. |
| P6.3 | Complete | Independent final review at a85ccb66; see independent-review.md, exact final PDF hash and all requirement/criterion verdicts. |
| P6.4 | Complete | Independent final review at a85ccb66; see independent-review.md, exact final PDF hash and all requirement/criterion verdicts. |
| P6.5 | Complete | Independent final review at a85ccb66; see independent-review.md, exact final PDF hash and all requirement/criterion verdicts. |
| P7.1 | Author confirmations recorded; final review pending | Authorship/order, existing approvals and no-submission history confirmed. Newly generated PDF must still be reviewed before upload. |
| P7.2 | History confirmed; final review pending | Non-intellectual LLM use confirmed; accurate acknowledgment retained. Final revised output review is later. |
| P7.3 | Author-reported prior pass | Latest report 7% similarity and 0% AI-detector score. No report inspected or rescreen of this revised PDF. |
| P7.4 | Deferred by author | CMT account/metadata and qualified presenter verification are later steps. |
| P7.5 | Public date checked; exact cutoff deferred | Official date is 25 October 2026; public page does not state exact cutoff/timezone. |
| P7.6 | Deferred final approval | PDF/source can be frozen now, but author approval of this exact regenerated file is still required before upload. |

32 execution tasks are complete. Eight remain partial/deferred, including unlocated compact-plot generators, final figure typography and author/CMT/screening obligations. The authorized local manuscript work is delivered; no new experiments were run. See the independent review for the point-by-point verdict.
