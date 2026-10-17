# ICEEICT 2027 submission revision plan

## Verdict and objective

Not ready for submission. The six-page preview passes all seven basic PDF checks, but figure typography fails the supplied template, scientific wording conflicts with archived evidence, and author/administrative confirmations remain open. The 19-criterion review scores 18 applicable criteria at 18/54, with one N/A. These are provisional judgments, not official scores, acceptance odds or an official cutoff.

Aim for excellent clarity, presentation and traceability through the smallest defensible edits. Excellent novelty and generalization cannot be promised from this fixed evidence set. State unresolved evidence limits honestly and reassess the final PDF.

This is the INITIAL anonymous submission to MIST ICEEICT 2027. Do not apply camera-ready author blocks, copyright notices or acceptance-stage instructions prematurely. Template examples do not authorize invented funding, affiliations or declarations.

## Locked scope

- No new training, optimizer steps, seeds, architectures, parameter studies, initial conditions or reference-solver benchmark experiments. Historical open run issues do not override this constraint.
- Use only the paper's 16 complete archived runs. Exclude the later stability study with different horizons/initial conditions, including (10,10,10).
- Reading saved artifacts and checking existing calculations is allowed. Any necessary checkpoint evaluation must be identified as verification of an existing model, never a new experiment or historical training reproduction.
- Preserve verified methods, equations, numbers, citations, tables, limitations and team voice. Correct factual inconsistencies with evidence. No wholesale rewriting, invented applications, unsupported causal attribution, fabricated approvals or AI-detector manipulation.
- Preserve both original ZIPs and the seven-page copy. Supply small commits and before/after diffs. This issue does not certify readiness or authorize CMT upload.
- Keep the identifying public repository link out of the anonymous paper. Keep private emails, approvals, account screenshots and full similarity reports out of this public issue.

## Baseline and provenance

Working candidate: writings/iceeict2027-six-page-preview/main.tex and main.pdf. Preserved copy: writings/iceeict2027-overleaf/. Do not edit older writings/conference-paper/ or writings/manuscript/conference-paper-iceeict2027-formatting/ by mistake.

Source baseline: f5938e37efae51804514b6416c193cb76ffffe2e. Preview: six A4 pages, 499,381 bytes, 31 embedded fonts, no Type 3. PDF SHA256: b8a8f55c3eeafb9be3023de487cc6136c2a055324c3fb81997b0e1d0ff79e99a. Its only changes are deletion of two redundant Results roadmap sentences (33 words) and removal of stale IEEEtriggeratref{13}. All remaining source/figures are identical. It is not author-approved.

Paper ZIP SHA256: 9e9534611bffc2a233d688185de0d3b73e5102622a3838f1709e95edd6b3f159. MIST template ZIP SHA256: 35440facca549be108d36a253da4fbf56884a06af7a0d9c429d3a1dac194c6be. Import baseline: 8579d0b07ea768a438a4eb946e2660ae65627281.

Public code: https://github.com/ihmorol/lorenz1960-windowed-pinn, main only, checked snapshot 423696d89fb2021d7b1abceafd92b7fba1dab92a. Prior checks recorded 80 passing tests, 16 CPU prediction checks at 13,265 DOP853 times and 150 run-file hashes. These are prior verification, not fresh retraining. Recovered source ed43703233264b5568ec4598dedc1cf84ac0f452 matches saved non-path settings. Original git_dirty=true has no archived dirty diff; exact historical working-tree reproduction remains uncertified.

## Parallel ownership

Claim a task ID in an issue comment with owner, starting SHA, files/sections, dependency and evidence deliverable. Use separate branches/worktrees. One coordinator integrates main.tex and the PDF; never let agents overwrite a shared checkout. Submit focused patches and recheck after integration. Suggested branches: dev/iceeict-<lane>, explicit target dev/iceeict2027-revision unless coordinator records a replacement.

| Lane | Ownership | Boundary |
| --- | --- | --- |
| Coordinator | Baseline, integration, shared main.tex, builds, release | Serializes shared sections and captions |
| A: evidence | Methods, protocol, Results facts | Supplies verified facts to B |
| B: writing | Abstract, Introduction, Related Work, Discussion/Conclusion | No result-number or BibTeX changes |
| C: references | references.bib, primary metadata | Related Work wording via B |
| D: figures | Sources and approved typography | Decision first; captions via coordinator |
| E: readiness | Policy status, independent final PDF audit | Private records stay private; final review read-only |

Historical context: #34 paper updates, #33 public code and #31 runs. Do not automatically close these issues or reopen their experiment tasks. Each patch references this master issue and task IDs, evidence, changed claims and remaining limits. No force pushes.

## Phase 0: freeze scope and author decisions

- [ ] P0.1 Coordinator: confirm working folder/full SHA, hash ZIPs/PDF and preserve the original. C1 and m1 are resolved in the preview, not in the seven-page copy. Deliver the starting manifest and diff.
- [ ] P0.2 Authors: decide whether labels may become 8-point Times New Roman at final printed size, allowing minimum spacing/geometry adjustment only if necessary. Earlier permission is exact geometry and embedding only. D may inspect/propose now; implementation remains gated. Unresolved conflict keeps M7 open.
- [ ] P0.3 Authors: supply the actual motivation, research question and defensible contribution; identify any ALREADY EXISTING RK4 or training-time retention evidence. Do not commission runs. Without such evidence, remove or narrow unsupported claims.
- [ ] P0.4 All lanes: read both full audits below; record new evidence separately rather than silently replacing the dated baseline.

## Phase 1: reconcile prose with archived evidence

Owner A. Numerical results stay frozen unless an evidenced transcription/calculation error needs an explicitly documented correction.

- [ ] P1.1 M1: remove 'and RK4' if no matching archived independent cross-check exists. Acceptance: all reference-solver descriptions agree with actual DOP853 evaluations; legacy RK4 implementation alone is insufficient.
- [ ] P1.2 M2: correct 'snapshot every 50 updates' using reported seed-0 snapshot_every=2000. Distinguish logging/checkpoint intervals. Acceptance: exact config field cited; do not generalize one run's setting without checking others.
- [ ] P1.3 M4: trace BOTH reported invariant drifts to producing code and saved records; define invariant pair and denominator/normalization. Older scripts/paper_physics_checks.py computes only 16*y[:,0]^2+y[:,1]^2, normalized by its initial absolute value. It does not prove the second metric. Acceptance: both definitions match existing calculations, or flag unsupported text/value for author correction/removal. Never invent a plausible second invariant and describe it as historically measured.
- [ ] P1.4 M5: document scripts/ablation_report.py's exact Mann-Whitney call: alternative='greater', method='exact', no-causal seeds 1-3 versus candidate seeds 0-4. Explain group order, larger-error direction, small samples and overlapping-seed independence/pairing limits. Retain p=0.196 only with matching inputs/rounding. Do not claim prespecification without a record; label exploratory if unknown. No favorable-test selection or claims of equivalence from nonsignificance. Keep F1 a separate single-run comparator.
- [ ] P1.5 m2: correct blanket seeds 1-3 wording; causal-single uses seed 0 and candidate uses seeds 0-4. Acceptance: table counts and all run records agree.
- [ ] P1.6 Check units/counts: reported seed-0 candidate has 34,765 Adam updates, 33,143 L-BFGS closure evaluations and two capped stages. Closure evaluations are not optimizer iterations. A completed high-error run is not an incomplete job; stage-cap attainment alone is not trajectory failure. There is no parameter-matched R5 control in this archive.
- [ ] P1.7 Deliver claim-to-evidence mapping: manuscript phrase, artifact/config field, proposed replacement and reason. Acceptance: verified result numbers unchanged; historical source uncertainty disclosed.

## Phase 2: local framing and language corrections

Owner B, dependent on A for factual claims. Preserve the authors' simple vocabulary and sentence structure where correct.

- [ ] P2.1 M3: replace necessity claims with final-accuracy comparisons under recorded configurations. Replace 'previous learning is lost or degraded' with observed final performance unless existing time-resolved evidence establishes forgetting. Replace 'passes every check' with the specific verified checks. Acceptance: no causality/generalization beyond the actual comparisons; shared-network control is not asserted parameter-matched or equivalent to bc-PINN.
- [ ] P2.2 Clarify the scientific question and contribution in existing Introduction/Abstract space: what this controlled, single-orbit comparison measures and what it cannot establish. Use author-supplied motivation. Do not present established windowing/causal training as a newly invented method or imply untested deployment value.
- [ ] P2.3 m3/m4: repair the abstract fragment, subject/verb agreement, tense, articles, spelling and comma splice identified in the audit. Acceptance: local before/after edits with unchanged technical meaning, except separately justified M3 narrowing. Avoid a full stylistic rewrite.
- [ ] P2.4 Consolidate short limitations: one orbit/initial state/coefficient set, small seed samples, unmatched settings/controls, final-error limits on forgetting inference and incomplete historical source provenance. Reuse existing prose rather than adding a long new section. Future work is explicitly untested.
- [ ] P2.5 m7: propose short specific captions only where they clarify an already-supported finding. Captions are editorial advice, not an invented venue rule; avoid extra paragraphs and unsupported claims.

## Phase 3: reference accuracy

Owner C, related-work coordination with B.

- [ ] P3.1 M6: correct reference [12] to Revanth Mattey and Susanta Ghosh using https://arxiv.org/abs/2106.07606 and publisher metadata. Check year/volume/article/DOI alignment; the journal article is 2022, volume 390, article 114474, DOI 10.1016/j.cma.2021.114474.
- [ ] P3.2 M6: state that bc-PINN preserves earlier-segment solutions while retraining the same network. Explain the paper's R6 lacks that retention constraint. Acceptance: readers cannot mistake the control for a faithful bc-PINN reproduction.
- [ ] P3.3 Verify metadata and supporting claims for every cited entry against primary records, prioritizing the closest Wang/Penwarden/Mattey works. Record checked source/status. Fix errors locally; no citation padding or unsupported exhaustive-novelty claim. Rebuild numbered IEEE bibliography; no uncited/undefined keys.

## Phase 4: figure typography, subject to P0.2

Owner D. Existing fonts are embedded, but printed labels are about 2.67-4.17 points in DejaVu/Segoe families. The supplied template calls for 8-point Times New Roman labels. Embedding alone is insufficient.

- [ ] P4.1 Locate each figure's generator, saved data and exact source revision. Pipeline source exists; compact plot generators may be missing. Identify any reconstruction explicitly; never label reconstructed code as original historical source.
- [ ] P4.2 Produce a minimal font-only comparison proposal with final column-width measurements. Preserve curves, values, colors, line styles, panel order and diagram meaning. Any necessary spacing/geometry change requires P0.2 authorization. If constraints cannot coexist, report the conflict rather than silently shrink text or certify compliance.
- [ ] P4.3 After authorization, render vector PDFs from existing data, use the required final-scale label font/size, embed fonts without Type 3 and inspect for overlap/clipping. Acceptance: original/new figure comparison and a per-figure table of generator/data hashes, placed dimensions, minimum printed font size/family and embedding status. No new experimental data.
- [ ] P4.4 Send caption proposals to coordinator. Avoid silent redraws or numeric changes; include source and rendered-PDF diffs.

## Phase 5: integrate and rebuild without squeezing typography

Coordinator, after A/B/C and approved D work.

- [ ] P5.1 m5: expand PINN and RMSE at first body use even if defined in Abstract; expand LHS at its first body occurrence. Acceptance: reader encounters definitions before subsequent uses.
- [ ] P5.2 m6: use nonbreaking citation spacing where appropriate. Keep this a local typesetting improvement, not an asserted desk-rejection rule.
- [ ] P5.3 Preserve IEEEtran conference A4 double-column defaults, margins, base font sizes, caption placement, equation numbering and bibliography style. Do not introduce negative spacing, compressed fonts, margin reductions or unreadable figure labels to force six pages.
- [ ] P5.4 Build pdflatex, bibtex, pdflatex, pdflatex (or the equivalent project compiler). Inspect complete logs and all pages. Acceptance: six pages including everything, no undefined references/citations, no overfull boxes or clipped content, correct figure/table references and readable equations/tables.
- [ ] P5.5 Recheck page count after every integrated substantive pass. Retain the tested roadmap deletion and removed stale trigger. If new accurate definitions/fonts push beyond six pages, first remove duplicated narration locally; preserve essential methods, results and limitations. Any larger content cut needs author decision, not typography compression.
- [ ] P5.6 Deliver a concise change log grouped into factual, grammar, bibliography and typography changes, plus full source diff, figure before/after comparison, build log and PDF hash. No changes to raw ZIPs or archived runs.

## Phase 6: independent final audit

Owner E, read-only on the integrated full SHA; coordinator resolves findings and rechecks affected gates.

- [ ] P6.1 Reassess all 24 submission rows and 19 scientific criteria below from the actual reviewer-visible PDF. State Pass/Fail/Conditional/Unverified/N/A with evidence; no blanket percentage, acceptance prediction or automatic upgrade to Excellent.
- [ ] P6.2 Inspect every page at final printed scale and enlarged view. Check A4, <=6 pages, <=3 MB, all fonts embedded, no Type 3, 8-point Times New Roman figure labels, tables/captions, equations, citations and first-use acronyms.
- [ ] P6.3 Recheck anonymous text, author/affiliation/contact identifiers, acknowledgments, self-citation wording, PDF metadata and identifying repository links. External online matching cannot be certified by a local scan. Public code remains published but unlinked from the anonymous PDF.
- [ ] P6.4 Reconcile every M/m finding against a patch and evidence. C1/m1 are already fixed in the preview but must remain fixed in the final build. Confirm frozen result values and honest statements of source/reproduction limits.
- [ ] P6.5 Record final ratings with reasons and remaining evidence ceilings. Close mechanical errors only with fresh proof. Acknowledged limited novelty or generalization is not disguised as an excellent result.

## Phase 7: author and submission readiness

Authors/coordinator; keep sensitive evidence outside the public issue.

- [ ] P7.1 Confirm contributions/authorship, author order, originality/publication history, no simultaneous submission without required TPC permission and final approval by every author. Record public status/date and a private evidence reference, not signatures or personal details.
- [ ] P7.2 Confirm full LLM-use history matches only formatting/language/proofreading; obtain careful review and approval of the final output and responsibility declaration. The acknowledgment is present but does not prove everyone approved it. Do not fabricate approval text.
- [ ] P7.3 Obtain similarity screening strictly below 30%; check problematic overlaps and citations with authors. Do not chase a score by synonym substitution or remove necessary attribution. No report is currently available.
- [ ] P7.4 Confirm every coauthor's correct-email CMT account privately, submission metadata/title/abstract/track, and a qualified presenter if accepted. Acknowledge no-show exclusion/no-refund policies and conditional Xplore inclusion; these are obligations/outcomes, not current PDF tests.
- [ ] P7.5 Recheck official initial-submission instructions and deadline before upload. Dated audit records 25 October 2026; exact cutoff/time zone is not established. Resolve through official CMT/organizer information without assuming midnight. Any later policy difference must be recorded.
- [ ] P7.6 Freeze the approved source commit and PDF hash; verify the file approved by authors is exactly the upload candidate. Initial-submission checks do not certify later camera-ready/PDF eXpress/copyright obligations. This task stops at a documented readiness verdict; CMT upload is separate.

Proposed internal sequence, not a verified conference schedule: 18 October baseline/decisions; 19-20 parallel revisions; 21 integration; 22 independent audit; 23 author review/similarity; 24 buffer. The 25 October cutoff must be confirmed. Do not delay an independent lane while another awaits a decision.

## Execution mapping for all 24 submission checks

Each checkbox means the specified verification/action has been completed, not that a future outcome is guaranteed. Historical statuses below describe the audited preview.

| ID | Requirement | Baseline status | Action and evidence owner |
| --- | --- | --- | --- |
| S01 | IEEE double-column template | Pass | P5/P6, Coordinator/E: preserve conference class and inspect final two columns. |
| S02 | PDF format | Pass | P5/P6: open final rebuilt PDF and record hash. |
| S03 | File size at most 3 MB | Pass | P6: measure final byte count <=3 MB. |
| S04 | A4 paper | Pass | P6: inspect dimensions of every page. |
| S05 | Maximum six pages including everything | Pass | P5/P6: count all pages including references after integration. |
| S06 | No Type 3 fonts | Pass | P4/P6: inspect font types in the final PDF, not source settings alone. |
| S07 | All fonts embedded | Pass | P4/P6: verify every final PDF font is embedded. |
| S08 | Every author meets authorship criteria | Unverified | P7.1, Authors: confirm contributions privately; record status/date. |
| S09 | Strict IEEE typesetting and bibliography | Fail | P3-P6: fix figure typography, metadata and acronym defects, then inspect full style. |
| S10 | Agreement on author sequence | Unverified | P7.1: obtain collective order agreement privately. |
| S11 | All authors approve submission | Unverified | P7.1/P7.6: obtain every author approval of the exact final PDF hash. |
| S12 | Original contribution; not previously published or submitted | Unverified | P7.1: attest originality and complete prior submission/publication history; resolve conflicts. |
| S13 | No simultaneous submission without TPC permission | Unverified | P7.1: attest no simultaneous submission or document required TPC permission privately. |
| S14 | Similarity strictly below 30% | Unverified | P7.3: obtain report <30%, review overlaps; status currently unverified. |
| S15 | LLM use acknowledged | Pass | P7.2/P6: retain accurate acknowledgment and check final wording. |
| S16 | LLM use solely non-intellectual | Conditional | P7.2: confirm complete history, resolve any discrepancy before declaring compliance. |
| S17 | LLM output carefully reviewed and approved | Unverified | P7.2: authors carefully review and approve final LLM-assisted output. |
| S18 | Authors accept accuracy/content responsibility | Pass | P1/P7.2: correct inaccuracies; retain accurate responsibility declaration. |
| S19 | Double blind: names, affiliations, addresses, emails, self-citation, other identifiers | Pass on inspection | P6.3: inspect final text/metadata and identifying links; no claim of online untraceability. |
| S20 | Qualified presenter if accepted | Future obligation | P7.4: identify a qualified presenter privately if accepted. |
| S21 | No-show papers excluded from Xplore | Policy consequence | P7.4: acknowledge attendance/no-show exclusion consequence, not a file-test pass. |
| S22 | No-show fees not refunded | Policy consequence | P7.4: acknowledge no-show non-refund consequence, not a manuscript defect. |
| S23 | Every coauthor has correct-email CMT account | Conditional | P7.4: privately verify each account/email and CMT metadata without publishing personal data. |
| S24 | Xplore inclusion subject to scope and quality | Conditional outcome | P7.4: record conditional scope/quality outcome; never promise Xplore indexing. |

## Improvement strategy for all 19 scientific criteria

Targets are conditional editorial goals, not achieved ratings. Re-score only after reviewing the final PDF. Excellent reporting can coexist with limited underlying novelty, design or generalization. No target authorizes new experiments.

| ID | Criterion | Current | Feasible target/limit | Execution and acceptance |
| --- | --- | --- | --- | --- |
| Q01 | Rationale: Motivation and grounding | 1 | 2-3 conditional | P0.3/P2: real author motivation, precise scope and supported gap; no invented application. |
| Q02 | Rationale: Venue scope | 2 | 3 target | P2/P7: make neural-network training relevance explicit and verify actual CMT track. |
| Q03 | Contribution: Importance of question | 1 | 1-2 evidence ceiling | P2: explain the bounded comparative question; practical importance needs evidence beyond prose. |
| Q04 | Contribution: Innovation | 0 | 1-2 ceiling, no promised 3 | P2/P3: distinguish empirical comparison from known components; no new mechanism/theory is established. |
| Q05 | Contribution: Potential disciplinary advance | 1 | 1-2 evidence ceiling | P2: identify what the fixed single-orbit evidence adds; broader advance remains untested. |
| Q06 | Clarity: Writing clarity | 0 | 3 target | P1/P2/P5: accurate solver/protocol/claims, grammatical local edits and author voice review. |
| Q07 | Clarity: Organization and flow | 2 | 3 target | P2/P5: preserve useful section order, remove repetition, connect question to findings concisely. |
| Q08 | Clarity: Reference appropriateness/completeness | 1 | 3 conditional | P3: correct all metadata and source-supported closest-work distinctions; not exhaustive novelty certification. |
| Q09 | Clarity: IEEE organization/typesetting | 1 | 3 conditional on P0.2 | P4-P6: meet actual figure typography and all class/acronym/citation requirements. |
| Q10 | Methods: Strategy and design | 2 | 2-3 reporting target | P1/P2: explain existing design and control differences. Missing matched controls remain a design limit. |
| Q11 | Methods: Information for reproducibility | 1 | 2-3 reporting target | P1/P5: exact metrics, invariant definitions, settings and seeds in reviewer-visible prose; dirty-source limit remains. |
| Q12 | Methods: Implementation rigor/reporting | 1 | 2-3 reporting target | P1/P2: accurately report complete runs, closure counts and capped stages; do not certify missing source history. |
| Q13 | Methods: Appropriateness of analysis | 1 | 2-3 reporting target | P1/P2: exact directional test, sample/seed caveats and bounded inference; statistical power cannot be fixed by writing. |
| Q14 | Results: Collection/analysis/reporting | 1 | 3 reporting target | P1/P5: trace every saved result and reconcile solver/protocol wording without changing verified numbers. |
| Q15 | Results: Reasoning and concision | 1 | 3 target | P2/P5: observations support claims, no necessity/forgetting overclaim, no redundant roadmap. |
| Q16 | Results: Tables and figures | 0 | 3 conditional on P0.2 | P4/P5: readable compliant labels, faithful graphics and correct compact tables/captions. |
| Q17 | Results: Supplemental material | N/A | N/A remains valid | P6/P7: no supplement is incorporated in the anonymous PDF. Do not add one solely for a score or link identifying code. |
| Q18 | Discussion: Interpretation in context | 1 | 2-3 reporting target | P2/P3: compare with closest work accurately and distinguish measured outcomes from proposed explanations. |
| Q19 | Discussion: Limitations and future work | 1 | 3 reporting target | P2: concise explicit limitations and clearly untested future work; do not conceal evidence ceilings. |

## Finding closure ledger

These IDs retain the original audits' severity. Each task owner supplies a patch/evidence reference; coordinator/E verify the final PDF, not merely a promise.

| Finding | Baseline state | Execution | Required closure proof |
| --- | --- | --- | --- |
| C1: seven-page original | Resolved in six-page preview only | P0.1/P5.4/P6.2 | Final integrated PDF <=6 pages without typography compression |
| M1: DOP853/RK4 mismatch | Open | P1.1 | Solver wording matches saved evaluation |
| M2: snapshot 50/2000 mismatch | Open | P1.2 | Config field and local text diff |
| M3: necessity/forgetting/universal-check claims | Open | P2.1 | All affected Abstract/body/Conclusion claims bounded by evidence |
| M4: undefined invariant pair/normalization | Open | P1.3 | Both actual producing calculations identified or unsupported claim corrected |
| M5: underspecified statistical test | Open | P1.4 | Exact call, groups, direction, sample/seed caveats and matching value |
| M6: Mattey metadata and bc-PINN distinction | Open | P3.1/P3.2 | Primary-record metadata and accurate comparison text |
| M7: figure font family/printed size | Open, author decision required | P0.2/P4/P6.2 | Explicit permission, final-scale measurements and visual verification |
| m1: redundant Results roadmap | Resolved in preview | P0.1/P5.5 | Tested 33-word deletion retained |
| m2: seed exceptions | Open | P1.5 | Prose agrees with all recorded seed groups |
| m3: Abstract grammar | Open | P2.3 | Minimal grammatical diff with factual meaning preserved |
| m4: Introduction grammar | Open | P2.3 | Minimal grammatical diff with team voice preserved |
| m5: first-use acronyms | Open | P5.1 | Body definitions precede reuse |
| m6: citation spacing | Open, editorial | P5.2 | Appropriate local nonbreaking spacing without layout damage |
| m7: captions | Open, editorial | P2.5/P4.4 | Concise supported caption or documented reason to retain existing text |

## Completion and stopping conditions

- [ ] Every task has an owner/status and links to its evidence or an explicit unresolved decision. Completed tasks need final integrated-SHA verification; superseded tasks need a reason.
- [ ] All seven major findings have been resolved or are explicitly blocking. The known figure requirement cannot be waved through under an incompatible author constraint.
- [ ] All 24 submission checks are reassessed individually. Future obligations and conditional publication outcomes retain their correct category. Required confirmations cannot be marked Pass without the responsible authors' evidence.
- [ ] All 19 scientific criteria receive an updated reasoned verdict. Report residual novelty/design/generalization limits even if editorial work is excellent. Do not manufacture a target aggregate score.
- [ ] Final deliverables: source commit/full diff, compiled anonymous PDF/hash, build log, figure typography/visual evidence, claim-to-artifact mapping, updated point-by-point audit and private-approval status references.
- [ ] No experiment reruns, silently changed results, leaked author identifiers in the anonymous PDF, unexplained broad rewrites or overwritten original work.
- [ ] Coordinator records Go/No-go for initial submission. Go requires resolved mandatory defects, required author/policy evidence and author acceptance of remaining scientific limits. Otherwise state exactly what blocks submission. Closing this execution issue means the agreed work and verdict are documented, not that acceptance/indexing is assured.

Official source pages recorded by the dated audit: https://iceeict.mist.ac.bd/authors/paper-submission/ ; https://iceeict.mist.ac.bd/authors/call-for-papers/ ; https://iceeict.mist.ac.bd/authors/important-dates/ . Recheck in P7.5; the following full audits are historical snapshots and must not be silently rewritten to imply later verification.
