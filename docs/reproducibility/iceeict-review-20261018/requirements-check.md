# Strict ICEEICT 2027 requirements audit

18 October 2026. Scope: the separate six-page preview; the preserved seven-page paper still fails length.
PDF SHA256: b8a8f55c3eeafb9be3023de487cc6136c2a055324c3fb81997b0e1d0ff79e99a

Verdict: not ready to submit. All seven basic file checks pass, but broad IEEE compliance fails and author/administrative declarations remain unverified. No overall compliance percentage is claimed: the requirements are not equally weighted. This follows the preceding pre-submission-reviewer skill review and directly rechecks the same final PDF against the pasted text.

## Every submission point

| Requirement | Status | Evidence or missing action |
| --- | --- | --- |
| IEEE double-column template | Pass | IEEEtran conference class and default dimensions. |
| PDF format | Pass | main.pdf opens correctly. |
| File size at most 3 MB | Pass | 499,381 bytes. |
| A4 paper | Pass | All pages A4. |
| Maximum six pages including everything | Pass | Six including tables, figures and references. |
| No Type 3 fonts | Pass | None among 31 fonts. |
| All fonts embedded | Pass | All 31 embedded. |
| Every author meets authorship criteria | Unverified | Requires contribution confirmation. |
| Strict IEEE typesetting and bibliography | Fail | Figure fonts below 8pt/not Times New Roman; acronym omissions; reference [12] wrong author. |
| Agreement on author sequence | Unverified | Collective approval not recorded. |
| All authors approve submission | Unverified | Final six-page version not approved by everyone. |
| Original contribution; not previously published or submitted | Unverified | Originality and submission history require author confirmation. |
| No simultaneous submission without TPC permission | Unverified | Author commitment required. |
| Similarity strictly below 30% | Unverified | No similarity report. |
| LLM use acknowledged | Pass | Explicit acknowledgment present. |
| LLM use solely non-intellectual | Conditional | Declaration matches your reported formatting/language/proofreading use; full history requires confirmation. |
| LLM output carefully reviewed and approved | Unverified | Final author approval not established; paper does not claim it. |
| Authors accept accuracy/content responsibility | Pass | Explicit responsibility statement; errors still need correction. |
| Double blind: names, affiliations, addresses, emails, self-citation, other identifiers | Pass on inspection | PDF text/metadata anonymous; no repository link. External online matching not certified. |
| Qualified presenter if accepted | Future obligation | Presenter not confirmed. |
| No-show papers excluded from Xplore | Policy consequence | Not a current manuscript pass/fail test. |
| No-show fees not refunded | Policy consequence | Not a current manuscript pass/fail test. |
| Every coauthor has correct-email CMT account | Conditional | You report accounts created; every account/email not inspected. |
| Xplore inclusion subject to scope and quality | Conditional outcome | Publication is not guaranteed by acceptance or this audit. |

## Every scientific assessment point

My provisional judgments, not official scores or acceptance probabilities. Scale: -1 Very Poor, 0 Poor, 1 Average, 2 Good, 3 Excellent. Reviewer-visible paper content determines these ratings; repository verification does not replace missing explanations.

| Category | Criterion | Score | Reason |
| --- | --- | ---: | --- |
| Rationale | Motivation and grounding | 1 | Relevant citations; practical need for this single-orbit study is weak. |
| Rationale | Venue scope | 2 | Track 1 covers machine learning and neural networks. |
| Contribution | Importance of question | 1 | Useful bounded training study; limited practical reach. |
| Contribution | Innovation | 0 | Established components; no clearly established new mechanism or theory. |
| Contribution | Potential disciplinary advance | 1 | One orbit, initial state and coefficient set. |
| Clarity | Writing clarity | 0 | Grammar errors; misleading RK4, logging and necessity statements. |
| Clarity | Organization and flow | 2 | Logical problem, method, setup, results and conclusion sequence. |
| Clarity | Reference appropriateness/completeness | 1 | Closest works cited; author error and metadata/detail corrections remain. |
| Clarity | IEEE organization/typesetting | 1 | Basic file checks pass; figure typography and acronym omissions remain. |
| Methods | Strategy and design | 2 | Hard ICs, per-window models and staged causal loss are well specified. |
| Methods | Information for reproducibility | 1 | Invariant definitions, logging and seed exceptions need correction. |
| Methods | Implementation rigor/reporting | 1 | Completed runs verified; control, scope and source-provenance limits need care. |
| Methods | Appropriateness of analysis | 1 | Small samples, test direction and forgetting attribution need care. |
| Results | Collection/analysis/reporting | 1 | Saved results support numbers; solver/protocol descriptions conflict. |
| Results | Reasoning and concision | 1 | Main findings clear; repetition and overclaims remain. |
| Results | Tables and figures | 0 | Useful tables; tiny figure labels impair reading. |
| Results | Supplemental material | N/A | None incorporated in anonymous PDF; named public repository is not linked. |
| Discussion | Interpretation in context | 1 | Prior methods discussed; necessity/forgetting conclusions too strong. |
| Discussion | Limitations and future work | 1 | One-state scope, small seeds and unmatched configurations acknowledged but scattered. |

All 19 reviewed; 18 scored, one not applicable. Total 18/54; mean 1.0 (Average). Distribution: 3 Good, 12 Average, 3 Poor. The published page gives no numerical acceptance cutoff. This is not a 33% acceptance estimate.

## Fix order

1. Fix printed figure fonts and first-use acronyms.
2. Correct DOP853/RK4 and snapshot 50/2000 statements; identify seed exceptions.
3. Correct reference [12] to Revanth Mattey and distinguish bc-PINN accurately.
4. Define invariants/drift normalization and the exact one-sided statistical test.
5. Bound necessity, forgetting and "passes every check" claims; correct grammar.
6. Rebuild after edits, then obtain similarity screening and author/order/LLM approvals; verify CMT accounts and confirm a presenter.

Sources: pasted requirements, supplied MIST template ZIP, current PDF/TeX, archived seed-0 configuration and the prior verified result audit.
Official requirements: https://iceeict.mist.ac.bd/authors/paper-submission/
Official scope: https://iceeict.mist.ac.bd/authors/call-for-papers/
Reference [12] primary source: https://arxiv.org/abs/2106.07606

Not performed: manuscript edits, new experiments, similarity screening, exhaustive novelty/reference searches, account inspection or final author approval. No acceptance guarantee is inferred from format compliance or repository tests.
