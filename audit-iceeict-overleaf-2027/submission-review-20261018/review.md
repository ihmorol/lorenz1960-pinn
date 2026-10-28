# MIST ICEEICT 2027 pre-submission review

Reviewed 18 October 2026 with the pre-submission-reviewer skill. This review concerns the final PDF for initial anonymous, double-blind submission, before acceptance. The existing manuscript is preserved. A separate six-page preview contains only the small changes below.

## Summary

- Current manuscript: CRITICAL 1, MAJOR 7, MINOR 7.
- Separate preview: CRITICAL 0, MAJOR 7, MINOR 6.
- First priorities: select the six-page layout, resolve the figure-font conflict, and correct the reference-solver/training-record statements.
- Recommendation: needs revision before submission. Page length alone does not establish full compliance.

## Smallest tested space change

Delete the two sentences immediately after the Results heading, original main.tex line 349:

> In this section we report the accuracy and physics checks of the windowed run against DOP853. Then we compare it with the full-interval run and we also show the ablation and failure results.

These 33 words repeat the subsection headings. Also remove the single line `\IEEEtriggeratref{13}` near the bibliography. Its default action is a column/page break, and the hard-coded reference number is stale after the deletion. IEEEtran then flows the references normally.

The tested preview is six A4 pages, 499,381 bytes. The exact source diff contains only these two changes. All other source files, including figures, bibliography, class and ablation table, are byte-identical. Equations, pseudocode, citations, tables and numerical results are unchanged. No margin, font-size or line-spacing reduction was used. The last-page column bottoms differ by approximately 10 points.

Removing only the manual break still gives seven pages. Deleting only the method roadmap also remains seven pages. Deleting both Results roadmap sentences and removing the break gives six. The nine-sentence alternative also fits, but removes more text than necessary. Later substantive fixes require another build because page count can change.

See `recommended.diff`, `verification.json`, and `six-page-contact-sheet.png`. The preview is `writings/iceeict2027-six-page-preview/main.pdf`.

## Both supplied ZIPs

Checked directly, without changing either archive:

- `C:/Users/User/Downloads/conference-paper-overleaf.zip`, the authoritative paper.
- `E:/University/FYDP/lorenz1960-pinn/writings/template-mist-ICEEICT.zip`, the supplied template and bibliography styles.

The paper's IEEEtran.cls matches both archives after line-ending normalization. Its standard IEEEtran.bst is version 1.14; the template includes version 1.12. This is a version difference, not evidence of a different bibliography style. The output uses numbered, unsorted IEEE references. Sample author blocks and funding text in the template are reference material, not instructions to disclose identities or invent funding.

## Venue and template requirements

The live initial-submission requirements are at https://iceeict.mist.ac.bd/authors/paper-submission/.

| Requirement | Existing manuscript | Separate preview |
| --- | --- | --- |
| IEEE conference class, two columns, default margins | Pass | Pass |
| A4 | Pass | Pass |
| At most six pages including figures, tables and references | FAIL: seven | Pass: six |
| PDF at most 3 MB | Pass: 500,269 bytes | Pass: 499,381 bytes |
| All fonts embedded; no Type 3 | Pass | Pass |
| Anonymous PDF text/metadata; named repository link absent | Pass on inspection | Pass on inspection |
| Figure captions below; table captions above | Pass | Pass |
| Resolved references/citations; no overfull boxes | Pass | Pass |
| LLM non-intellectual-use acknowledgment | Present | Present |
| Figure labels: template specifies 8-point Times New Roman | FAIL | FAIL, unchanged |
| Acronyms expanded again at first body use | Minor omissions | Same omissions |
| Final author approval and author-reviewed LLM output | Requires author confirmation | Same |
| Similarity below 30%, originality/no simultaneous submission | Not independently certified | Same |
| Authenticated CMT fields and all coauthor accounts | Not inspected; user reports accounts created | Same |

The supplied template states: "Use 8 point Times New Roman for Figure labels." Approximate minimum text sizes at column-width scaling are 4.17 pt in the pipeline (Segoe UI), 2.67 pt in the full-interval plot (DejaVu Serif), and 3.84 pt in the trajectory plot (DejaVu Serif). Embedding resolves portability; it does not resolve the size/family requirement. These estimates use a 252-point column width, within 0.1% of the actual class width.

The earlier instruction to preserve exact geometry and repair embedding only is honored. Full template compliance cannot be certified while that conflict remains. The official deadline is 25 October 2026; the public page gives no exact cutoff time: https://iceeict.mist.ac.bd/authors/important-dates/.

## Dimension 1: Macro logic

This is a bounded STEM numerical study of one orbit and initial state. Its components and experiments are identifiable. Generic dataset quotas and fixed paragraph counts from the skill are not ICEEICT rules and were not imposed.

| ID | Specific finding | Severity | Suggested fix |
| --- | --- | --- | --- |
| M1 | Prediction Accuracy says "SciPy's DOP853 and RK4"; Reference Solution and saved evaluations use DOP853. | MAJOR | Remove "and RK4" unless a matching independent comparison record is supplied. |
| M2 | Training Procedure says "we save a snapshot every 50 updates"; the reported seed-0 config records snapshot_every=2000. | MAJOR | State the recorded snapshot interval and distinguish other logging intervals. |
| M3 | "separate window networks is needed", "previous learning is lost or degraded", and "passes every check" exceed what final errors alone establish. | MAJOR | Bound conclusions to final accuracy under recorded configurations; define checks; reserve forgetting claims for training-time evidence. |
| M4 | "two independent invariants" and "maximum relative drift" do not identify the pair or denominator. | MAJOR | Give both invariant forms and their normalization. |
| M5 | "Mann-Whitney p=0.196" omits the exact one-sided alternative and justification for the seed-set comparison. | MAJOR | Specify test direction, exact calculation and compared seeds; retain the small-sample caveat. |
| M6 | Reference [12] identifies "R. K. Mattey"; the primary paper identifies Revanth Mattey. The related-work description omits bc-PINN's preservation of previous-segment solutions. | MAJOR | Correct the BibTeX author and distinguish the shared-network control from backward-compatible training. |

The focused novelty/metadata retrieval found the three closest works already cited: [Wang et al.](https://www.sciencedirect.com/science/article/pii/S0045782524000690), [Penwarden et al.](https://arxiv.org/abs/2302.14227), and [Mattey and Ghosh](https://arxiv.org/abs/2106.07606). This is not an exhaustive novelty certification. The last source directly supports the author correction and previous-segment preservation detail.

## Dimension 2: Writing details

| ID | Specific finding | Severity | Suggested fix |
| --- | --- | --- | --- |
| m1 | Results begins "In this section we report... Then we compare..." | MINOR | Delete the two roadmap sentences as tested. Resolved in the preview. |
| m2 | "For the ablations we used seeds 1-3" omits causal-single seed 0 and candidate seed 4. | MINOR | Identify exceptions explicitly; preserve the table's correct counts. |

## Dimension 3: English grammar

| ID | Specific finding | Severity | Suggested fix |
| --- | --- | --- | --- |
| m3 | Abstract: "Also a dense-grid residual mean square...", "networks is needed", "improves accuracy and reduced training steps". | MINOR | Repair the fragment; apply G2 subject-verb agreement and G3 tense consistency. The strength of "needed" is covered by M3. |
| m4 | Introduction: "problem like this", "usefull", "use cases still needs", and "system , its reference solution...". | MINOR | Apply G1 article/plural form, spelling, G2 agreement, and G4 clause punctuation. |

Rule IDs refer to the skill's `references/grammar-rules.md`. These corrections were not applied to the page-saving preview.

## Dimension 4: LaTeX format

| ID | Specific finding | Severity | Suggested fix |
| --- | --- | --- | --- |
| C1 | Existing PDF has seven pages; the venue allows six including references. | CRITICAL in existing version | Use the tested deletion and remove the stale bibliography trigger. Resolved in preview. |
| m5 | First body uses "A PINN" and contribution "RMSE" rely on abstract definitions; "LHS points" precedes its expansion in a later table footnote. | MINOR | Expand PINN/RMSE again at first body use and LHS at first use, as the template requests. Rebuild afterward. |
| m6 | Some citations use a normal space, for example `coefficients \cite{raissi2019}`. | MINOR | Use a nonbreaking tilde. This is typesetting advice, not an explicit venue desk-rejection rule. |

All nine equation references resolve. Captions and cross-references exist, and the preview log has no undefined references/citations or overfull boxes. The class prescribes text dimensions and sizes; no typography overrides were introduced.

## Dimension 5: Figure quality

| ID | Specific finding | Severity | Suggested fix |
| --- | --- | --- | --- |
| M7 | Three PDFs retain approximately 2.7-4.2-point minimum text in DejaVu/Segoe families, against the 8-point Times New Roman instruction. | MAJOR | Address final-scale label size/family in a separate figure pass. Embedding alone is insufficient; retain geometry unless the earlier constraint is changed. |
| m7 | Captions such as "Single network on the full interval" and "Implementation pipeline" name the object without its principal finding. | MINOR | Consider a short outcome statement. This is reviewer clarity advice, not a venue caption-content rule. |

The diagrams/plots remain vector PDFs. Labels, line styles and separate panels supplement colors. Pages 3, 5 and 6 were inspected at larger scale, and every page was checked in the contact sheet. No clipping was observed. Tiny figure labels remain difficult to read.

## Banned-vocabulary and em-dash scan

The entire active main.tex and ablation-table.tex were scanned, excluding comments and external bibliography titles. No listed terms or sentence-connector em dashes were found. `vocabulary-scan.json` records scope. This is editorial advice, not an ICEEICT requirement or an authorship test.

## Score and submission recommendation

Existing manuscript: 5/10. Separate six-page preview: 6/10. Needs revision before submission. Seven substantive/template findings remain after the page fix. Do not describe the preview as fully compliant or author-approved.

Not performed: full experiment reruns, similarity/originality screening, authenticated CMT checks, final author approval or exhaustive reference verification. Saved-run evidence was reused; the reported snapshot interval was checked against the actual seed-0 configuration.
