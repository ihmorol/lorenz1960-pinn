# ICEEICT 2027 manuscript audit

Audit date: 17 October 2026. Verdict: revise before initial submission.

I think the paper has a usable IEEE structure, but its current source is not submission-ready. The immediate barriers are artifact mismatch, anonymity, page size, and prohibited fonts. Several revised sentences also change or obscure the reported experiment.

## Evidence and scope

The manuscript is in `E:/University/FYDP/lorenz1960-pinn/writings/manuscript/conference-paper`, not in the active detached checkout. The supplied template is in `E:/University/FYDP/lorenz1960-pinn/writings/template-mist-ICEEICT`.

I read `main.tex`, the existing `main.pdf`, `references.bib`, `main.bbl`, the local IEEE template and its class documentation. I compiled an isolated copy with pdflatex, BibTeX, and two further pdflatex passes, examined all five rendered pages, and inspected PDF dimensions, fonts, image resolution, and build warnings. No manuscript, figure, bibliography, account, or submission was changed. No scientific experiments were rerun.

Evidence locations in this folder:

- `diagnostic-build/main.pdf`: current source compiled without content changes, not an upload-ready paper.
- `diagnostic-build/main.log`, `build-pass1.txt`, `build-bibtex.txt`, `build-pass2.txt`, `build-pass3.txt`: compilation evidence.
- `current-source-page-1.png` through `current-source-page-5.png`: inspected renderings.
- `existing-pdf-text.txt` and `current-source-pdf-text.txt`: separate text extractions.

Source line references below refer to the original `main.tex` captured for this audit. Page numbers refer to the diagnostic build unless explicitly called the existing PDF.

### Artifact identity

| Artifact | Existing manuscript PDF | Diagnostic build of current source |
| --- | --- | --- |
| Title | Sequential Causal Training of Windowed PINNs for a Lorenz–1960 Orbit | Sequential Causal Training of Windowed PINNs for The Lorenz ODE System |
| Pages | 5 | 5 |
| Bytes | 338,073 | 1,409,305 |
| Page dimensions | 612 × 792 PDF points, US Letter | 612 × 792 PDF points, US Letter |
| Author identity | Visible | Visible, including student IDs |
| Font embedding | All inspected fonts embedded | All inspected fonts embedded |
| Type 3 fonts | None found | Three found |
| Bibliography | Six entries | Seven entries |

The existing PDF's internal creation date is 24 September 2026; the file's copied filesystem date does not establish freshness. Text differences independently prove the mismatch. The source adds a general equation, a new reference, author blocks and different figures, while removing some invariant and optimizer details. Neither artifact should be assumed to supersede the other scientifically without reconciliation.

Original source SHA-256: `d25f44a2ef9663d60a22520b65410faa2745068af064b67a3b7962745913f6b0`.

Original PDF SHA-256: `845af7ca2e399aa5a952279290fe97986a5d6d51e1e13052c23a19f0db8810a9`.

## Conference requirements

Official sources checked live:

- [Initial submission](https://iceeict.mist.ac.bd/authors/paper-submission/)
- [Scope](https://iceeict.mist.ac.bd/authors/call-for-papers/)
- [Dates](https://iceeict.mist.ac.bd/authors/important-dates/)
- [Camera-ready instructions](https://iceeict.mist.ac.bd/authors/camera-ready/)
- [CMT submission route](https://cmt3.research.microsoft.com/ICEEICT2027/Track/1/Submission/Create)

The linked IEEE template website blocked automated retrieval. Its supplied local template and HOWTO were inspected directly. CMT redirected to Login, so private form questions, field limits, and declarations remain unverified.

### Initial submission checklist

The following condenses the public initial-submission requirements; assessment dimensions are mapped separately below.

| Requirement | Status |
| --- | --- |
| IEEE two-column PDF | Pass |
| A4 | Fail: both PDFs are Letter |
| At most six pages, references included | Pass: five |
| At most 3 MB | Pass: both |
| Embedded fonts; no Type 3 | Embedding passes; current source build fails Type 3 ban |
| Double-blind anonymity, including indirect identification and identifying self-citations | Fail: author blocks |
| IEEE typesetting and bibliography | Partial; see template audit |
| LLM use declaration; acknowledgment if applicable | Unresolved; no acknowledgment present |
| Original work; no prior publication or simultaneous submission without TPC approval | Author confirmation needed |
| Similarity below 30% | Unchecked |
| Authorship eligibility, author order, and all authors' submission approval | Author confirmation needed |
| All coauthors have CMT accounts | Account creation reported; individual coverage unverified |
| CMT title, abstract, coauthors, subject areas, paper | Not entered or submitted |
| Presentation by an author or qualified representative if accepted | Future obligation; no-show excludes Xplore submission and refund |

Public initial deadline: 25 October 2026, labeled final extension. Acceptance notification: 5 December; registration/camera-ready: 20 December 2026. The public date page does not specify a submission cutoff timezone. Track 1 includes machine learning and neural networks, making it a reasonable scope fit for this paper. Source: [scope](https://iceeict.mist.ac.bd/authors/call-for-papers/) and [dates](https://iceeict.mist.ac.bd/authors/important-dates/).

## Required fixes and template compliance

Here, “conference requirement” means an explicit public rule. “Template instruction” means guidance in the supplied IEEE files. “Recommendation” means my review judgment, not a newly invented eligibility rule.

### 1. Reconcile the source and PDF before preparing an upload

Evidence: `main.tex:13`, existing PDF page 1, bibliography differences, and the artifact table above.

The existing PDF is not the rendered current source. Its comparatively clean font check therefore does not validate the source you are editing. Regenerate the bibliography and PDF from the agreed source, then check that final artifact. Preserve the existing PDF as a reference while resolving content differences; do not blindly restore its scientific wording.

### 2. Remove identifying material from the initial-review PDF

Conference requirement. Evidence: `main.tex:24–52`, diagnostic page 1, existing PDF page 1.

Both versions show author names and contact information. The revised source also shows student IDs. Remove the visible author block for review. Inspect acknowledgments, figures, hyperlinks, supplementary files and identifying references if subsequently added. PDF author/title metadata are currently empty, but that does not compensate for visible identities. CMT author metadata should still contain the real authors as requested by the submission system.

For a later identified version, the current student blocks omit university/city/country, and the last block includes an honorific and job title. Remove student IDs and academic titles; use names and proper affiliations in the requested order. This is a camera-ready author-format issue, not a reason to retain identity now.

### 3. Select A4 explicitly

Conference requirement. Evidence: `main.tex:1`; all ten inspected pages across both PDFs are 612 × 792 points. A4 is approximately 595.28 × 841.89 points.

The supplied generic template also starts with the default conference class and can produce Letter output. The conference's explicit A4 rule controls. The intended class choice is `\documentclass[conference,a4paper]{IEEEtran}`. Recompile with A4; do not resize a finished Letter PDF or change margins manually. Recheck page count and all figures after reflow.

### 4. Regenerate the trajectory figure without Type 3 fonts

Conference requirement. Evidence: `main.tex:292`, `figures/solution_vs_reference.pdf`, diagnostic page 4.

The figure itself contains `DejaVuSerif-Bold`, `DejaVuSerif-Italic`, and `DejaVuSerif` as Type 3 fonts. Those three fonts are imported into the diagnostic manuscript. All are embedded, but embedding does not make Type 3 acceptable. Regenerate the plot with embedded TrueType/Type 1 fonts and verify both the plot and final manuscript. The existing older manuscript PDF has no Type 3 fonts; that result does not apply to this replacement plot.

### 5. Resolve the LLM declaration truthfully

Conference requirement if applicable. There is no acknowledgment section or LLM statement in the source. Establish the actual use history and make the required declaration. If LLMs were used, acknowledge the actual permitted use and author review. Do not claim that all use was non-intellectual if that does not describe what occurred. The public instructions allow a not-applicable declaration; the private CMT mechanism could not be inspected.

### 6. Repair duplicate figure labels and references

Template instruction and build defect. Evidence: `main.tex:104,136,178`; final log reports multiply-defined `fig:method`.

The baseline image and method pipeline have the same label. Give them unique labels, cite each figure before its placement, and use `Fig.~\ref{...}` consistently with the supplied template. The current resolved reference at line 178 points to Fig. 2, but the duplicate remains ambiguous and fragile. There are no unresolved citations after the diagnostic BibTeX cycle.

### 7. Make every figure readable at its final size

Template instruction: the supplied template specifies 8-point Times New Roman figure labels (`IEEE-conference-template-062824.tex:237–245`).

- Fig. 1 on page 2 is a three-panel image compressed into one column. It is high resolution, approximately 1,279 effective pixels/inch, but its text is very small. Resolution does not solve a physically tiny label.
- Fig. 2 on page 3 is a raster pipeline at approximately 683 effective pixels/inch. Its many labels and equations are too small at the printed size; rasterized font family and point sizes could not be independently measured. Simplify it or use fewer/larger elements.
- Fig. 3 on page 4 has measurable text around 2.49–4.26 points, with tick labels about 3.25 points and a legend about 2.96 points. It also uses DejaVu rather than the requested figure-label font. Redesign for one-column size or use a deliberate two-column figure with suitable layout. Increasing export DPI alone will not fix it.

Keep captions below figures and above tables, which the source already does. Replace generic captions such as “Complete implementation pipeline” with concise explanations of what the reader should notice. The baseline plot should state which configuration and error definition it shows; its displayed MSE and the table's combined RMSE are different metrics and should not be left for readers to infer.

### 8. Use IEEE-compatible algorithm presentation

Template/HOWTO instruction. Evidence: packages at `main.tex:6–7`, algorithm float at `183–214`; supplied `IEEEtran_HOWTO.pdf`, page 11.

The HOWTO warns against the floating environment from `algorithm.sty`, since its caption/presentation does not follow IEEE float handling. Keep pseudocode if useful, but place it in a supported IEEE figure presentation using an IEEE caption. The `algpseudocode` content need not be rewritten merely to match the sample package name.

### 9. Avoid forced mid-column figure placement

Template instruction. Evidence: `main.tex:100` uses `[H]`, producing Fig. 1 midway down page 2. Template lines 207–212 prefer column tops/bottoms. Use normal float placement and an explicit preceding citation. Do not force placement by adding spacing commands.

### 10. Remove mathematical markup from the abstract

Template instruction, distinct from a numeric abstract-length rule. The supplied template at lines 63–64 explicitly warns against math/symbols in titles/abstracts. The present abstract at line 58 contains a time interval in math mode and scientific-notation expressions. State the interval and values in plain text, retaining the actual measurements. It has approximately 191 whitespace-separated source tokens; no conference-specific abstract word limit was found publicly, and any CMT limit is unverified.

The title also needs editorial attention: lowercase “the” after “for,” remove doubled spacing, and consider spelling out avoidable acronyms. I recommend keeping “Lorenz–1960” to identify the system accurately. A possible title is “Sequential Causal Training of Windowed Physics-Informed Networks for a Lorenz–1960 Orbit.” This is a suggested wording, not an imposed conference title.

The stopping criterion is also typeset incorrectly at lines 179, 204 and 244: `min$\_w_{k,i}>0.99$` prints a literal underscore. Use proper minimum notation such as `$\min_i w_{k,i}>0.99$`, after confirming the intended set of points.

### Typography and structure already in place

| Item | Observed status |
| --- | --- |
| IEEEtran class | v1.8b; normalized content matches the supplied class |
| Columns and body text | Two columns; standard Times-family body, nominal 10-point, measured about 9.96 PDF points |
| Title | Nominal 24-point, measured about 23.91 points |
| Abstract/keywords | IEEE environments present |
| Sections | Roman-numbered sections and lettered subsections; unnumbered References |
| Math | Sequential equation numbers; equation/align environments; labels after equation content |
| Figure/table captions | Correct relative position; tables use Roman numbering |
| References | Numeric IEEEtran bibliography, regenerated successfully in diagnostic build |
| Page numbers | No visible running page numbers |
| Manual compression | No margin, column, font-size or spacing override found in the source preamble |
| Final build warnings | Duplicate figure label; two underfull boxes near lines 83–85; no overfull-box warnings found |

Nimbus Roman is the Times-compatible body font supplied by the class/toolchain; it is not itself the font violation. Do not replace the body font or shrink it to accommodate larger figures. The local bibliography style is v1.14 while the downloaded bibliography bundle is v1.12; this version difference alone does not demonstrate noncompliance.

Equation symbols are generally defined, but reserve `k,l` for the physical parameters and another index for windows to avoid overloading `k`. Use one indexing convention for collocation points. Spell out RMSE in the body even though it is expanded in the abstract; expand L-BFGS, LHS and IVP at their first relevant uses. PINN and ODE are expanded in the body. Use consistent capitalization for PyTorch, Xavier initialization, Lorenz and Python.

The final page has noticeably unequal column lengths. Balance it after the content and A4 layout are settled, particularly for camera-ready output; do not insert arbitrary blank space now. If funding exists, handle it using the template mechanism while preserving review anonymity. No funding statement can be inferred from the current source.

The template's unit rules, table-footnote rules, translations, and unpublished/in-press reference labels are conditional instructions. No mixed physical-unit system, table footnotes, translated publication, or claimed in-press item was found that needs those particular repairs. No template filler text was found in the manuscript. The generic template's section titles and example references are not mandatory content.

## Writing and scientific clarity fixes

These are accuracy and reviewer-quality issues. They are not additional hard formatting eligibility rules.

| Priority | Current evidence | Problem and suggested correction |
| --- | --- | --- |
| High | Lines 58, 98, 107, 260, 303; Table III | RMSE 2.237 is attributed to a full-interval network in most places, but line 107 attributes it to one network trained sequentially across 27 segments with forgetting. Identify the actual compared run and describe it consistently. Do not merge different experiments into one narrative. |
| High | Lines 179, 246, 323 | L-BFGS is described in epochs, iterations and closure calls as though they share the same 1,000 count. They do not. The older PDF reports 33,143 total closure calls; treat that as a discrepancy requiring saved-record verification, not an automatically authoritative correction. |
| High | Line 297 | “DOP853 reference drifted less than 4 × 10^-10” has no defined invariant or drift metric in the current source. Restore definitions and paired PINN/reference results if supported, or remove the orphan statement. Do not compare invariant drift with residual MSE. |
| High | Lines 67, 89, 93, 98, 331 | The paper alternates between a closed orbit, a full cycle, a closed loop and a nearly closed orbit. Use the carefully bounded description at line 89 consistently. Identify Lorenz–1960 rather than leaving it open to confusion with Lorenz–1963. |
| High | Lines 93, 161, 174, 303, 325 | “Individual components,” “ablation,” and general statements that residual loss is insufficient imply evidence beyond this unmatched comparison. Present two configurations descriptively. A cited improvement in another study does not establish the causal-weight benefit here. |
| High | Line 325 | Increasing collocation count changes the accumulated residual sum and thus the weights; it does not guarantee improved error or physical fit. Remove the claimed guaranteed decrease/improvement and explain grid dependence precisely. |
| High | Lines 323, 331 | “Passes each of the tests” is undefined and conflicts with the two stages that missed the stopping threshold. Conclude with measured state/residual error and acknowledge the capped stages. |
| Medium | Lines 140, 179, 194–198, 242 | Explain local endpoints and global uniqueness consistently. The stated total is arithmetically compatible with 27 windows × 1,535 cells + 1 = 41,446 unique points. Distinguish 1,536 local time nodes from 1,535 cells and identify exactly which nodes enter each threshold. |
| Medium | Lines 179, 227, 246, 260; Table III | Include the actual baseline optimizer budgets and L-BFGS settings for reproducibility. These are clearer in the older PDF but must be reconciled with the saved run before being restored. Clearly distinguish learning rate from scheduler settings. |
| Medium | Lines 227, 272 | “Distances away from training grid” is unintelligible. A separately constructed evaluation grid need not be entirely disjoint from training points. State the construction accurately. Replace “overlap completely” with “overlap at the plotted scale.” |
| Medium | Table III, line 318 | Hardware/environment and timing protocol are absent. Report times as recorded observations, with that limitation, rather than hardware-normalized speed claims. The older PDF acknowledges the gap. |
| Medium | Line 313 | Clarify whether the parameter count is total stored model capacity or simultaneously optimized parameters. Each window is optimized separately, so “Trainable parameters” can be misunderstood as all 304,641 being jointly updated. |
| Medium | Lines 67, 72, 93, 269, 331 | State a narrow research question and contribution, beyond “implementation and saved results.” Explain what the longer orbit experiment demonstrates and what it leaves unresolved. Do not invent novelty or generalization. |
| Medium | Throughout | A sentence-level language pass is needed. Errors include “adress,” “a a,” “stucking in a platue,” “weigths,” “seperately,” “measuere,” and “they are finding was.” Several errors alter meaning rather than merely polish it. |
| Low | Lines 161–174, 179, 217, 269, 323–325 | Repeated mechanism descriptions and section-opening summaries consume space. Explain causal weighting once; reserve Results for observations and Discussion for interpretation. |

### Practical wording improvements

Preserve numerical values and the team's voice. These examples illustrate local corrections rather than a replacement manuscript:

- “To train a a long horizon problem using traditional PINN is not suggested by previous works” can become “Training a single PINN over a long time interval can be difficult.” Support the narrower claim with the cited work.
- “As it failed by stucking in a platue” can become “Training reached a plateau with substantial trajectory error,” provided that describes the identified baseline.
- “These are described with the word configurations” can become “The comparison is descriptive because the configurations differ in capacity, sampling, loss and optimizer schedule.”
- The conclusion can state the observed RMSE and residual MSE, followed by the single-seed, single-IVP and unmatched-comparison limits. Avoid saying all tests passed.

Suggested contribution framing: an empirical evaluation of sequential causal window training over a nearly closed Lorenz–1960 orbit. The result supports accuracy for the reported configuration and IVP. It does not isolate the effect of causal weighting or establish performance on arbitrary trajectories.

### References

All seven source citation keys exist in `references.bib`. The checked-in `main.bbl` omits the newly cited Matthews/Bihlo item, but the diagnostic BibTeX run produces all seven without missing-citation warnings. Rebuild rather than hand-editing the generated bibliography.

The Matthews/Bihlo paper is real, and its Section 4.2 gives the same general equations, k = 2, l = 1 and initial state. It uses a shorter interval in that example. This attribution is appropriate for the benchmark setup, not evidence for the present longer-orbit results. See [PinnDE, Section 4.2](https://arxiv.org/html/2408.10011v2). Its arXiv record has a 2024 first version and a 2025 revision; cite the version actually used. I did not verify all other publication metadata or every supporting claim externally.

Protect proper-name capitalization in BibTeX, especially `{PinnDE}`, `{Python}` and relevant acronyms. Use normal IEEE reference punctuation. Seven references is not itself a rule failure, and no public minimum was found. Add literature only where it supports a specific gap or comparison; do not pad the bibliography.

## Mapping to the conference's assessment dimensions

| Dimension | Current reviewer-facing assessment | Improvement |
| --- | --- | --- |
| Rationale and scope | Track fit is reasonable; motivation repeats known PINN difficulties | Explain why this longer IVP is a useful test and name the exact question |
| Contribution | Accurate case-study result is plausible; general novelty/component claims are weakly supported | Position as a bounded empirical evaluation |
| Clarity and format | Structure is usable; grammar and figure readability are substantial weaknesses | Correct meaning first, then language and layout |
| Methods | Equations and handoff are explicit; baseline identity and optimizer units conflict | Reconcile experimental descriptions and report reproducible settings |
| Results | State errors and residuals are quantified; comparison changes several factors | Keep grid/metric definitions clear and comparisons descriptive |
| Discussion | Important limits are mentioned; some statements contradict them | Interpret handoff error, derivative discontinuity, threshold misses and configuration confounding |

This is a qualitative audit, not an official review score or an acceptance forecast.

## Optional improvements after the required repairs

The largest value without new experiments is a cleaner research question, readable figures, a compact reproducibility table and consistent claims. Keep the existing mathematical setup, hard initial condition and endpoint handoff explanations.

If time and trustworthy artifacts permit, matched repeated-seed comparisons, a windowed run without causal weighting, and additional initial conditions could strengthen the evidence. These are proposed extensions, not conference mandates or completed tests. Add them only after checking provenance and updating all related text and tables. Do not rush new exploratory results into the locked paper merely to fill a novelty gap.

I would complete the work in this order:

1. Reconcile which source and baseline/results describe the intended submission.
2. Repair the baseline, optimizer, drift, orbit and conclusion inconsistencies.
3. Prepare anonymous A4 output, regenerate plots with compliant fonts, and add the truthful LLM declaration if applicable.
4. Fix labels, pseudocode presentation and small figure text; then proofread locally.
5. Rebuild bibliography/PDF; repeat dimensions, page count, size, fonts and anonymity checks on the exact upload file.
6. Complete similarity and author-approval checks, then validate the authenticated CMT form fields before upload.

## Camera-ready obligations to keep separate

These are acceptance-stage instructions, not defects to fix by adding identifying material to the blind-review PDF now. The [published camera-ready page](https://iceeict.mist.ac.bd/authors/camera-ready/) calls for reviewer revisions, the same A4/six-page/3-MB/font limits, similarity below 30% and below 3% from a single source, and the LLM declaration. It also requires:

- Conference/venue text at the first-page top left and the applicable copyright notice at the bottom. Recheck the live notice when preparing the final paper rather than guessing its year or code.
- PDF eXpress validation, with conference ID currently published as `62016X`.
- IEEE electronic copyright completion through CMT.
- A `PID.zip` containing the validated `PID.pdf`, confirmation-email PDF, revision summary, formatted abstract, and the required presentation materials; the eCF PDF is a separate file.
- The listed oral materials include a seven-minute video and slides; the poster route specifies an electronic poster under 70 MB.

The public page currently specifies `PID_PDF_Express_Email.pdf`, `PID_Revision_Summary.pdf`, `PID_Abstract.pdf`, `PID_Video.mp4`, `PID_Slide.pdf` or `PID_Poster.pdf`, and `PID_Copyright.pdf`. Obtain the actual paper ID before naming final deliverables. Camera-ready deadline: 20 December 2026.

There is a concrete source-quality warning on that page: the link labeled “Step by Step Guideline” opens a 2024 three-minute-thesis event guide, and the sample page is headed 2024 6th ICEEICT with a 2024 copyright notice. These are stale or mislinked downloads. Do not apply that event's team-size, fees or video limits to this paper. Use the current initial-submission rules for this audit, and obtain the corrected camera-ready materials before preparing acceptance-stage deliverables. The abstract-template download was not parsed.

The site still labels technical co-sponsorship “Pending Approval,” and describes Xplore inclusion conditionally. These are current venue-status limits, not manuscript-formatting failures.

## Verification limits

The original source and existing PDF hashes were rechecked unchanged during the audit. The diagnostic build succeeds but retains the identified duplicate-label warning and compliance failures; successful compilation is not submission approval.

Private CMT requirements remain unknown because the linked route required login. No similarity scan, full external citation audit, authorship verification, experimental rerun, PDF eXpress pass or submission was performed. Font/size/page passes apply only to the particular PDFs inspected and must be repeated after changes.
