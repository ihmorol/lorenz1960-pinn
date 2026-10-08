# Uploaded manuscript layout review

Reviewed 8 October 2026. Source: C:/Users/User/Downloads/main.tex. The upload was compiled before editing. The revised PDF was rebuilt with BibTeX and all six pages were visually inspected.

## Original build

Six pages. The collocation display at uploaded-source line 294 was 5.5111 pt wider than the column; the error-vector display at line 528 was 10.68002 pt too wide. Both figures shared the label fig:method. Repeated adjacent displays increased congestion.

## Applied changes

Applied the agreed shorter headings. Split the two overflowing displays into aligned rows. Combined adjacent vector-field and coefficient displays; removed the optional diagonal-matrix display because it repeats the explicit nonlinear vector field. Shortened the product-rule and parameter-count displays without changing their results. Retained residual-component expansion, dimensions, point loss, causal weights, window loss, and evaluation measures. Made figure labels unique, fixed the pipeline reference, and changed the baseline figure placement from H to t.

Prose and numerical claims were preserved. No new run data, ablation table, or failure-case prose was added in this layout pass.

## Revised build

Six pages. Final build has no LaTeX errors, overfull-box warnings, undefined citations/references, or duplicate labels. All rendered pages were inspected; equations stay within their columns. IEEE font, margins, and column geometry are unchanged. This is not yet a five-page submission.

## Remaining findings

- The requested prose definition of the window index is still after Equation 3. Physical parameters kappa and ell are assigned values without a visible explanation of their meaning.
- Raw network output is still described as x,y,z, although the trial solution uses it as a correction vector.
- Batch-symbol definitions and the distinction between autograd and the ODE vector field are in percent-sign comments. Those explanations do not appear in the PDF.
- Training Procedure says the minimum uses 1535 points; equations and algorithm use 1536.
- The handoff prose can be read as attributing continuity to weight copying. Continuity follows from the endpoint state and hard trial solution.
- Pipeline artwork uses windows 1 through 27; equations use 0 through 26. This needs an artwork correction.
- Equation 6 fits but occupies three rows for a scalar reduction. Several definitions remain repeated in prose and tables.
- The evaluation-error definition ends a column while Equation 9 begins on the next page. Large pipeline and algorithm floats interrupt surrounding explanations.
- The upload lacks the planned dense residual-MSE equation and visible invariant derivation. Reference drift is reported without identifying the measured quantity there.
- The comparison table lacks optimizer-budget and scheduler rows found in the previously inspected project version. The planned ablation table and failure-case subsections are absent. No results were invented to fill them.
- The discussion says no no-causal windowed run was saved. Reconcile this with completed ablation evidence during integration.
- Announcement-only paragraphs, repeated causal explanations, and Discussion and Limitations remain under the instruction to preserve prose.
- Existing grammar/spelling issues remain, including To train a a, aprroximately, intial, earliar, measuere, and stucking in a platue.

## Five-page requirement

Layout-only compaction has not achieved five pages while preserving all prose, readable figures, and derivation coverage. The earlier agreed deletion of announcement-only paragraphs and removal or relocation of Discussion and Limitations are the next space-saving changes. They were not applied because the latest instruction preserves writing. Further blanket equation deletion would remove useful definitions.
