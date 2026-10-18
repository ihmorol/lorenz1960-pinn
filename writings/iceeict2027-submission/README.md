# ICEEICT 2027 manuscript revision

This is a separate revision of the supplied Overleaf manuscript. Previous folders and original ZIPs are preserved. The current build is six A4 pages, uses IEEE conference defaults and contains the same verified numerical results, equations, algorithm and tables.

## Files

- main.tex: editable manuscript source.
- main.pdf: compiled anonymous initial-submission candidate.
- manuscript.txt: complete UTF-8 text extracted from the compiled PDF.
- references.bib, main.bbl, IEEEtran.cls and IEEEtran.bst: bibliography and template files.
- ablation-table.tex and figures/: result table and vector figures.
- figure-sources/: recovered pipeline source, candidate and visual evidence; candidates are not used by main.tex.

Compile from this folder with pdflatex main.tex, bibtex main, then pdflatex main.tex twice. The included class/style and three figure PDFs make the manuscript build independent of the research code and run archives.

The complete strategy, baseline hashes, source diff, primary-reference audit, verification and point-by-point review are in ../../docs/reproducibility/iceeict-revision-20261018/ .

## Remaining submission matters

The figures are preserved by author instruction because the tested font candidate does not meet both visual preservation and final printed-size requirements. Embedded fonts/no Type 3 pass, but the supplied template's 8-point Times New Roman figure-label instruction remains unmet. Do not describe the paper as fully template-compliant.

The submitting author confirmed authorship/order and prior-work approvals, no previous or planned parallel submission, and non-intellectual LLM use. The author reports 7% similarity and a 0% AI-detector result for the screened version; reports were not independently inspected and the revised PDF was not rescanned. Final review of this new PDF, CMT metadata/accounts, presenter arrangements and exact cutoff time remain later author steps. Only the anonymous PDF is intended for initial upload; do not upload this source folder or audit files unless requested by the conference.
