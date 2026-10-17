# ICEEICT manuscript draft

The author-designated Overleaf ZIP was imported without edits in commit `8579d0b07ea768a438a4eb946e2660ae65627281`. This folder contains the conference formatting revision. Scientific prose, measured values, pseudocode and numbered equations are retained.

## Build

From this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
```

`main.pdf` is an anonymous, seven-page A4 draft. It is below 3 MB, has embedded fonts and no Type 3 fonts, and compiles without warnings. It exceeds the six-page conference limit and is not ready for submission. The user requested all prose and exact figure geometry to be retained, so page reduction and small printed figure labels remain pending.

The figure repair changes only font resources. The original page streams, text positions, axes, curves and page boxes are preserved; the pipeline PDF is byte-identical. `../../scripts/embed_figure_fonts.py` implements the repair. `../../scripts/pipeline_figure.py` recovers the original standalone pipeline HTML/SVG scene. Exact generators for the compact numerical PDFs were not saved, so those figures are retained as PDF sources with reproducible font-resource repair.

See `../../docs/reproducibility/` for run provenance, checksums, manuscript findings and verification. Compare this folder against import commit 8579d0b for the formatting diff. Final author approval, similarity screening and CMT checks remain pending. Do not link the named public reproduction repository from the anonymous paper.
