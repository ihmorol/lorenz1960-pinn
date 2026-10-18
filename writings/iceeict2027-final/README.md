# ICEEICT 2027 final manuscript

main.pdf is the anonymous initial-submission PDF. main.tex is the editable source; manuscript.txt is its complete text export. Previous manuscript folders remain preserved.

The three scientific figures are regenerated as vector PDFs with readable Times New Roman labels. Numeric plots use deterministic inference from the archived final checkpoints and the locked DOP853 reference; the included NPZ/CSV data and generator reproduce the plotted figures without training or checkpoint access. The detailed pipeline restores the original nine-stage flow, network subblocks, decision diamonds and stage/window return paths. See figure-sources/ for generators, SVG, provenance and data checks.

Build from this folder: pdflatex main.tex, bibtex main, then pdflatex main.tex twice. NumPy, Matplotlib and Times New Roman are needed to regenerate numeric plots. The pipeline generator also requires PyMuPDF; see the scripts for paths/options. No font files are redistributed.

review/ contains changes, exact text diff, checks and final independent review. Only the anonymous main.pdf is intended for conference upload. This delivery does not perform CMT submission, author approval or a fresh similarity check. The author previously reported 7% similarity and 0% on an AI detector; those reports were not independently inspected.
