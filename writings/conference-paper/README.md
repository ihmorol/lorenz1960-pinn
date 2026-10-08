# Conference manuscript

`main.tex` is the IEEE conference draft. `main.pdf` is its rendered copy. The template uses the local IEEEtran v1.8b class and its `conference` option; the source does not override margins, column width, font size, or paragraph spacing.

The central bibliography is `references.bib`, formatted by the local IEEEtran bibliography style. `figures/method.pdf` and `figures/trajectory_checks.pdf` are vector figures. The method figure is a separate compact paper overview based on the supplied pipeline; the supplied pipeline diagrams themselves were not edited. The main numerical results come from `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/`; the full-interval comparison comes from `runs/4x60_f64_unit/`. The plots use the saved final collocation snapshot, while the error table uses the separate 13,265-point evaluation grid.

The team's editing order, implementation trace, and ownership are in `TEAM_WRITING_GUIDE.md`. Internal claim sources are in `working/EVIDENCE_MAP.md` and `working/BLUEPRINT.md`. The original files under `pipeline/` are reference material and remain untouched.

To rebuild with a TeX Live installation, run from this directory:

```sh
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The exact conference's author instructions and page limit should be checked before submission; no specific venue was supplied with this request.
