# Section III Overleaf Folder

This folder contains only the Section III contribution:

- `main.tex`: a standalone IEEE conference preview for a blank Overleaf
  workspace;
- `section_III_pinn_method.tex`: the complete method section, written in short,
  plain sentences;
- `fig01_pinn_method_pipeline.pdf`: the method figure used by the section;
- `fig01_pinn_method_pipeline.png`: a raster copy for previewing or non-LaTeX
  editors.

The files are written for the IEEE conference class
`\documentclass[conference]{IEEEtran}`. It is a section fragment, not a
standalone final paper. The included `main.tex` is only a preview wrapper. It
sets the section counter to 2 so the method appears as ``III.'' in a blank
workspace.

## Blank Overleaf workspace

Upload this complete folder. In Overleaf, set
`section_III_overleaf/main.tex` as the **Main document** and click Recompile.
This preview will show your method as Section III with its figure, equations,
table, and Algorithm 1. The placeholder title and author are only for preview.

## Add to the main IEEE file

Upload this folder to the Overleaf project and add these packages in the main
file preamble if they are not already present:

```latex
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{placeins}
```

For the complete team paper, use the team's main IEEE file and insert the section
where Section III belongs:

```latex
\input{section_III_overleaf/section_III_pinn_method.tex}
```

Keep the folder name and image path unchanged when you upload it. The source
uses IEEE figure placement, captions below figures, labelled equations, and a
numbered Algorithm 1 block. Float barriers preserve the order of the method
figure, settings table, and algorithm. Do not manually type ``III.'' in the
section title.

The section uses only the pipeline figure in this folder. Other result figures
remain in `section_III_assets/` for the team and should normally be placed in the
Results section by the teammate responsible for that section.

## Before submission

- Keep the figure and `.tex` file in this folder so the image path resolves.
- Cite the figure and Algorithm 1 before they appear in the final paper.
- Check that the main paper's Section II defines the Lorenz equations and symbols
  referenced here.
- Do not change the reported architecture, hard-IC formula, loss, or training
  settings without recording a new verified experiment.
