# A0 defense poster

- Print master: `poster-a0.pdf` (one A0 portrait page, 841 × 1189 mm). Print at actual size / 100%; do not fit to another paper size.
- Preview: `poster-a0-150dpi.png` (4967 × 7022 pixels, equivalent to 150 dpi at A0).
- Editable source: `poster.html` with local `assets/`.
- Layout QA: `qa.json` records the 3179 × 4494 CSS canvas, loaded assets, zero panel overflows, and aligned column bottoms.

Claims and figures were checked against `writings/conference-paper/main.tex` in the main checkout and the saved candidate in `runs/causal-window/candidate-1536pt/`. The full-interval table is descriptive: multiple configuration settings changed. The paper's method and trajectory-check figures were excluded at the user's request; the method flow is rebuilt in HTML from the paper and implementation, and trajectory evidence comes from the saved candidate run.

`render.js` exports the files using Playwright. It requires an available `playwright` module; no package is added to this branch.
