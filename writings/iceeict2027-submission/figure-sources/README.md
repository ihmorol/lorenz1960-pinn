# Figure source audit workspace

`pipeline_figure_times.py` is a copy of the recovered pipeline generator with two isolated changes: it writes under `generated/`, and its HTML body uses Times New Roman. `pipeline-original.html` is the recovered generator's unmodified output. `render_pipeline.js` prints either HTML source with Edge after waiting for fonts and MathJax.

The generated files are audit candidates. They are deliberately separate from the manuscript's `figures/` directory. The Times candidate was rejected because its text baselines differ visibly from the original and its printed labels remain below 8 pt at one-column scale.

The compact numerical figures do not have exact saved generators that reproduce their present page geometry. The saved run plotting path for the trajectory is `src/pinn/viz/figures.py`, while the historical plateau generator at commit `45e55fa` produces a different two-panel figure. The current numerical PDFs therefore remain the authoritative sources.


`pipeline_figure_original.py` is the byte-identical recovered generator copied from scripts/pipeline_figure.py for provenance. Its original output-path logic is preserved; use the separate candidate generator and render script for isolated comparisons.
