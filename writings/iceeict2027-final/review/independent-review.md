# Recovered method diagram review

18 October 2026. Review of the recovered original method SVG and the frozen final PDF, by a separate Claude Code session.

Paper source commit: `80c6266a3c9524ca437e7c130181ea8c26ced4d3`

PDF SHA-256: `2a3ac3710bb8d34b48aa61571007cacb3b9e2f4720d79918f6b98cbc533b29f9`

## Verdict

PASS within this figure/layout scope. Findings: **0 critical, 0 major, 0 minor**.

| Check | Evidence and result |
| --- | --- |
| Original design retained | Compared the supplied method PDF with the new render (method-original-vs-restored.png). Same nine numbered stages, palette, badges, four network subblocks, three decision diamonds, YES/NO branches and both return paths. The subblock arrows already in the original source, previously hidden behind the blue panel, are now visible. |
| Typography | Regular labels are embedded Times New Roman at 8.5 pt at column width; formulas keep the original MathJax notation as vector glyphs. |
| Defects fixed before freezing | The next-epsilon and next-window return paths had been drawn on the same line; they are on separate lanes again (inner and outer, as in the original). The window tag no longer overlaps the problem-setup box, the frame no longer crosses the YES label, and the next-window label clears the box borders. |
| Diagram semantics | 27 windows indexed 0 to 26 as in the manuscript, 1,536 points/window, 4x60 tanh network, hard IC, causal loss with detached weights, epsilon 0.01 to 10, stage test min w > 0.99 or 4,000 steps, unweighted L-BFGS, endpoint handoff, DOP853 evaluation only. |
| Final page 3 | Viewed at readable resolution. Figure, caption and adjacent text do not overlap or clip. |
| Pagination | Six A4 pages (595.28 x 841.89 pt). Page 6 keeps free space after the references. |
| File size | 2,845,787 bytes, below 3,000,000 bytes. |
| Fonts | All 27 PDF font programs embedded; zero Type 3 fonts. |
| Build | No LaTeX errors, overfull boxes or undefined references. |
| Prose preservation | main.tex differs from 34ae92a only in the method float placement ([t] to [!t]). |
| Numerical figures, tables, references | plateau-results.pdf, solution_vs_reference.pdf, ablation-table.tex, references.bib, main.bbl and IEEEtran files are unchanged since 34ae92a. |

## Limits

No literature, statistics or historical-run audit was repeated. No training, similarity/AI screening, CMT authentication or upload. Final author approval of this exact PDF remains a separate gate.
