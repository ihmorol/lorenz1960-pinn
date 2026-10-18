# Figure audit

## Decision

Retain the three manuscript PDFs unchanged. The method figure's exact HTML/SVG generator was recovered and verified. A Times New Roman candidate was also produced, but it does not preserve the current visual closely enough and it cannot raise the printed labels to 8 pt without a layout redesign. The exact compact-plot generators for the current page geometry were not recovered.

No training, parameter fitting, data replacement, or manuscript edit was performed.

## Method figure provenance

| Artifact | SHA-256 | Result |
| --- | --- | --- |
| `scripts/pipeline_figure.py` | `f7ef125fa48c00fe7e2adb8fdacaae7dd68b963a4180eb743a24599d94da24a1` | Recovered generator |
| `figures/pipeline.html` | `523301fa8c5f9c0c1db6dfba2eabe1e0e33523bbea12ec45a7220b9f032eae37` | Recovered 816 by 1170 pt HTML/SVG scene |
| Submission `method.pdf` | `73beef10d98a5a16b8c3a79497587b023faec26791cffd1a080aa1cec6b8a235` | Authoritative figure |
| Re-rendered original scene | `97c3847f98f04e9f5b5e72f8bd579555e3c850ff2819f698b9ef52dc070e483e` | Different PDF metadata, identical rendered page |
| Times New Roman candidate | `b421305d6b1b4bf4068b2479bd076b95dd9ca1fd3ef987f2368ea72a8cdb8e95` | Rejected candidate |

The recovered scene was printed through Edge after `document.fonts.ready` and MathJax completion. At 144 dpi, the authoritative PDF and recovered-scene re-render are identical at every one of 3,818,880 pixels. They have the same 816 by 1170 pt page, and both embed Segoe UI regular, bold, and bold italic fonts. This establishes that the recovered Python-to-HTML/SVG path is the actual method-figure source.

## Font-only candidate result

The candidate changes only the HTML body family to Times New Roman and preserves all explicit node, arrow, frame, page, and font-size coordinates. Its PDF embeds Times New Roman regular, bold, and bold italic fonts.

| Comparison at 144 dpi | Value |
| --- | ---: |
| Identical pixels | 94.8611% |
| Changed pixels | 5.1389% |
| Mean absolute channel error | 6.6926 |
| Root mean square channel error | 33.8001 |
| Chromatic-region intersection over union | 0.97720 |

Manual inspection found shifted vertical baselines in numbered badges and several labels. The explicit boxes and routes stay fixed, but the changed font metrics alter alignment within them. The smallest method label in the final paper remains 4.153 pt because changing family does not change the 816-to-column-width scale. An 8 pt minimum would require source sizes near 26 pt or a wider placement, both of which require reflow and new geometry. The candidate therefore fails the user's replace-only-if-the-visual-matches condition.

## Compact numerical figures

The final paper exposes these minimum text sizes:

| Figure | Minimum extracted size | Source status |
| --- | ---: | --- |
| `plateau-results.pdf` | 2.663 pt | Exact current generator absent |
| `solution_vs_reference.pdf` | 3.804 pt | Saved run plotting path exists; exact compact page regeneration was not established |

The historical plateau generator at commit `45e55fa` creates `plateau_training.pdf`, a two-panel plot. The submission figure has three panels, so that source is not an exact generator. The later `gen_fig_data.py` artifact is documented as a data-only typography reconstruction and has different page geometry. It is not evidence of the current figure's original layout.

The trajectory's scientific plotting path is `src/pinn/viz/figures.py::fig_solution_vs_reference`. Existing provenance records show that the pre-font-repair manuscript copy was byte-identical to the archived run figure. The submission PDF was subsequently repaired by embedding its existing DejaVu Serif resources while preserving content streams and geometry. Re-running current plotting code would create a new typography/layout artifact, so it was not substituted.

## Recommendation

Keep `method.pdf`, `plateau-results.pdf`, and `solution_vs_reference.pdf` unchanged for this submission. Their fonts are embedded and non-Type-3, and the attempted method-family change did not pass visual equivalence. The small printed labels remain a known quality limitation rather than a falsely claimed compliance pass.
