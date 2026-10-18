# Final revision changes

Previous version: writings/iceeict2027-submission, delivery commit c30964c4d1a8f6173ba11f95b3cfe692da424798. That folder and all earlier versions remain unchanged.

- Regenerated all three affected scientific figures as vector PDFs with Times New Roman labels of at least 8.5 pt at final column width.
- Full-interval figure: vertically separated solution/reference, signed component errors and saved residual-loss history, with a clear shared legend and readable axes. All 13,265 evaluation points and 46,251 recorded loss values are retained.
- Windowed figure: enlarged the three solution/reference pairs and signed component-error panels, with an explicit 10^-4 error scale and clearer line styles. All 13,265 evaluation points are retained.
- Pipeline: replaced the tall source layout with a concise six-step vector diagram; preserved hard initial conditions, causal Adam stages, verified threshold/cap, unweighted L-BFGS and endpoint/weight handoff. Changed its displayed indices to k=0,...,26 to match the manuscript.
- Corrected one protocol wording overstatement: updated causal weights are certified before accepting a stage transition, rather than unconditionally after every Adam update. No training logic was changed.
- Retained the previous factual/reference/claim corrections. No extra prose was added to fill space and no unsupported claims were restored.

All existing numbered equations, algorithm contents, tables, bibliography entries and IEEEtran class/style are unchanged. Numeric inference agrees with archived RMSEs (full interval 2.2370605122; windowed 6.96696530566e-5). No new training or experiment was run.

Final PDF: six A4 pages, 2,697,666 bytes, all fonts embedded, no Type 3 fonts, no overfull boxes or unresolved citations/references. Final figure labels measure at least 8.5 pt in the actual manuscript PDF.

PDF SHA-256: 9494f54f1ec97d4a3bee13c866bb03aeec4b747d7478bdbca2be2a3a50f80de9

manuscript.diff contains the exact text/caption changes from the preceding version. figure-sources/ contains the new generators, data and provenance; the new PNGs show the changed geometry directly.
