# Revised A0 portrait layout plan

## Goal
A reader should see the initial-value problem, the complete training loop, and the saved result at normal A0 viewing distance. The pipeline is the central visual. The two rejected manuscript visuals remain excluded.

## Canvas and typography
- Physical page: 841 × 1189 mm, portrait. CSS canvas: 3179 × 4494 px at 96 dpi; print PDF at actual size.
- Margins: 60 px minimum. Neutral white background, navy headings, restrained x/y/z blue/orange/green accents and teal process accents.
- Target type: title ≥105 px; major section headings ≥56 px; diagram step headings ≥42 px; body/captions ≥32 px; table ≥30 px. Original figure labels should remain legible at their placed widths.
- Avoid fixed empty-height cards. Use a measured grid with tight section-specific content and no filler blocks.

## Reading path and area budget
1. Header (~580 px): full title, manuscript authors/affiliation, one-sentence scoped finding, narrow key-metric ribbon.
2. Problem + observed full-interval behavior (~760 px): Lorenz equations and IVP in a narrow left panel; complete original saved full-interval plot in a wide right panel.
3. Sequential causal training pipeline (~1400 px): full-width, source-checked vector HTML infographic with explicit loop and branch. Ordered: IVP → 27 windows → set u0,k and warm-start θk → 1,536 residual times → trial state/autodiff residual → four causal Adam stages and verified stopping → unweighted L-BFGS → save and endpoint/weight handoff → repeat → stitched prediction and DOP853-only evaluation. Show 41,446 unique times globally and the boundary-value/derivative distinction.
4. Saved candidate evidence (~830 px): a wide plot from the candidate run with trajectory/reference overlay, signed errors, and training-loss history. No generated scientific curves.
5. Comparison + scope (~724 px): measured two-run table beside one source figure or compact mechanism note. Prominent descriptive-comparison caveat and limits.

## Figure handling
- `assets/full_interval_plateau.png`: complete, no crop; its claim is only about the saved full-interval configuration.
- `assets/candidate_results.png`: complete, no crop; caption separates plotted sample MSE from the 13,265-time combined RMSE.
- `assets/causal_weights.png`: use only if the final layout keeps its legend readable; label as selected saved optimization states, not as all epsilon stages.
- Exclude method-preview/trajectory_checks-preview and their source-equivalent SVG/PDF figures, per user instruction.

## Acceptance checks
- No text/figure/panel overflow or overlap; inspect full-page PDF and magnified method/figure crops.
- One A0 page; source figures ≥150 effective dpi at placed size; fonts embedded.
- No unsupported method step or isolated-component claim. Numbers trace to paper and saved CSVs.
- Bottom content reaches the intended margin; no conspicuous blank half-panels.
