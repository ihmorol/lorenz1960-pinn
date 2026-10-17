# Latest manuscript findings

Scope: the author-designated Overleaf ZIP, not the obsolete single-seed manuscript. Scientific prose and numerical claims are preserved in this formatting revision. These findings are recommendations for a separately authorized writing/accuracy pass.

## Confirmed numerical evidence

All sixteen runs have saved summaries and final checkpoints. Recomputed aggregates support the table values: WPINN has five seeds, median RMSE 8.12941083e-5 and range 6.96696531e-5 to 2.60122036e-4; no-causal has three seeds and median 1.88943222e-4; no-warm-start has three seeds and median 2.85482396e-4; shared-network has three seeds and median 0.486230383; causal single-network gives 0.275133889; full-interval gives 2.23706051. Seed-0 WPINN records 34,765 Adam steps, 33,143 L-BFGS closure evaluations and two capped stages.

The aggregation code reproduces the exact one-sided Mann-Whitney value p=0.196 for the no-causal RMSE comparison. This verifies the calculation, not the independence assumptions or strength of the statistical conclusion. Runs share seed labels and have only three to five observations; report individual paired observations and the test specification before making inferential claims.

## Priority findings

1. The prediction-accuracy paragraph says DOP853 and RK4, but the reported metrics/configurations use DOP853. Remove RK4 unless a documented independent RK4 cross-check is supplied.
2. The ablation environment records a dirty training checkout whose diff is unavailable. State that provenance limit in the reproduction material; do not claim exact historical source recovery.
3. Define the checks behind the conclusion that the model passes every check. Replace complete overlap with a statement about visual agreement at the plotted scale. These are reporting scope issues, not changes to the recorded error values.
4. Define both quadratic invariants and the drift normalization so readers can reproduce those measurements. The current invariant family alone does not identify the reported pair and denominator.
5. Final shared-network RMSE supports poorer final accuracy. It does not by itself establish the mechanism of forgetting without measuring retention of earlier windows. Similarly, the fixed-point citation does not establish a fixed-point mechanism for this particular learned trajectory.
6. Clarify the endpoint convention in the pseudocode: prose says each window already contains both endpoints, while the algorithm says to add the right endpoint to the routed set. Explain the base routed set rather than implying a duplicate sample.
7. Scope periodic/non-chaotic claims to the chosen system, coefficients and initial state with supporting mathematics or a suitable citation. Use significant only when a statistical test supports the specific comparison.
8. Restore a concise discussion of the one-system/one-initial-state scope, small seed counts, unmatched parameter counts/grids/schedules, derivative discontinuities and endpoint-error propagation. No broader generalization has been established by these runs.

## Formatting and editorial status

A4, anonymity, abstract math presentation, acronym definitions, figure references and IEEE caption handling are corrected in the formatting draft. All prose remains, as requested. The draft is seven pages, exceeding the six-page maximum. Exact figure geometry is retained, leaving small printed text; the plots had Type 3 fonts, which are repaired independently of their geometry. Further page reduction or figure reflow needs a new user decision.

The bibliography has thirteen cited keys, all resolved. Two uncited entries are retained. Metadata completeness, grammar, abstract sentence fragments, keyword selection and prose repetition remain for the writing pass.

Sources checked 17 October 2026: https://iceeict.mist.ac.bd/authors/paper-submission/ ; https://iceeict.mist.ac.bd/authors/important-dates/ ; https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/ ; https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/improve-your-graphics/ .
