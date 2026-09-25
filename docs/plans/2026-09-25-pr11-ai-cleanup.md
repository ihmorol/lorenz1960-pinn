# PR #11 AI-pattern cleanup plan

Scope: the integrated review branch's plotting entry points and run notes.
Do not rewrite the selected causal trainer, alter saved metrics, or connect
the three active runs through a shared evaluation abstraction.

Behavior lock: `test_run_extras_are_written`, `test_causal_extras_are_written`,
`test_load_run_reproduces_the_saved_prediction`, and
`test_root_scripts_compile` passed before edits. The saved final and
shared-network checkpoints remain separate manual QA gates.

1. Remove dead cross-run comparison wrappers and their unused implementation.
   Drop the uncalled Manim renderer, which requires an undeclared dependency
   and draws an arrow that the historical audit found is not a gradient. The
   unique historical images, CSV, and rendered films stay in their archives.
2. Load a parameter trail once in `generate_run_extras` and use explicit
   `WindowedPINN` typing instead of checking for an `edges` attribute. Detach
   joint residual values before converting them to plot scalars.
3. Mark the older full-orbit causal run's inherited gradient figures as
   historical diagnostics. Do not transfer that caveat to the later 1,536-point
   run without evidence: its trainer already clears model-wide gradients.
4. Re-run targeted tests after each code pass, then the full suite, compile,
   saved-checkpoint evaluations, and diff check.

Fallback inventory: the `torch.load` `TypeError` paths target Torch 2.0-2.5
compatibility under `torch>=2.0`; Plotly's `ImportError` path is documented as
an optional output. These are grounded boundary fallbacks, not masking paths,
and this pass will preserve them. No broad compatibility branch is added.

## Outcome

The dead comparison and Manim source were removed; archived images and films
remain. Plotting now loads parameter trails once and uses explicit model types.
Joint-residual plotting no longer converts an autograd tensor implicitly.
The focused plotting tests and full 43-test suite pass, and both saved active
checkpoints still reproduce their recorded trajectories. LSP diagnostics are
unavailable for this isolated worktree because the service is scoped to the
original checkout; Python compilation and executable tests provide validation.
