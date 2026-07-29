# ADR 0001 — Implement a PINN for Lorenz-1960 before the plain-ANN study

**Status:** Proposed — pending supervisor approval
**Date:** 2026-07-16

## Context

The locked research scope (`CLAUDE.md`, `CONTEXT.md`) is "plain feedforward ANN
only; PINNs are a literature comparison, not implemented." The team decided to
implement a Physics-Informed Neural Network (PINN) **first** — to verify a PINN
can actually solve the Lorenz-1960 system and to establish a baseline PINN
architecture — before the plain-ANN architecture study.

## Decision

1. Implement a forward PINN (hard initial condition via trial solution,
   residual-only loss) for Lorenz-1960, following paper1 (Matthews & Bihlo,
   PinnDE) as the primary method reference.
2. Place all new code in a self-contained top-level **`fydp2/`** folder. This
   **supersedes**, for the FYDP-2 phase, the `CLAUDE.md` guidance to put new code
   under `src/models/` and `src/experiments/`.
3. The locked baseline `src/baseline/lorenz1960_baseline.py` is imported for
   equations, ground truth, and error metrics — never modified.
4. The research question in `CONTEXT.md` is **not** edited until a supervisor
   approves this scope expansion.

## Consequences

- The study gains a PINN implementation and a PINN-vs-ANN comparison on
  Lorenz-1960 (previously only cited from the literature).
- Equation authority stays single-sourced (imported from the baseline).
- Until supervisor approval, this ADR remains **Proposed**; the documented scope
  in `CONTEXT.md` is unchanged.

## References

- `docs/superpowers/specs/2026-07-16-pinn-lorenz1960-harness-design.md`
- `references/paper1.pdf` (PinnDE), `references/Lagaris1998_ANN_for_ODE_PDE.pdf`

---

*Copied verbatim from the FYDP research repository, where this decision was
recorded. The files it cites (`CONTEXT.md`, `CLAUDE.md`, `references/paper1.pdf`,
the design spec) live in that repository, not here. The path in Decision 2 —
"a self-contained top-level `fydp2/` folder" — corresponds to `src/fydp2/` in
this code-only repository.*
