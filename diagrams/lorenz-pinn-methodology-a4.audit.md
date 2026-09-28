# Lorenz PINN methodology diagram audit

Source: `lorenz-pinn-methodology-a4.drawio`  
Target: single A4 portrait page, 794 × 1123 logical units

| ID | Element | Role | Audit status |
|---|---|---|---|
| 1 | Title | Defines the complete method | Accepted |
| 2 | Problem setup | ODE, initial state, domain, windows, collocation | Accepted |
| 3 | PINN architecture | Input, 4×60 tanh, 3-output network, hard trial | Accepted |
| 4 | Physics residual | Autodiff residual construction | Accepted |
| 5 | Causal loss | Per-time loss, causal weights, epsilon curriculum | Accepted |
| 6 | Adam stage | Adam and StepLR optimization | Accepted |
| 7 | Stage gate | Early-stage threshold or 4000-step cap | Accepted |
| 8 | Epsilon gate | Advances through all four epsilon values | Accepted |
| 9 | L-BFGS | Final unweighted refinement | Accepted |
| 10 | Artifact save | Checkpoint, history, prediction snapshot | Accepted |
| 11 | Window gate | Chooses handoff or final evaluation | Accepted |
| 12 | Window handoff | Endpoint state and warm-start weights | Accepted |
| 13 | Reference evaluation | Independent DOP853 baseline | Accepted |
| 14 | Reported evidence | Accuracy, residual, invariant, and cost outputs | Accepted |
| 15 | Feedback loops | Adam, epsilon, and next-window iteration | Accepted |

Acceptance checks:

- [x] Draw.io structure check passes: 43 cells, 22 vertices, 19 edges, zero embedded images.
- [x] Export is A4 portrait at 2383 × 3370 px.
- [x] No element or connector overlaps another element's text.
- [x] All labels remain readable at full-page view.
- [x] Feedback loops are visually distinct and correctly directed.
- [x] No training/evaluation information beyond the code-backed method is added.
- [x] Labels are concise; explanatory filler and promotional wording are omitted.
