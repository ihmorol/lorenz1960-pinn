# Poster copy, grounded in the final manuscript

Source: `E:/University/FYDP/lorenz1960-pinn/writings/conference-paper/main.tex` (read 2026-09-30), checked against `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/run_summary.csv` and `runs/4x60_f64_unit/run_summary.csv`.

## Title
Sequential Causal Training of Windowed PINNs for the Lorenz ODE System

Authors and affiliation follow the manuscript author block. No venue, date, sponsor, or contact claim is added.

## Opening claim
On one nearly closed Lorenz-system orbit, the saved 27-window PINN follows a DOP853 reference with combined state RMSE 6.97 × 10⁻⁵ on 13,265 evaluation times. This is one seed, initial state, and coefficient set.

## 01 | Problem and full-interval observation

- Solve ẋ = −0.1yz, ẏ = 1.6xz, ż = −0.75xy, with u(0) = (0.5, 0.75, 1.0)ᵀ over t ∈ [0, 13.26446].
- The saved full-interval network reached combined state RMSE 2.237. Its trajectory remained away from the numerical reference despite a lower training objective.
- Figure: paper `platue-results.png`, shown complete.

## 02 | Sequential causal window training

1. Split [0, T] into 27 equal windows. Use 1,536 uniform collocation times per window (41,446 unique times overall).
2. In window k, train a separate 1–60–60–60–60–3 tanh network. The trial solution ûₖ(t) = u₀,ₖ + (t − tₖ)Nₖ(t; θₖ) fixes its starting state.
3. Compute rₖ(t) = dûₖ/dt − f(ûₖ(t)) by automatic differentiation and the mean squared residual at each ordered time.
4. Use causal weights wₖ,ᵢ = exp(−ε Σⱼ<ᵢ Lₖ,ⱼ). Adam minimizes the weighted residual loss through ε = 0.01 → 0.1 → 1 → 10. A stage ends when the verified minimum weight exceeds 0.99 or reaches 4,000 updates.
5. Refine with L-BFGS on the unweighted residual MSE. Save the window model and set u₀,ₖ₊₁ = ûₖ(tₖ₊₁); initialize the next network from θₖ. The networks remain separate after training.
6. Evaluate the stitched prediction against DOP853 after training. The reference does not enter the loss, gradient, or stopping test.

## 03 | Saved-run evidence

- Combined state RMSE: 6.97 × 10⁻⁵; maximum state-error norm: 3.09 × 10⁻⁴ on 13,265 evaluation times.
- Dense-grid ODE residual MSE: 1.872 × 10⁻⁷; collocation-grid residual MSE: 1.874 × 10⁻⁷.
- The paper's candidate figure shows the solution against the reference, signed state errors, and training history. Its plotted sample-grid MSE is a different measure from the headline combined RMSE.

## 04 | Comparison and scope

- The saved full-interval and windowed configurations have combined RMSE 2.237 and 6.97 × 10⁻⁵, respectively. The comparison is descriptive: network count, collocation grid, loss, warm start, and optimizer schedule differ.
- Only one seed, initial state, and coefficient set were tested. Predicted-state handoff can propagate error. Window values are continuous by construction; derivatives are not constrained to match.
- Next test: matched configurations over several seeds, changing one training option at a time.
