# Section III Writing Guide: Physics-Informed Neural Network Method

This guide is for writing Section III of the IEEE conference paper. It does not
draft the section. All project-specific facts below come from the implementation
and the saved run under `src/fydp2/`.

## Recommended size

Target **650--850 words of prose**, normally **0.8--1.1 pages in IEEE two-column
format**, including equations, one compact configuration table, and one method
figure. Treat this as a planning target; the conference's official page limit has
priority. Do not let Section III exceed about 1.25 pages unless the venue allows
a longer paper.

Suggested allocation:

| Subsection | Prose target | Main content |
|---|---:|---|
| A. Network architecture | 100--140 words | Input/output, four hidden layers, width, tanh, initialization, 11,283 parameters |
| B. Hard initial-condition trial solution | 150--200 words | Trial solution, ramp factor, exact IC argument, no IC penalty |
| C. Automatic-differentiation residual | 130--180 words | Derivative, Lorenz right-hand side, component residual |
| D. Collocation loss and optimization | 180--240 words | MSE residual, 3,000 LHS points, Adam schedule, precision, seed |
| Transitions and figure/table references | 50--90 words | Connect the method stages without repeating definitions |

Use four short subsections, three or four numbered equations, one compact table,
and one main method figure. A second method-support figure is optional only if
the final layout remains readable.

## The story Section III must tell

Write the method in execution order: a scalar time enters the network; the raw
three-state output is wrapped in a trial solution; automatic differentiation
provides its time derivative; the Lorenz right-hand side is evaluated; the
residual is squared and averaged at fixed collocation points; Adam updates the
network parameters. This ordering matches `pinn.py` and `train.py` and keeps the
section easy to follow.

The section must make these facts explicit:

1. The model is a time-to-state MLP for **one fixed initial-value problem**, not
   an operator model or a general solver.
2. The network is `1 -> 60 -> 60 -> 60 -> 60 -> 3`, with `tanh` in each hidden
   layer and a linear output layer. Weights use Xavier-uniform initialization and
   biases are initialized to zero. The verified trainable parameter count is
   **11,283**: `(1*60+60) + 3*(60*60+60) + (60*3+3)`.
3. The reported run uses the hard trial solution
   `u_T(t) = u_0 + g(t) N(t;theta)`, where
   `g(t) = (t-t_0)/(t_f-t_0)`. With `t_0=0` and `t_f=1`, `g(t)=t`, so
   `u_T(0)=u_0` for every parameter value.
4. The reported loss is physics-only. There is no initial-condition penalty in
   the run because the hard construction satisfies the initial condition by
   design. The soft-IC branch in the code is an available alternative, not a
   reported comparison.
5. The residual is `r(t)=du_T/dt-f(u_T)`, with the derivative obtained by
   PyTorch automatic differentiation. For `u_T=(x_T,y_T,z_T)`, the right side is
   `(c_x y_T z_T, c_y x_T z_T, c_z x_T y_T)` using coefficients imported from
   the baseline module.
6. The loss is the mean of the squared three-component residual over **3,000
   fixed Latin-hypercube points**. The points are sampled once before training;
   they are not the evaluation grid.
7. The reported optimizer is full-batch Adam for **20,000 iterations**, with a
   linear learning-rate decay from `1e-3` to `1e-4`, float32 arithmetic, seed 0,
   and no L-BFGS stage.

## Equations to include

Do not repeat the complete Lorenz system if it is already defined in Section II;
refer back to its equation and use Section III for the PINN construction. The
minimum equation set is:

1. Network map: `N(t;theta): R -> R^3`.
2. Trial solution: `u_T(t)=u_0+g(t)N(t;theta)` and `g(t)`.
3. Physics residual: `r(t)=d u_T/dt-f(u_T(t))`.
4. Training objective: `L(theta)=(1/(3N_c)) sum_i ||r(t_i)||_2^2`.

Immediately explain the purpose of each equation. For the trial solution, show
the one-line substitution at `t=t_0`; for the loss, state that no reference
trajectory values enter the objective.

## Figure and table plan

### Main method figure: required

Use `section_III_assets/figures/fig01_pinn_method_pipeline.pdf` (or the PNG for
drafting). It shows the exact project flow: time input, MLP, hard trial
solution, automatic differentiation, Lorenz right-hand side, residual, and loss.
Place it after the first paragraph or at the beginning of Section III. Cite it
before it appears and caption it as a computational pipeline, not as a claim of
generalization.

### Optional method-support figure

Use `fig02_collocation_design.pdf` only if the paper needs visual evidence that
the 3,000 points are a fixed LHS design. The four panels show density, empirical
CDF, normalized spacing, and the collocation-point residual norm. It is usually
better as supplementary material because the collocation procedure can be
described in one paragraph and one table row.

### Results figures, not Section III figures

Keep these for Section V unless the editor permits cross-section placement:

- `fig05_solution_vs_reference`: predicted/reference trajectories and signed errors;
- `fig06_training_dynamics`: loss, learning-rate schedule, and reference-error tracking;
- `fig07_physics_residual`: dense-grid residual components and their distributions;
- `fig08_phase_portraits`: trajectory projections and 3-D orbit;
- `fig09_error_analysis`: error growth, distribution, parity, relative error;
- `fig10_metrics_summary`: metric bars and table.

`gradient_diagnostics`, `invariant_drift`, and the legacy `results.png` are
supporting diagnostics. They are not necessary for a compact Section III and
should not be used to imply convergence guarantees or broad physical validity.

### Configuration table

Include one compact table, either at the end of Section III or at the start of
Section IV. If Section IV owns the full table, Section III should still state the
loss, collocation count, optimizer, and hard-IC choice in prose. Do not duplicate
the same values in multiple paragraphs.

## Pseudocode plan

Use one algorithm block titled **Algorithm 1: Training and evaluation of the
hard-initial-condition Lorenz-1960 PINN**. The ready-to-edit skeleton is in
`section_III_assets/pseudocode/algorithm_1_hard_ic_pinn.md`.

Keep it to roughly 18--25 lines. It must show: coefficient construction, fixed
LHS sampling, MLP initialization, trial-solution construction, automatic
derivative, residual, residual MSE, Adam update, linear learning-rate decay, and
separate-grid evaluation. It must not show source-code syntax, a reference value
inside the loss, an unreported soft-IC comparison, or an L-BFGS stage.

## What not to write in Section III

- Do not call the DOP853 trajectory a ground truth or analytical solution; that
  belongs to Section IV as a high-accuracy numerical reference.
- Do not report the final RMSE table as part of the method; put it in Results.
- Do not describe the configurable soft-IC branch as part of the reported run.
- Do not claim multi-initial-condition generalization, long-horizon validity,
  real-time speed, solver-cost savings, or convergence guarantees.
- Do not include long Python listings, repository/module maps, or implementation
  traceability tables in the main paper.
- Do not use the training loss and reference error as if they were the same
  quantity. The reference error is logged for monitoring only and is not used to
  optimize the network.

## Claims you can safely make from this project

Use wording such as:

- “The network output is constrained by a hard initial-condition trial solution.”
- “The derivative in the residual is obtained with automatic differentiation.”
- “The reported run minimizes a physics-only residual loss at 3,000 fixed LHS
  collocation points.”
- “The model is a single-trajectory approximation for the selected initial state,
  parameter setting, and interval.”

Avoid “proves,” “exact solution,” “universal,” “state of the art,” “real-time,” and
“outperforms.”

## Code/evidence map for your notes

| Writing fact | Repository evidence |
|---|---|
| MLP and hard/soft IC | `src/fydp2/pinn.py` |
| Depth, width, activation, optimizer settings | `src/fydp2/config.py` and `src/fydp2/train.py` |
| LHS sampling and separate evaluation grid | `src/fydp2/train.py` |
| Lorenz coefficients and DOP853 setup | `src/baseline/lorenz1960_baseline.py` |
| Saved loss and diagnostics | `src/fydp2/history/` |
| Final metrics | `src/fydp2/results/metrics.csv` |
| Figure generation definitions | `src/fydp2/figures.py` |

## Final Section III check

- [ ] The section says this is a time-to-state PINN for one fixed trajectory.
- [ ] The architecture and verified parameter count are correct.
- [ ] The hard trial solution and exact IC argument are shown.
- [ ] The autograd residual is mathematically defined.
- [ ] The physics-only MSE and all reported training settings are stated.
- [ ] Training and evaluation grids are distinguished.
- [ ] Algorithm 1 and Fig. 1 are cited before they appear.
- [ ] No results, unsupported claims, or source-code listings have leaked into
      the method section.
