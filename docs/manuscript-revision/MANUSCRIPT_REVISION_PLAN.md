# Five-page manuscript revision plan

Prepared 7 October 2026. This is an editing and verification plan, not a revised manuscript. No paper source, tables, results, or runs were changed for this plan.

## Updated scope: three work parts

The user's latest direction preserves the paper's scientific claims. Ablation training is already running elsewhere. Do not remove claims because the supporting runs were incomplete at the earlier inspection, and do not start duplicate training or interrupt those runs. Preserve claim wording and numerical anchors during the editorial passes; attach the completed run evidence during integration. Pending evidence is not completed verification. Any actual discrepancy found after completion must be recorded and resolved explicitly rather than silently changing a claim or reporting a conflicting number as verified.

This direction supersedes earlier recommendations below to delete or weaken scientific claims, exclude failure experiments, or require separate authorization for the already-running ablations. Earlier artifact inventories describe the inspection time only. All equation definitions, headings, dimensional explanations, table corrections, and five-page requirements remain in scope.

### Part 1: Equation updates

Deliver the mathematical chain in the existing sections, using the detailed derivations below:

- II-A: original Lorenz equations, physical-parameter notation, substitution for the three coefficients, square-bracket state/coefficient vectors, and the explicit nonlinear vector field `f:R^3→R^3`.
- III-A/B: partition boundaries, window-index ranges, network mapping and parameter count, hard trial solution, its initial-state identity, and state continuity through endpoint handoff. Distinguish state transfer from parameter copying.
- III-C: product-rule trial derivative, componentwise autograd, residual function and component signs, scalar point loss, earlier-loss prefix, causal weights, and scalar window-stage objective. Include the unweighted refinement objective.
- IV Evaluation: component error measures, combined error-norm RMSE, dense residual MSE, and their different normalizations. Explain invariant differentiation wherever the corresponding verification measurement is presented.
- Across the paper and algorithm: standardize vector brackets, index conventions, dimensions, tensor shapes, epsilon definitions, and equation-label references.

Acceptance: every symbol is defined before use, each reduction identifies its input/output dimensions, and each applicable derivation takes roughly three or four lines rather than a long proof. New numbered equations may shift numbering; preserve labels and update references automatically.

### Part 2: Table updates

Update all three existing tables using the detailed table checklist below. Preserve current headline claims and keep the original seed-0 result distinguishable from new seed aggregates.

Add one compact **ablation and failure-case table**, populated from completed runs. Proposed columns: configuration/case; completed seeds; combined RMSE median [min, max]; dense residual MSE median [min, max]; capped stages or completion status. If this is too wide, retain four columns in the paper and place the full records in supporting artifacts. Include actual seed IDs in a caption or note, not only the seed count.

Rows must correspond to experiments actually run: the causal windowed configuration, its matched no-causal ablation, and the failure configurations included in the running protocol. Add sequential single-network forgetting or changed initial-state/coefficient cases only when those experiments supply the evidence. Do not fabricate case names or values from directory names. Keep baseline single-network results in the existing comparison table unless duplicating them is necessary to interpret the new table.

Distinguish a capped causal stage, an incomplete job, numerical divergence, and a completed run with poor prediction accuracy. A stage cap alone is not a failed trajectory. Report the criterion used to classify each failure case; do not choose a numerical failure threshold after seeing the outcomes without identifying it as post-hoc.

Acceptance: every cell traces to a completed run and a defined metric, budget scopes are explicit, and single-seed values are not presented as multi-seed summaries. Record unsuccessful runs rather than silently dropping them.

### Part 3: Writing updates

The 8 October equation review and exact writing edits are in [Part 3 equation explanations](PART3_EQUATION_EXPLANATIONS.md). It supplies current source locations, searchable anchors, insert/replace instructions, and copy-paste LaTeX prose for each equation group. Use it for the equation-related passages below. The additional table and its final accompanying prose are deferred until the later discussion.

Apply every heading replacement below and remove section-announcement paragraphs. Keep scientific claims and their citations; shorten repeated exposition without deleting a claim's substance. Introduce the equations in their natural order and explain their outputs in short prose.

Add **Ablation Protocol** and **Failure Cases** as subsections under Experimental Setup, following Reference Solution and Evaluation. The training algorithm remains in III Training Procedure; the experiment protocol belongs in IV. Ablation Protocol identifies configurations, shared settings, seeds, and budget-matching rules. Failure Cases states the tested departures or failure mechanisms, evaluation criteria, and artifact recording. Put measured outcomes and the new table in Results, with a short **Ablation and Failure Results** subsection if needed.

Remove the standalone Discussion and Limitations heading to reclaim space, while preserving all distinct scientific statements by moving them to Window Handoff, Evaluation, Run Comparison, the new failure subsection, or Conclusion. This includes derivative behavior at interfaces, propagated error, configuration differences, seed/generalization scope, and the current claims about causal training and sampling. Do not delete those claims during space reduction.

Acceptance: shorter headings, no announcement-only paragraphs, no lost substantive claim, clear protocol/result separation, and exactly five rendered pages with unchanged IEEE geometry.

### Integration order

Complete the equation pass, then update table structure and captions, then edit prose around the settled notation. Integrate numerical entries when the already-running ablations finish. Finally verify claim-to-artifact links, equation arithmetic, bibliography, cross-references, and the rendered five-page layout. Do not claim completed ablation verification while jobs remain running.

## Source and current evidence

Target source: `E:/University/FYDP/lorenz1960-pinn/writings/conference-paper/main.tex`.
Associated bibliography, figures, PDF, and build log are in the same directory.
The source is outside this chat's checkout. Re-read it before implementing this plan because concurrent edits have already occurred.

The latest inspected `main.log` reports **six pages**, rather than the five pages reported by the earlier build. The revision must therefore recover a page and accommodate the new explanations. A log is build evidence, not a visual layout check; inspect the revised PDF before accepting it.

Current figure labels are already distinct (`fig:plateau`, `fig:pipeline`). Do not repeat the earlier proposed label repair. The current source also contains new related-work citations and references to released seed and ablation evidence. Those claims require artifact checks.

Implementation checked: `src/pinn/functions/physics.py`, `derivative.py`, `losses.py`, `trial.py`, `collocation.py`, `src/pinn/pinn.py`, and the endpoint handling in `train.py`. These establish the vector-field evaluation, autograd derivative, component mean, detached causal weights, hard trial solution, and endpoint accounting. They do not independently validate every saved numerical result.

## Editing rules

- Preserve the reported experiment unless a verified artifact requires a correction. Do not silently replace seed-0 results with another seed.
- Use short definitions at first occurrence, followed by at most three or four lines of mathematical breakdown where useful.
- Keep existing equation labels. New explanatory displays can be unnumbered; adding numbered equations changes subsequent numbers, so references must use labels.
- Use square brackets for explicit vectors, parentheses for function arguments and tensor shapes, and braces for sets.
- Distinguish mathematical vector dimension from implementation batch shape. These are not physical units.
- Remove section announcements, repeated motivation, and unsupported absolute claims before shrinking figures or adding layout overrides.
- Keep five pages with the existing IEEE font, margins, and column geometry.

## Headings and section openings

| Current heading | Replacement |
| --- | --- |
| Problem and Related Work | Problem and Background |
| Windowed PINN Method and Implementation | Windowed PINN |
| Experimental Protocol | Experimental Setup |
| Results | Results |
| Conclusion | Conclusion |
| Lorenz System Initial-Value Problem | Lorenz Equations |
| Related Work | Related Work |
| Model Construction and Time Routing | Network Structure |
| Hard Initial State and Window Handoff | Window Handoff |
| Collocation, Residual, and Causal Loss | Training Loss |
| Optimizer Sequence and Saved State | Training Procedure |
| Baseline Reference | Reference Solution |
| Measures and Comparison Run | Evaluation |
| State Accuracy and Physics Checks | Prediction Accuracy |
| Comparison and Training Record | Run Comparison |

Delete the paragraphs beginning “In this section we introduce”, “This section explains the implementation”, “This section explains the experimental setup”, and “This section reports the measurements”. Their content is supplied by the headings and substantive paragraphs.

## Abstract and introduction

Keep the interval, 27-window configuration, headline state error, residual metric, reference-only evaluation, and descriptive-comparison qualification. Correct grammar locally. Use “nearly closed orbit” consistently instead of alternating between exact closure and approximate closure.

Remove repeated descriptions of the interval and handoff. Do not describe the work as isolating individual components unless matched ablations support that statement. Separate the full-interval baseline failure from sequential single-network forgetting: they are different configurations and need different artifact sources. The value 2.237 belongs to the saved full-interval comparison unless another artifact proves otherwise.

## II-A: Lorenz equations

### Original system and coefficient substitution

Retain the original three scalar equations and their citation. Define the physical wavenumbers at first use. Rename them to `\kappa` and `\ell` so lowercase `k` can consistently index windows. Do not introduce uppercase `K` for a second window index.

Show the coefficient reduction in one compact aligned display:

\[
a_x=2(1/5-1/4)=-1/10=-0.1,
\qquad a_y=2(1-1/5)=8/5=1.6,
\qquad a_z=(2/2)(1/4-1)=-3/4=-0.75.
\]

In the two-column paper, break this into three aligned rows if the line is too wide. This derives the reduced coefficients directly from Equation 1 for `\kappa=2`, `\ell=1`; it does not claim to derive the original Fourier truncation from the governing flow equations.

### Vector field after the reduced scalar equations

Define the state and coefficient column vectors:

\[
u=[x,y,z]^\mathsf T\in\mathbb R^3,
\qquad a=[a_x,a_y,a_z]^\mathsf T=[-0.1,1.6,-0.75]^\mathsf T.
\]

Collect the three right-hand sides to introduce the function used later:

\[
\dot u=f(u),\qquad
f(u)=\begin{bmatrix}a_xyz\\a_yxz\\a_zxy\end{bmatrix}
=\begin{bmatrix}-0.1yz\\1.6xz\\-0.75xy\end{bmatrix},
\qquad f:\mathbb R^3\to\mathbb R^3.
\]

Here `a_x yz` means the coefficient `a_x` multiplied by `y z`; typeset with spacing to avoid reading it as a single subscript. Optional matrix form, only if space remains: `f(u)=diag(a)[yz,xz,xy]^T`. It is a diagonal matrix acting on nonlinear products, not a constant linear system `Au`.

Use lowercase `f` throughout, matching the existing residual equation. Explain in two sentences that each output component is the prescribed derivative of the corresponding state component. The known function is evaluated at the predicted state; it is not learned by the network.

Replace the initial condition with `u(0)=[0.5,0.75,1.0]^T`. Preserve the rounded final time and the qualification that it is not an exact period.

## II-B: Related work

Retain citations that directly support PINNs, hard initial conditions, causal weighting, and temporal decomposition. Remove the second statement about causal sweeping if it repeats the first. Replace the claim that this implementation investigates individual component performance with a description of the combined configuration unless verified ablations are reported.

The newly added loss-landscape, NTK, gradient-flow, and benchmark citations need bibliographic and claim checks before inclusion. This plan did not verify their external publications. Keep only the shortest relevant synthesis needed for this paper; do not spend the recovered column on a broader literature survey.

## III-A: Network structure

Define the equal partition once: `t_k=kT/27`, `k=0,...,27`, with windows indexed `k=0,...,26` and width `Delta t=T/27≈0.4912763`. Distinguish the 28 boundary times from the 27 networks.

State `N_k:R→R^3`, with four 60-unit tanh hidden layers. Its output is a correction vector used in the trial solution, not directly the physical state `[x,y,z]^T`. Keep Xavier initialization, zero biases, independent stored networks, and parameter copying for warm starts.

Remove the repeated justification for a width near 0.5. Treat that width as a configuration choice; a citation does not prove it is optimal for this problem. Keep only a short distinction between full-interval training and the separate sequential single-network forgetting experiment if both are supported by artifacts.

Parameter-count verification, suitable for the plan or a short table note:

\[
(1\cdot60+60)+3(60\cdot60+60)+(60\cdot3+3)=11{,}283,
\qquad 27\cdot11{,}283=304{,}641.
\]

## III-B: Window handoff

Introduce `k` in the opening sentence before the trial equation: it indexes windows, not the physical wavenumber. Define `u_{0,k}` as the fixed starting state and `theta_k` as network parameters.

Retain Equation 3, with the index range alongside it if it fits:

\[
\hat u_k(t)=u_{0,k}+(t-t_k)N_k(t;\theta_k),\qquad k=0,...,26.
\]

Three-line breakdown: `t-t_k` is scalar; `u_{0,k}`, `N_k`, and `hat u_k` lie in `R^3`; setting `t=t_k` makes the correction zero and gives `hat u_k(t_k)=u_{0,k}` independently of the parameters.

Retain Equation 4: `u_{0,k+1}=hat u_k(t_{k+1})`, `k=0,...,25`. Derive continuity briefly: the next trial solution at its start equals this transferred state, so both windows have the same boundary value. State continuity does not guarantee derivative continuity or eliminate propagated prediction error.

Separate state handoff from warm start: `theta_{k+1}←theta_k` copies parameters before independent optimization. The first window uses the prescribed initial condition, not a “trial initial state”.

## III-C: Training loss

### Collocation and shapes

Each window trains on 1536 uniform points including its endpoints. Consecutive windows share boundary times. The unique global-grid count is `27(1536-1)+1=41,446`; the total number of per-window point uses is `27×1536=41,472`. These counts are different and must not be interchanged.

The implementation assigns each shared boundary to the next window for global routing, then appends the outgoing endpoint when training the preceding window. The last window already contains the final endpoint. Replace “ideally 1535 points and 1 handoff point” with this precise description or its shorter count-based equivalent.

Use one point convention throughout: `i=1,...,n_k`, `n_k=1536`, and chronological order `t_{k,1}<...<t_{k,n_k}`. Array indices in code can remain zero-based.

| Quantity | At one time | Batch shape |
| --- | --- | --- |
| Time | scalar | `(1536,1)` |
| Network correction and trial state | three-component vector | `(1536,3)` |
| Time derivative and prescribed vector field | three-component vector | `(1536,3)` |
| Residual | three-component vector | `(1536,3)` |
| Point losses and causal weights | one scalar per point | `(1536,)` |
| Window-stage objective | scalar | scalar tensor |

Use a compact prose version of this table in the paper unless a small table fits better.

### Trial derivative and residual function

Apply the product rule to Equation 3, with the initial state fixed during training:

\[
\frac{d\hat u_k}{dt}=N_k+(t-t_k)\frac{\partial N_k}{\partial t}.
\]

Explain that PyTorch autograd computes this derivative of the entire trial solution component by component. It does not compute `f` by differentiation.

Before Equation 5, introduce the vector-valued residual function:

\[
r_k(t)=\frac{d\hat u_k(t)}{dt}-f(\hat u_k(t))\in\mathbb R^3.
\]

For clarity, an optional three-row expansion is:

\[
r_{k,x}=\dot{\hat x}_k+0.1\hat y_k\hat z_k,\quad
r_{k,y}=\dot{\hat y}_k-1.6\hat x_k\hat z_k,\quad
r_{k,z}=\dot{\hat z}_k+0.75\hat x_k\hat y_k.
\]

Use either the component expansion or a longer explanatory paragraph, not both. The residual function compares the learned trajectory derivative with the original ODE vector field evaluated at that trajectory. “Residual function” names the mapping; “residual vector” names its value at a time; “residual loss” names the scalar reduction.

### Point loss, Equation 6

Show the reduction `L_{k,i}=||r_k(t_{k,i})||_2^2/3=(r_x^2+r_y^2+r_z^2)/3`. Squaring removes signs, summing combines the three components, and division by three gives their mean. The output is a nonnegative scalar at each point, not a three-component vector or state-reference error.

### Causal weights, Equation 7

Define the earlier-loss prefix `S_{k,i}=sum_{j=1}^{i-1}L_{k,j}`, then `w_{k,i}=exp(-epsilon S_{k,i})`. The index `k` identifies the window; `i` identifies its point; `j` ranges over earlier points; epsilon is the stage's scalar weighting parameter.

The first prefix is zero, hence `w_{k,1}=1`; all weights lie in `(0,1]` for finite nonnegative losses. A larger epsilon suppresses later weights more strongly for the same earlier losses. Calling it a penalty on late-time residuals is misleading: it reduces their contribution while earlier residuals are large.

There are 1536 weights per window. `n_k=1536` is a count, not a weight `W`. Weights are recomputed from current losses and detached for the parameter-gradient calculation.

### Window-stage loss, Equation 8

Retain `mathcal L_{k,epsilon}=(1/n_k)sum_i w_{k,i}L_{k,i}`. Multiply each point loss by its scalar weight and average over points. This is one scalar objective for a window and epsilon stage at the current parameters, changing at each update. It is divided by `n_k`, not by `sum_i w_{k,i}`.

Define the unweighted L-BFGS objective compactly: `mathcal L_k^raw=(1/n_k)sum_i L_{k,i}`. It is also the mean of all `3n_k` squared residual components. Keep the four epsilon values as an ordered schedule, not a physical-state vector.

Delete the repeated paragraph beginning “For a long-horizon system like Lorenz-1960”. Replace universal claims that residual-only training is insufficient or reducing early errors guarantees later accuracy with a short statement of the intended causal-weighting mechanism.

## III-D: Training procedure and algorithm

Keep the Adam rate, StepLR interval, four stages, per-stage update cap, verified post-update weight threshold, unweighted L-BFGS refinement, snapshots, and saved-state handoff. Use “updates”, “iterations”, and “closure evaluations” accurately rather than calling them all epochs.

Correct the threshold wording to the minimum over all 1536 training-point weights. A cap records an unmet threshold; it does not demonstrate successful convergence. Check the optimizer configuration before equating a 1000-iteration budget with exactly 1000 closure calls.

Algorithm edits:

- Use a square-bracket initial-state vector and consistent `k` and `i` ranges.
- Label the raw output as a correction vector `N_k`, then form `hat u_k`.
- Compute the time derivative by autograd and assemble `r_k` using the ODE vector field.
- Compute `L_{k,i}`, chronologically ordered weights, and the scalar objective explicitly.
- Keep the detached-weight step and post-update stopping check.
- Distinguish outgoing-endpoint inclusion from state transfer.
- State warm-start copying once, before optimizing the next network.
- Preserve save/checkpoint and stage-cap records; avoid implementation inventory that does not explain reproducibility.

## IV-A: Reference solution and configuration table

Keep DOP853 tolerances `rtol=1e-10`, `atol=1e-12`, 13,265 uniform evaluation times, and evaluation-only usage. Replace “disjoint from the training grid”: endpoints are shared, so strict disjointness is false. Say “a separate evaluation grid” unless overlap has been measured explicitly.

Table I updates:

- Give the initial condition as `[0.5,0.75,1.0]^T`; include coefficient vector or refer directly to II-A.
- Show 27 windows and their width once; keep architecture and verified parameter count.
- Distinguish 1536 training points per window from 41,446 unique global points.
- Name the threshold `min_i w_{k,i}>0.99` and the cap as 4000 Adam updates per stage per window.
- Label L-BFGS as a configured per-window iteration budget; report observed closure calls separately if verified.
- Keep seed 0 for the original headline run. Do not change it because new folders exist.
- Keep warm start, float64 precision, scheduler, and reference-grid size.
- Include hardware only if recorded reliably; otherwise describe wall time as recorded, without a speedup claim.

## IV-B: Evaluation equations

Define `e_j=hat u(t_j)-u_ref(t_j) in R^3`. For component `c`, provide compact definitions if needed: `MAE_c=mean_j |e_{j,c}|`, `RMSE_c=sqrt(mean_j e_{j,c}^2)`, and `Max_c=max_j |e_{j,c}|`.

Derive Equation 9 with three steps: compute the three-component error; take its squared Euclidean norm; average across evaluation times and take the square root. Consequently `RMSE_2=sqrt(RMSE_x^2+RMSE_y^2+RMSE_z^2)`. It does not divide by three, unlike the residual MSE.

Define the dense residual metric as `MSE_r=(1/(3 n_eval))sum_j ||r(t_j)||_2^2`. The result is scalar. Keep residual error separate from state-reference error; a small residual alone does not prove a small state error.

For the “State norm” table row, define `mean_j ||e_j||_2`, `sqrt(mean_j ||e_j||_2^2)`, and `max_j ||e_j||_2`. Its MAE entry is a mean error norm, not a componentwise MAE average. Rename the row “Error norm” and explain the aggregation in the caption or text.

## V-A: Prediction accuracy and Table II

Retain original seed-0 numbers only after tracing them to the candidate run's metrics and summary files. Verify rounding, grid size, component order, and maximum-error aggregation. The new seed-1 aggregate value is not interchangeable with the original result.

Table II can retain its layout with a clearer error-norm row and metric definitions in IV-B. Add the seed identifier in the caption if multiple-seed results appear elsewhere. Avoid adding several per-seed copies of this table.

Explain dense-grid and collocation residual MSE using the same normalization. Identify what “reference drift below 4e-10” measures. If it is invariant drift, name the invariant and whether the drift is absolute or relative; do not present it as reference state error.

Optional invariant derivation, only when the reported check needs it: for `I(u)=b_x x^2+b_y y^2+b_z z^2`, differentiation gives `dI/dt=2xyz(b_x a_x+b_y a_y+b_z a_z)`. Choosing a coefficient vector orthogonal to `a` gives a conserved quadratic. Match the reported basis to the evaluator; do not invent a different invariant and attach existing drift numbers to it.

## V-B: Run comparison and Table III

Retain the descriptive full-interval versus windowed comparison. State that architecture count, sampling, loss, warm start, and optimizer schedule differ. Replace “uncontrolled types of ablation” with “comparison of two saved configurations”. Remove the speculative smaller-network accuracy claim unless tested.

Table III updates:

- Identify both run sources and seeds in the caption or adjoining text.
- Label total parameters and unique training points precisely.
- State Adam budgets with their scope: baseline total versus per-stage, per-window cap.
- Label L-BFGS entries as configured budgets, with per-window scope, rather than observed iterations.
- Preserve scheduler differences and verified outcome metrics.
- Keep recorded wall times with a hardware/measurement qualification; do not infer an algorithmic speedup from unmatched configurations.
- Verify the 34,765 observed Adam updates, 108 stages, two capped stages, and per-window closure-call claim against saved histories before retaining them.

## Seed runs and ablations: evidence work, not prose cleanup

Current inspection found directories `r1_no_causal_seed1`, `r1_no_causal_seed2`, `r1_no_causal_seed3`, `r4_candidate_seed1`, and `r4_candidate_seed2` under `runs/paper_ablations`. The aggregate `ablations.csv` contains only one result row, for `r4_candidate_seed1` (combined RMSE `8.12941083e-05`). Directory existence does not prove completion. `physics_checks.json` is present but its numerical checks have not been audited for this plan.

Before any new training, inventory every directory: configuration, seed, system, interval, points, dtype, architecture, optimizer schedule, completion status, all 27 window checkpoints, history, summary, and evaluation arrays. Reconcile the CSV with completed artifacts. Record incomplete or failed runs separately; never omit them silently.

Use a matched seed set for causal and no-causal configurations. Seeds 1, 2, and 3 are a practical proposed set because no-causal folders already exist, but the seed-3 causal run is not established by this inspection. Include the original seed-0 pair only if their configurations are comparable. Check whether “matched budget” means equal maximum budget or equal observed updates; threshold-based early stopping can make these different.

For each accepted seed, calculate the same combined RMSE, maximum error norm, dense and collocation residual MSE, actual updates/closure calls, capped stages, and recorded runtime. Report per-seed values in supplementary artifacts and median plus range in the paper if space permits. Keep the original seed-0 result explicitly identified rather than rewriting it into a multi-seed statistic.

One compact additional table may replace repetitive comparison prose: configuration, number of completed seeds, median RMSE with range, and median dense residual MSE with range. Populate only from audited artifacts. Do not invent pending values or label a small seed set as statistical proof of causal superiority.

Run further seeds only as separately authorized experiment work after the audit identifies missing pairs. This request produces a revision plan; it does not launch training. Broad coefficient/initial-state sweeps and forgetting experiments belong outside the five-page paper unless they directly support a retained claim.

## Discussion removal and conclusion

Remove the standalone Discussion and Limitations section. Relocate derivative discontinuity and propagated state error to III-B; unmatched comparison limits to V-B; seed, initial-state, and coefficient scope to the conclusion or experimental setup.

Delete the claim that increasing collocation points necessarily decreases training loss. The causal prefix depends on sampling count and spacing, but that does not establish the claimed optimization outcome. Remove repeated capped-stage counts.

Keep the conclusion to one short paragraph: method, verified headline measurement, descriptive comparison, and the actual remaining scope limitation. Retain reference cross-solver and invariant-verification claims only if their artifacts have been audited. Do not say “every number is traceable” without completing that audit.

## Five-page space allocation

First remove section-announcement paragraphs, the standalone discussion, repeated causal-loss explanation, repeated window-width justification, and redundant related-work sentences. Use the recovered space for the coefficient substitution, vector field, trial derivative, and vector-to-scalar loss explanation.

Keep most derivation steps inline or in compact unnumbered aligned displays. The full shape table and detailed seed ledger can stay in supporting files. If the paper still exceeds five pages, shorten algorithm narration and overlapping captions before reducing figure sizes. Do not compress margins or body text. Float movement makes exact space savings uncertain until compilation and visual inspection.

## Acceptance checks before implementing and delivering

- Re-read the live manuscript and preserve changes made by others.
- Verify equation labels and numbering after new displays; use one symbol for each role.
- Check substitution arithmetic, product-rule derivative, residual signs, component ordering, loss normalization, and first causal weight.
- Confirm global-grid counts and batch shapes against the implementation.
- Audit all retained numbers, seed labels, table budgets, and reference-verification statements against the actual saved artifacts.
- Verify newly introduced or changed bibliography claims against their primary sources if retained.
- Build with the manuscript's existing multi-file workflow; check undefined citations/references and overfull content.
- Inspect all five rendered pages, including equations, table widths, captions, float order, and readable algorithm lines.
- Deliver the revised editable source and compiled PDF only after the exact five-page condition is observed. Report any unverified experiment claim explicitly.
