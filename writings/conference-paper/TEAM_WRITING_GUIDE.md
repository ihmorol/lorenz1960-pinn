# Five-member writing and integration guide

## Source of truth

The central manuscript is `conference-paper/main.tex`. It uses `IEEEtran` in `conference` mode and a separate `references.bib`. Keep the IEEE class defaults for margins, font, columns, and paragraph spacing. The figures in `conference-paper/figures/` are vector PDFs. This is a draft for the team to rewrite in its own voice; the paper's factual boundary is already fixed by the saved branch results.

Use the final `candidate-1536pt` run as the main experiment:

- `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/history/config.json`
- `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/run_summary.csv`
- `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/metrics.csv`
- `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/history/stages.csv`
- `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/breakdown/epoch_067908.csv`
- `runs/4x60_f64_unit/run_summary.csv` for the separate full-interval run.

Read `docs/results/2026-09-22-two-branch-code-audit.md` and `docs/results/2026-09-23-three-causal-run-comparison.md` before interpreting comparisons. `docs/paper_vs_branch_analysis.md`, `manuscript/`, and `manuscript-v2/` explain the history but are not the authority for numbers or method details. In particular, the old `[0,1]` result is a different experiment and must not be presented as a baseline for this orbit.

## Locked statements for every member

1. The ODE is $\dot x=-0.1yz$, $\dot y=1.6xz$, $\dot z=-0.75xy$, with $u(0)=(0.5,0.75,1)$ and $T=13.26446$. The rounded endpoint is nearly, not exactly, one period.
2. The main model uses 27 independent networks of shape `1-60-60-60-60-3`: 11,283 parameters each and 304,641 in total. Warm start copies weights. It does not make the networks share trained parameters.
3. The trial solution is $u_k(t)=u_{0,k}+(t-t_k)N_k(t)$. The previous predicted endpoint is the next hard initial state. This guarantees value continuity at joints, not derivative continuity.
4. The causal loss uses each time point's mean of three squared ODE residuals. Weights are $w_i=\exp(-\epsilon\sum_{j<i}L_j)$ and are detached during differentiation. Four epsilon stages use `0.01, 0.1, 1, 10`; each ends after `min(w)>0.99` or 4,000 Adam steps. Two stages hit their cap in the final run. L-BFGS then uses the **unweighted** residual.
5. The global uniform collocation grid has 41,446 unique points. Nonfinal windows also train on their outgoing endpoint; all windows have 1,536 residual times during optimization. The dense reference grid has 13,265 times. The reference is SciPy DOP853 with `rtol=1e-10` and `atol=1e-12`. Its error is logged during training as a diagnostic, but it is absent from the objective, gradient, and stopping rule.
6. The main combined RMSE is `6.966965305847747e-05`. It is $\sqrt{\operatorname{mean}_j\|e(t_j)\|_2^2}$, not the mean of component RMSEs. Maximum state-error norm is `3.093231691686221e-04`; dense residual MSE is `1.87248116e-07`.
7. The main run has one seed. The full-interval run changes the network count, training grid, loss, optimizer schedule, and warm start. Its RMSE `2.23706051` describes that saved configuration. It is not an ablation proving which change caused the difference.
8. The conserved forms are $I_b=b_xx^2+b_yy^2+b_zz^2$ for $b\cdot(-0.1,1.6,-0.75)=0$. The evaluator uses two orthonormal null-space basis vectors. **Do not write** $x^2+y^2$ or $x^2+z^2$ as conserved forms; those were incorrect in the rough draft.
9. The saved wall times are 881 s for the main CUDA run and 4,529 s for the full-interval CUDA run. The GPU model was not saved. Do not call the method intrinsically faster or compute hardware-normalized speedups.
10. Avoid claims of a new causal loss, causal proof, essentiality of any component, universal failure of full-interval PINNs, or a controlled precision result. The work tests a particular combination of established ideas on one orbit.

## Exact implementation trail for the human rewrite

Read the code in this order. Write each operation in the same order in Section III and Algorithm 1. Every code fact below should be explained as a design choice, then as the actual operation, then as the resulting saved artifact or limitation. Do not replace the implementation with generic PINN prose.

| Step | Current source and function | Design and exact operation to describe | Check or output |
|---|---|---|---|
| 1. Run setup | `run_causal.py`; `src/pinn/sweep.py:sweep_config,run_one`; `src/pinn/config.py:Config,ensure_record` | The candidate fixes interval, 27 windows, uniform points, float64, hard unit trial, four epsilon values, threshold, cap, warm start, StepLR, and L-BFGS budget. `sweep_config` assigns the settings-hash directory; the config manifest guards against reuse with different settings. | Quote the saved `history/config.json`, not class defaults that are inactive in this causal path. |
| 2. Construct modules | `src/pinn/pinn.py:PINN,WindowedPINN,build_model`; `src/pinn/functions/windows.py:split_windows` | Split the interval equally. Build 27 separate `PINN` modules in a `ModuleList`. Each has input 1, four hidden width-60 tanh layers, output 3, Xavier-uniform weights, zero biases. State 11,283 parameters per module and 304,641 total. | The modules are independent after their initial weight copy; never call them one shared network. |
| 3. Route time and form trial | `src/pinn/pinn.py:WindowedPINN.window_of,forward,PINN.trial,set_window_start`; `src/pinn/functions/trial.py:hard_initial_condition` | `torch.bucketize(..., right=True)` assigns a shared edge to the incoming window. The trial state is `u0 + (t-t0)N(t)`. The outgoing predicted state is detached and copied into the next module's `u0` buffer. | State continuity follows at joints. Derivative continuity is not imposed. |
| 4. Make residual batch | `src/pinn/functions/collocation.py:uniform_points`; `src/pinn/train.py:train_windows` | Use 1,535 grid cells per window, giving 41,446 unique global times. The nonfinal outgoing edge is appended to its local batch because global routing assigns that edge to the next window. Every local loss then has 1,536 times. | Distinguish unique grid points from per-window loss times. |
| 5. Differentiate and score | `src/pinn/pinn.py:residual_parts,loss_terms`; `src/pinn/functions/derivative.py:time_derivative`; `src/pinn/functions/physics.py` | Clone times with `requires_grad`. Use three `autograd.grad` calls with `create_graph=True` to obtain the state derivatives. Subtract `(-0.1yz,1.6xz,-0.75xy)`. Average the three squared residual components at each time. | The hard initial state makes the extra soft initial-condition term zero in this run. |
| 6. Weight causal loss | `src/pinn/functions/losses.py:causal_loss,causal_weights` | Sort times, compute `L=r[order].pow(2).mean(dim=1)`, preceding sum `cumsum(L)-L`, `w=exp(-eps*preceding).detach()`, then `mean(w*L)`. Recompute weights each step while holding them fixed for that step's gradient. | Explain that the accumulated per-point sum depends on the number of points. |
| 7. Update each stage | `src/pinn/train.py:train_windows`; `src/pinn/functions/optimizers.py:adam_with_decay` | For each window create a fresh Adam and StepLR. Keep that optimizer across epsilon stages `0.01,0.1,1,10`. Run at most 4,000 updates in each stage, StepLR factor 0.9 per 1,000 updates. When the pre-update minimum first exceeds 0.99, recompute after the update; accept the stage only if the new minimum exceeds 0.99. At cap, record whether it passed. | Two of 108 stages reached cap without passing. Do not say all stages converged. |
| 8. Refine and hand off | `src/pinn/functions/optimizers.py:run_lbfgs`; `src/pinn/train.py:train_windows` | L-BFGS uses the unweighted residual mean, up to 1,000 iterations, history size 50, strong-Wolfe line search, float64 tolerances `1e-12` for gradient and `1e-16` for change. Save `window_XX.pt`, copy its endpoint state, and copy weights before the next window trains. | The 33,143 logged L-BFGS values are closure evaluations, not iterations. |
| 9. Save, resume, evaluate | `src/pinn/train.py:_save_progress,_load_progress,collect_artifacts,run_summary`; `src/pinn/history.py`; `src/pinn/viz/figures.py:invariant_series` | Save `progress.pt` atomically with config, all model states, history, next window, and random states. Reject a mismatched config on resume. Record reference errors at evaluation intervals for monitoring only. After training, evaluate the full piecewise model, two residual grids, state errors, and invariant drift; write summary CSVs and snapshots. | `final_loss` in the summary is a fresh whole-domain collocation residual. It is not the last local objective. Invariants use an orthonormal null-space basis. |

Do not change the supplied diagrams in `pipeline/`. Preserve their original source and exported files. If a diagram label conflicts with code, correct the paper prose and report the conflict to Member 1; do not redraw or silently relabel the pipeline.

## Paper order and content contract

| Order in PDF | Section and purpose | What its text must contain | Handoff |
|---|---|---|---|
| Title, authors | Identify the studied system and method | Use the existing author order and verified affiliations from the previous manuscript; confirm spelling and email before submission | Member 1 maintains metadata |
| Abstract, keywords | Give a self-contained miniature of the paper | Problem; 27-window design; causal Adam and L-BFGS; DOP853 monitoring without optimization feedback; combined RMSE; one-seed and non-isolating comparison boundary. No citations or unsupported novelty claims | Member 1 writes **last** |
| I. Introduction | Move from the problem to the paper's exact question | Lorenz initial-value problem; residual-trained continuous approximation; temporal ordering issue; what the 27-window experiment actually tests; short, bounded contribution statement | Member 1 sends the promised claims to Members 2–5 before integration |
| II-A. Lorenz--1960 IVP | Define the object solved | ODE, coefficients, initial state, interval; explain nearly closed orbit; derive the two-dimensional family of quadratic invariants | Member 2 gives symbols $u,a,T,I_b$ to Member 3 |
| II-B. Related work | Position the work without padding | Raissi for PINNs; Lagaris for hard constraints; Wang et al. for causal weighting; Penwarden et al. for causal sweeping with temporal decomposition. Name the borrowed idea at each citation | Member 2 maintains bibliography keys and verifies metadata |
| III-A. Model construction and time routing | Let a reader reconstruct the model | Entry path and configuration; 27 equal windows; separate 4-by-60 modules; initialization; parameter count; `bucketize` edge routing | Member 3 sends model symbols and code evidence to Member 4 |
| III-B. Hard state and handoff | Define what is exact and what is predicted | Hard trial formula; detached endpoint copied into a buffer; warm-start weight copy; value continuity and lack of derivative constraint | Member 3 audits Fig. 1 caption against code; the diagram stays unchanged |
| III-C. Collocation, residual, and causal loss | State every tensor operation behind the loss | Unique grid versus 1,536 local points; cloned gradient-enabled times; three autodiff calls; ODE residual; per-time squared mean; sort, cumulative sum, detach, weighted mean | Member 3 hands equations and operation order to Member 4 |
| III-D. Optimizer and saved state | Explain the executable training order | Fresh optimizer and scheduler per window; four Adam stages and post-update gate; unweighted L-BFGS; per-window file, atomic resume record, reference diagnostics, final evaluation | Member 4 owns Algorithm 1 and checks it against `src/pinn/train.py` |
| IV. Experimental protocol | Make the reported run reproducible | Table I; seed and precision; DOP853 tolerances; separate evaluation grid; exact metric definitions; full-interval run's different settings | Member 4 hands metric names and units to Member 5 |
| V-A. Accuracy and physics checks | Report observations only | Table II component and combined errors; final trajectory figure; collocation and dense residual MSE; invariant drifts; value-joint and slope-joint distinction | Member 2 checks every number against the saved CSV and aligns Fig. 2 wording with Member 5 |
| V-B. Comparison and training record | Describe contrast without causal inference | Table III full-interval versus windowed; 34,765 Adam updates; 33,143 L-BFGS closure evaluations; two capped stages; recorded time caveat | Member 5 sends the limits of the comparison to Member 1 for abstract and conclusion |
| VI. Discussion and limitations | Explain what the evidence can and cannot say | What the hard trial and handoff guarantee; why one seed, one initial state, changed settings, incoming-state error, and point-dependent causal gate limit inference. Propose matched ablations as future work, not as completed evidence | Member 5 sends final interpretation to Member 1 |
| VII. Conclusion | Answer the Introduction's question | One short paragraph with the specific interval, 27-window result, descriptive comparison, and need for matched ablations | Member 1 writes after Member 5 |
| References | Support each cited prior-work statement | Every BibTeX key must be cited; keep publisher/arXiv metadata exact; do not cite papers merely to increase count | Member 2 audits before final build |

## Equal ownership plan

The five blocks are balanced by total effort, including evidence and integration work, rather than by identical word counts. A member owns the named text and its matching artifact. Everyone reviews one adjacent block in the final pass.

| Member | Primary writing | Additional concrete responsibility | Review handoff |
|---|---|---|---|
| **1 — Fariha** | Section I, abstract, keywords, title, Section VII | Central editor: merge the five reviewed drafts, preserve IEEE layout, check that the abstract and conclusion match the final results | Reviews Section II after Member 2 finishes |
| **2 — Md. Abu Bakar** | Section II-A/B and Section V-A | Own Table II; verify bibliography entries and citation-to-claim links; audit coefficients, invariant forms, and accuracy figures | Reviews Section III-A/B after Member 3 finishes |
| **3 — Morol** | Section III-A through III-C | Audit Fig. 1 caption without editing the diagram; check architecture, routing, trial, handoff, residual, and causal-weight equations against current code | Reviews Algorithm 1 and Section IV after Member 4 finishes |
| **4 — Md. Touhidul Islam** | Section III-D and Section IV | Own Algorithm 1 and Table I; audit optimizer order, resume state, stage caps, point counts, and metrics definitions against config and `src/pinn/train.py` | Reviews Section V after Member 5 finishes |
| **5 — OMlan** | Section V-B and Section VI | Own Table III and Fig. 2; verify its plotted data, comparison settings, and interpretation limits; send figure-caption wording to Member 2 | Reviews the final abstract and conclusion after Member 1 integrates |

## How to rewrite and centralize

1. **Freeze the factual contract.** Each member first reads the locked statements above and the files for their block. If a number conflicts with a rough draft, use the saved CSV or code and tell Member 1 what changed. No one edits experiment code or reruns training for prose work.
2. **Rewrite from the evidence.** Read the code or saved table, close the rough draft, and explain the design in your own short sentences. Each methods paragraph should answer: what was designed, what operation the code performs, and what limitation that creates. Each results paragraph should state the measure, the observed value, and the scope of that observation. Keep exact symbols, numerical values, and citations; do not preserve awkward wording merely because it is in the draft.
3. **Write in private draft files.** Each member writes `conference-paper/team-drafts/member-1.md` through `member-5.md`. Include section headings matching the table above, the replacement prose, and a short evidence table with file path or DOI beside each nontrivial claim. Keep the evidence table outside the replacement prose.
4. **Use a shared terminology set.** Say “windowed PINN,” “full-interval network,” “combined state RMSE,” “collocation residual MSE,” and “L-BFGS closure evaluations.” Keep $k$ as window index, $i$ as ordered point index, $u_{0,k}$ as the window's incoming state, and $u_{\rm ref}$ as the numerical reference.
5. **Pass adjacent reviews.** Each member sends its prose and evidence table to the reviewer named above. The reviewer checks equations, numbers, citations, figure meaning, and transitions, then returns specific corrections. Avoid rewriting another member's block directly.
6. **Integrate once.** Member 1 applies reviewed changes to `main.tex` and `references.bib`. Other members do not edit those two central files concurrently. Preserve `\documentclass[conference]{IEEEtran}`, `\IEEEauthorblockN`, `\IEEEauthorblockA`, `\cite{}`, two-column floats, and IEEE reference ordering. Do not set margins, font size, line spacing, or paragraph spacing by hand.
7. **Run final checks.** Build the PDF; read it at normal zoom; confirm figures have readable text, full captions, no cut-off table cells, and no blank pages. Check every numbered equation, algorithm line, table value, and cited reference. Scan for old `[0,1]`, 3,000 Latin-hypercube points, float32, 20,000 effective epochs, and false invariant formulas. Those belong to earlier work or inactive config fields.
8. **Approve claims together.** All five members sign off on the abstract, Tables I–III, Fig. 1, and the limitations paragraph. Any new result requires a saved artifact and a matched update to the methods, results, abstract, and conclusion.

## Figure instructions

Fig. 1 (`figures/method.pdf`) is a concise solution overview. It shows the Lorenz IVP, 27 windows, per-window trial solution and residual, causal Adam, unweighted L-BFGS, state and weight handoff, and final reference evaluation. The training code also logs reference errors as diagnostics. Its caption states this distinction. The supplied diagrams in `pipeline/` are fixed reference material and must not be changed. The large A4 pipeline in `pipeline/diagrams/` is useful for internal explanation but too dense at conference column size.

Fig. 2 (`figures/trajectory_checks.pdf`) uses the saved final `breakdown/epoch_067908.csv`. The top panels compare the three predicted and reference states; the bottom panels show Euclidean state-error and ODE-residual norms. Its plotted points are collocation samples. Table II is based on the separate 13,265-point evaluation grid, so the caption must preserve that distinction. It may state where peaks occur, but it must not claim that a particular joint or training stage caused them. Both figures are vector PDFs. Keep labels at least 8 pt at final PDF size, use line style as well as color when two series share a plot, and make each caption self-contained.

## Final editorial scan

- Short sentences and plain terms. Define an acronym once. Prefer “we measured” or “the saved run shows” over “proves,” “guarantees accuracy,” or “revolutionizes.”
- Results describe measurements. Discussion interprets them and states limits. Conclusion does not introduce another result.
- Do not insert a figure because it exists. Every included figure must support a specific sentence and have a readable caption.
- If the target conference provides its own IEEE package or page limit, replace only the template-specific front matter and trim text by evidence priority; never compress spacing to force a page count.
