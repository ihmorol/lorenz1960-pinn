# Defense Question Bank and Answer Sheet

**Project:** Sequential Causal Training of Windowed PINNs for the Lorenz-1960 ODE System
**Role-play:** an external examiner who knows only the paper (`main.tex`), the pipeline figure, and the poster, and who reads them as a conference referee for a PINN venue.
**How this was built:** I read the paper, the poster brief, the evidence map, the code (`pinn.py`, `train.py`, `losses.py`, `optimizers.py`), the saved run `runs/causal-window/candidate-1536pt/…f1cbeebd/`, the earlier runs, and the repo audits in `docs/results/`. Numbers marked **[recomputed]** I re-derived myself from the saved CSVs or by re-solving the ODE, so they are checked against files, not copied from the paper.

**How to use it:** read Part 0 and Part 1 first (cheat sheet and the ten attacks that will hurt). Then read the sections in order. Every question has a short answer you can say aloud, then the evidence, then a trap to avoid. Tags: **[HIGH]** likely and damaging, **[MED]** likely, **[LOW]** possible.

Anything marked **[confirm with team]** depends on a fact only the team knows (who did what, which GPU, why a choice was made). Settle those before the defense.

---

## Part 0. Cheat sheet: know these numbers cold

### The problem
| Item | Value |
|---|---|
| Equations | dx/dt = −0.1·y·z, dy/dt = 1.6·x·z, dz/dt = −0.75·x·y (k = 2, l = 1) |
| Initial state | (0.5, 0.75, 1.0) |
| Horizon | T = 13.26446, one near-closed loop |
| True period for this orbit | ≈ 13.264462 **[recomputed]**; the saved reference is 1.65×10⁻⁶ away from the start at T |
| Conserved quantities | x² + y²/16 and x² − (2/15)z² are constant **[recomputed, drift ~10⁻¹⁰]** |
| State ranges | x ∈ [0.342, 0.534], y ∈ [−1.642, 1.642], z ∈ [−1.124, 1.124] **[recomputed]** |
| System type | Integrable and periodic. **Not chaotic.** Not Lorenz-63. |
| Reference solver | SciPy DOP853, rtol 1e-10, atol 1e-12; invariant drift ≈ 3×10⁻¹⁰ |
| DOP853 run time, whole orbit | about 0.016 s on a laptop CPU **[recomputed]** |

### The method
| Item | Value |
|---|---|
| Windows | 27 equal windows, each 0.4913 time units wide (T/27) |
| Network per window | 1–60–60–60–60–3, tanh, Xavier-uniform weights, zero biases |
| Parameters | 11,283 per network; 304,641 total (27 × 11,283) |
| Trial solution | û_k(t) = u₀,k + (t − t_k)·N_k(t; θ_k) |
| Handoff | u₀,k+1 = û_k(t_k+1), so values are exactly continuous at the 26 joints |
| Collocation | uniform; 1,535 points per window plus the outgoing endpoint; 41,446 unique points; spacing ≈ 3.2×10⁻⁴ |
| Causal weights | w_i = exp(−ε Σ_{j<i} L_j), detached, ε ∈ {0.01, 0.1, 1, 10} |
| Stage rule | advance when min w > 0.99, or after 4,000 Adam steps |
| Adam | lr 1e-3, ×0.9 every 1,000 steps, a fresh optimizer in each window |
| L-BFGS | up to 1,000 iterations per window, history 50, strong-Wolfe line search, on the plain (unweighted) residual |
| Other | warm start θ_k ← θ_k−1, float64, seed 0 |

### The results
| Item | Windowed run | Single network |
|---|---|---|
| Combined state RMSE | **6.97×10⁻⁵** | 2.237 |
| Max state-error norm | 3.09×10⁻⁴ | 3.346 |
| Per-state RMSE (x, y, z) | 2.74×10⁻⁵, 4.96×10⁻⁵, 4.05×10⁻⁵ | not in paper |
| Residual MSE (collocation) | 1.87×10⁻⁷ | 2.80×10⁻⁴ |
| Relative L2 error | ≈ 4.6×10⁻⁵ **[recomputed]** | ≈ 1.5 |
| Wall time (CUDA, GPU model not recorded) | 881 s | 4,529 s |
| Parameters | 304,641 | 11,283 |
| Training points | 41,446 uniform | 39,793 Latin hypercube |

More facts you may be asked: median error norm 3.6×10⁻⁵, 90th percentile 1.1×10⁻⁴, 99th percentile 2.35×10⁻⁴. Endpoint error at T ≈ 5.5×10⁻⁵ **[recomputed]**. Dense-grid residual / collocation residual = 0.999. Invariant drift of the PINN: 2.6×10⁻⁴ and 1.7×10⁻⁴ (reference: ~3×10⁻¹⁰). Adam updates 34,765; L-BFGS function evaluations 33,143; 2 of 108 causal stages hit the cap (window 10, min w = 0.972; window 11, min w = 0.903). Peak GPU memory 589 MB.

### The four causal-window runs in the repo (you will be asked about run selection)
| Run | Points/window | L-BFGS max iters | Device | Combined RMSE |
|---|---|---|---|---|
| Historical "Run B" (older code) | ~1,474 | 1,500 | CUDA | 8.43×10⁻⁵ |
| 512-point | ~512 | 1,500 | CUDA | 1.57×10⁻⁴ |
| 1500-point | ~1,499 | 500 | CPU | 1.08×10⁻⁴ |
| **1536-point (the paper's run)** | 1,536 | 1,000 | CUDA | **6.97×10⁻⁵** |

The paper reports only the last row. The spread over these four runs (0.7 to 1.6 ×10⁻⁴) is the only visible estimate of how much the number moves when settings move. Treat 6.97×10⁻⁵ as "about 10⁻⁴", not as a precise value.

### Two mechanisms nobody in the paper explains, and you must know (details in Part 1)
1. **The causal stopping rule is a loss threshold in disguise.** min w > 0.99 at ε = 10 means mean residual < 1.005×10⁻³ / 1,535 ≈ **6.5×10⁻⁷**. The Adam phase ends between 5 and 6.6×10⁻⁷ in 20 of the 27 windows (below that in windows 2, 13, 14, 23, 25; above it in the capped windows 10 and 11) **[recomputed from `loss_history.csv`]**. The drop to 10⁻⁸–10⁻⁹ comes from the unweighted L-BFGS step: its median improvement over the Adam end-loss is 19×, and it improves by less than 3× in exactly the five stuck windows 0, 9, 10, 22, 25.
2. **Error is local, not accumulated.** The worst windows (0, 9, 10, 22, 25) are exactly the windows where Adam ran long and L-BFGS stalled at 4×10⁻⁷ to 1.9×10⁻⁶ **[recomputed]**. Per-window error does not grow with time (window 26 error is 5.8×10⁻⁵, window 10 is 2.0×10⁻⁴).

---

## Part 1. The ten attacks that will hurt most

Rehearse these first. They are ordered by how damaging they are if you fumble.

### Attack 1 [HIGH]. "You never isolated causal weighting. Why is the title about causal training?"
- **What they see:** the title and abstract say "causal", but the paper itself says the comparison is "uncontrolled" and no windowed run without causal weights exists.
- **Answer to give:** "You are right that our evidence does not isolate causal weighting. The comparison in Table III changes window count, loss, grid, optimizer schedule, and warm start together, and we say so in the abstract and limitations. The contribution we claim is the assembled pipeline and its measured result on this orbit, not an attribution to any one component. We also looked inside the run and found the causal stage is short: 93.6% of the Adam updates sit in the ε = 10 stage, and the other three stages together take about 2,200 updates in total (about 80 per window). The ablation, windowed with plain residual under the same budget, is the first experiment we would run next. A preliminary CPU run of it was 4× worse than the causal run, but it is one run and not in the paper."
- **Evidence:** `stages.csv`: mean steps per stage are 18, 32, 32, and 1,206 for ε = 0.01, 0.1, 1, 10. **[recomputed]**
- **Trap:** do not say "causal weights are why it works". You have no evidence for that. Do not say "Wang et al. showed it helps", because their result is for their problem, not yours.
- **Strongest move:** if there is a GPU available before the defense, run the ablation (config: same as `run_causal.py` with `causal_eps_schedule=()`). Even one result, either way, turns this attack into a finding.

### Attack 2 [HIGH]. "Your baseline is a straw man."
- **What they see:** the only comparison is a single 4×60 network on [0, 13.26] with plain loss and unscaled time, giving RMSE 2.237, which is worse than predicting the mean trajectory (1.44) or the constant initial state (1.91) **[recomputed]**. The loss plot shows it parked at a fixed point.
- **Answer to give:** "The single network converges to the equilibrium family (0, y, 0), where the right-hand side vanishes, so its residual is small (about 3×10⁻⁴) while its state error is large. That is the known failure mode of vanilla PINNs on long horizons, and it is why we moved to windows. We agree it is not a tuned baseline. A fair baseline would add input scaling, a larger network or Fourier features, and causal training on the whole interval. We did not run those, and we say the comparison is descriptive."
- **Evidence:** `platue-results.png` (z→0, x→0, y→≈1.7 while the true solution oscillates); Table III footnote text in Discussion.
- **Trap:** do not call it "the state of the art single-network PINN". It is a plain 4×60 net.

### Attack 3 [HIGH]. "Lorenz-1960 is integrable with a closed-form solution, and DOP853 solves it in 16 milliseconds to 10⁻¹⁰. Why a PINN?"
- **Answer to give:** "We do not claim a PINN beats a numerical integrator on this problem; it does not. It is slower by about five orders of magnitude and less accurate by about six. We chose it because truth is cheap and exact, the orbit closes, the invariants are checkable, and the failure of a single network on long horizons is clean to observe. The study is about whether windowed causal training can drive a PINN through a long horizon, as a stepping stone to settings where a mesh solver is not available: inverse problems, data assimilation, PDEs."
- **Evidence:** DOP853 0.016 s vs 881 s **[recomputed]**; error 3×10⁻¹⁰ vs 7×10⁻⁵.
- **Trap:** the project docs say "there is no simple formula for the answer" (`docs/the-short-version.md`). That is wrong. The system is Euler's rigid-body type, solvable with Jacobi elliptic functions. Do not repeat that sentence.

### Attack 4 [HIGH]. "Which Lorenz? Your title says 'Lorenz ODE System'. Wang et al.'s causal training was for chaotic Lorenz-63."
- **Answer to give:** "This is the 1960 truncation, not the 1963 convection model. It is conservative and periodic, with two quadratic invariants, so it is not chaotic. We use it as a smooth, bounded long-horizon test. We make no claim about chaotic regimes, where error amplification is exponential and our results would not transfer."
- **Fix before the defense:** change the title to include "Lorenz-1960" or "Lorenz (1960) ODE system". Mention the non-chaotic nature in the abstract. **[HIGH priority edit]**
- **Trap:** "Lorenz system" without a year will make examiners assume chaos. Do not let them.

### Attack 5 [HIGH]. "Is this novel? Wang, Sankaran and Perdikaris already did time-marching windows with hard initial conditions and causal weights on Lorenz-63."
- **Answer to give:** "Correct, and our paper says the components are prior work: hard initial conditions from Lagaris, causal weights from Wang, temporal decomposition from Penwarden. Our contribution is an implementation and a recorded result for a different system: per-window networks, endpoint handoff, warm start, a four-stage ε schedule with a minimum-weight stopping rule, and an unweighted L-BFGS finish, together with the diagnostics. We do not claim a new algorithm."
- **Evidence:** Related Work wording "experimental investigation of a series of well known concepts"; `docs/for-supervisor.md` notes the same Wang recipe (40 windows of 0.5, hard IC).
- **What is arguably new:** (a) the diagnostic finding that the causal gate is equivalent to a loss threshold of about 6.5×10⁻⁷ and that point density rescales it; (b) the observation that the final accuracy comes from the L-BFGS phase; (c) the junction diagnostics. Say (a) to (c) only if you can show the plots.
- **Trap:** do not say "we propose" or "novel framework". The abstract and conclusion must match the modest claim.

### Attack 6 [HIGH]. "Single seed, single initial condition, best of several runs. How do I trust three significant figures?"
- **Answer to give:** "You should not trust three significant figures. We ran one seed. Across four causal-window runs with different point counts, L-BFGS budgets and devices, the RMSE ranged from 0.7×10⁻⁴ to 1.6×10⁻⁴, so the honest statement is 'about 10⁻⁴'. The reference was used for monitoring only, but we did choose the reported configuration after seeing the reference errors of earlier runs. We state the single-seed limit in the paper. When we later re-ran the same configuration on a CPU with two seeds we got 1.5×10⁻⁴ and 1.3×10⁻⁴, so the saved run is at the favorable end of about a 2× spread."
- **Evidence:** the table in Part 0; earlier 9-architecture sweep on [0, 1] (`runs/comparison.csv`) is non-monotone in size (3×60 gave 1.3×10⁻⁴ while 3×50 gave 1.8×10⁻⁵), showing optimization noise dominates at this scale.
- **Trap:** do not say "the reference was never used". It was not in the loss or stopping, but it was used in choosing which run to report. Say that precisely.

### Attack 7 [HIGH]. "Where does the accuracy come from? Your Adam phase ends at a residual of 6.5×10⁻⁷ and the final is 10⁻⁷."
- **Answer to give:** "The stage rule min w > 0.99 at ε = 10 is equivalent to mean residual below about 6.5×10⁻⁷ for 1,535 points. Adam terminates there in most windows, and the unweighted L-BFGS phase then reduces the loss by one to two orders of magnitude in most windows. So in this run, most of the numerical accuracy is produced by L-BFGS after the causal phase. Whether the causal phase is needed to land L-BFGS in a good basin is the open question, and it is exactly the ablation we have not run."
- **Evidence:** per-window loss at the start and end of each phase **[recomputed]**: e.g. window 3: Adam ends 5.8×10⁻⁷, L-BFGS ends 1.0×10⁻⁸; window 13: 4.0×10⁻⁷ → 1.2×10⁻⁹. Windows 0, 9, 10, 22 and 25 do not improve under L-BFGS (they stay near 6×10⁻⁷, window 10 at 1.9×10⁻⁶).
- **Trap:** the paper's Methods text makes the causal stages sound like the core of the accuracy. Be ready to concede that this has not been shown.

### Attack 8 [HIGH]. "Your text says the same number, 2.237, belongs to two different experiments."
- **What they see:** the Introduction/Method says a single network trained sequentially on 27 segments "lost its previous learning" and gave "RMSE of 2.237". The abstract and Table III say a single network on the full interval gave 2.237.
- **Answer to give:** "The 2.237 comes from the saved full-interval run (Run A). The sequential shared-network experiment was an early attempt we described qualitatively; we do not have a saved result for it, so the number in that sentence is misplaced." **[confirm with team: is there a saved sequential-shared-network run? If not, delete that number from the paper.]**
- **Fix before the defense:** rewrite that paragraph so 2.237 appears only for the full-interval network, and describe the shared-network attempt as qualitative.

### Attack 9 [MED-HIGH]. "Windowing gives C⁰ continuity only. What happens to the derivative at the 26 joints, and is that where your error comes from?"
- **Answer to give:** "Values are exactly continuous by construction (the jump is at floating-point round-off, about 10⁻¹⁶). The slope is not forced to match: the right-hand slope at a joint is N_k+1(t_k+1), the network's own output, and it is trained through the residual at the joint's first collocation point; the left slope is trained through the endpoint we add to the outgoing window. Both are driven toward f(u) at the same u, so they agree up to the residual. Measured on the saved checkpoint at the 26 joints **[recomputed]**: value jump ≈ 10⁻¹⁶; slope jump (left minus right derivative) mean 1.2×10⁻³, median 7.4×10⁻⁴, max 5.6×10⁻³ (at t = 5.404, the end of the hard window 10); that is 0.17% of |f| on average and 0.6% at worst. The one-sided ODE residuals at the joints average about 1.1–1.3×10⁻³, which is higher than the run-wide RMS residual of 4.3×10⁻⁴. A joint-adjacent band (within 0.01) holds about 3.9% of the points and about 16% of the residual. That says joints are harder, but it does not show they dominate the final error."
- **Evidence:** `figures/joint_continuity.png`; `docs/results/2026-09-22-two-branch-code-audit.md` §7.
- **Trap:** do not say "C¹ continuous" and do not say "the derivative jumps a lot". Say "C⁰ exact, slope matched to about the residual".

### Attack 10 [MED-HIGH]. "Error should compound across 27 hand-offs. Your limitations say it does, your data say it does not."
- **What the saved checkpoint shows [recomputed from `pinn.pt`]:** the state error at the joints (the inherited error) rises from 1.0×10⁻⁵ at joint 1 to 6.7×10⁻⁵ at joint 9, so there is a visible accumulation over the first ten windows. It then drops to 2.2×10⁻⁵ at joint 11 after the hard window 10, and climbs again to 5.5×10⁻⁵ at the end. Mean joint error is 2.7×10⁻⁵. Separately, the error inside a window tracks the local residual left by the optimizer (log-log correlation about 0.89), and the windows with the worst residual (0, 9, 10, 22, 25) carry about 73% of the squared error.
- **Answer to give:** "Both effects are present. Each window starts from a predicted state, and the joint error does grow over the first ten windows. But the largest contributions are local fit errors in five hard windows, and the one large drop in joint error, across window 10, is not something we designed. We cannot claim a growth law from one trajectory. On a periodic orbit perturbations do not grow exponentially, so the inherited part stays small; we would not extrapolate that to chaotic systems."
- **Trap:** do not say "no accumulation" and do not say "accumulation dominates". Do not claim a growth law. The repo's own audit says the log-log `error_growth` fit is descriptive only. A drop in joint error across window 10 is not a "reset" mechanism; it means window 10's bias happened to push the state back toward the truth.

---

## Part 2. Problem and motivation

**A1 [HIGH]. Why this problem?**
It is a three-state quadratic ODE with an exactly known closed orbit, so error is measurable against a trusted reference, the long horizon (13.26 time units, about 27 windows of the length that Wang et al. use) makes a plain PINN fail, and it is cheap to train. It is a benchmark for the training method, not an application. **[confirm with team: the supervisor's original motivation.]**

**A2 [HIGH]. Is Lorenz-1960 chaotic?**
No. The equations are of Euler rigid-body type, with two independent conserved quadratics: x² + y²/16 = const and x² − (2/15)z² = const **[recomputed]**. Each trajectory lies on the intersection of two quadric cylinders, a closed curve, so motion is periodic. Maximum Jacobian norm along the orbit is about 2.0 **[recomputed]**. Sensitivity to initial conditions is at most linear in time (phase drift), not exponential.

**A3 [HIGH]. Is there a closed-form solution?**
Yes: Jacobi elliptic functions (sn, cn, dn). You did not use it as a reference because DOP853 at 10⁻¹⁰ is sufficient and simpler. If asked to confirm the period, say about 13.264462, and your T = 13.26446 is a rounded value that leaves a 1.65×10⁻⁶ gap between the start and end of the reference orbit.

**A4 [MED]. Why is T = 13.26446? Is it an exact period?**
It is the rounded period of the loop starting at (0.5, 0.75, 1.0). The reference returns to within 1.65×10⁻⁶ of the start, so the paper says "nearly closed orbit", not exact period. A note in `run_batch.py` claims a 6×10⁻⁹ return, which the saved data contradict; ignore that note.

**A5 [MED]. Why a full loop and not an arbitrary interval?**
A closed loop gives a clean periodic target, long enough that a single network fails, and a built-in sanity check (the endpoint should match the start). It does not make the problem easier: the network does not know the orbit is closed.

**A6 [HIGH]. What problem does your method solve that a standard integrator cannot?**
On this problem, none. The relevant strengths of PINNs are a mesh-free continuous solution you can differentiate and query at any t, the ability to fold in data or unknown parameters, and a route to PDEs. This paper only shows that the training scheme can reach 10⁻⁵ accuracy on a long horizon. Say that plainly.

**A7 [MED]. What is the value of a continuous, queryable solution here?**
A PINN gives u(t) and u′(t) at any time with no interpolation, but with piecewise structure (27 networks). It is also differentiable with respect to t analytically. A dense-output integrator offers the same for this problem at 10⁻¹⁰. So the value is conceptual, not practical, here.

**A8 [MED]. Why not Neural ODEs, SINDy, reservoir computing or sequence models?**
They answer different questions: those fit data, while a PINN is trained from the equations alone with no observed trajectory. The study asks whether a physics-only loss can reproduce a long trajectory. Comparison against data-driven methods is out of scope.

**A9 [LOW]. Why not Fourier features, SIREN, or an architecture that handles periodicity?**
Not tried. For a periodic orbit, a periodic input embedding would be natural and might solve the single-network problem without windows. State this as a plausible competitor and an untested baseline.

**A10 [MED]. Is the problem stiff?**
No. Rates are O(1) (|f| ≲ 1.6 × 1.1 × 0.5). The difficulty is optimization over a long horizon, not stiffness.

---

## Part 3. PINN fundamentals (the questions your own `feedback.md` already caught)

**B1 [HIGH]. In û_T = u₀ + g(t)·N(t), is the u₀ the same for every window?**
No. For window 0 it is the true initial state (0.5, 0.75, 1.0). For window k+1 it is the predicted value of window k at its right edge: u₀,k+1 = û_k(t_k+1). So the formula with u_{i+1} = u_i + g·N in your notes is the same idea with window index i.

**B2 [HIGH]. How is the derivative ̇û(t_i) computed if û(t_i) is just three numbers?**
Automatic differentiation. During the forward pass PyTorch records the operations (the computational graph). `torch.autograd.grad(u[:, j].sum(), t, create_graph=True)` walks that graph backward and returns du_j/dt at every row. It is exact for the network function (not a finite difference). `create_graph=True` keeps the derivative itself in the graph so that the loss can be differentiated with respect to the weights. Summing before the gradient works because row i depends only on t_i, so the gradient of the sum is the vector of row derivatives.

**B3 [MED]. Why does each residual term have a 1/3 factor? Are you looping three times?**
The residual r(t_i) has three components (x, y, z). L_i = (r_x² + r_y² + r_z²)/3 is simply the mean over the three equations, so each time point carries one scalar loss of comparable size to a single-equation loss. It is a scale convention, not a loop. It rescales the loss by a constant and, through ε, shifts the meaning of the causal gate.

**B4 [MED]. Is the whole batch really processed at once? Is that equivalent to looping over points?**
Yes, and equivalent. Each row's output depends only on its own t_i, so the mean loss over the batch has the same gradient as summing per-point gradients. The test suite checks this by training batched and looped from the same seed and comparing weights (`test_pinn.py`, batched-versus-looped test). The causal loss breaks that independence on purpose (weights depend on earlier points), but because the weights are detached, the gradient is still a weighted sum of per-point gradients.

**B5 [HIGH]. Why does the hard trial solution help?**
It satisfies the initial condition exactly for any weights, so there is no soft penalty γ to tune and no initial-condition error. For first-order ODEs the initial value is the anchor that makes the solution unique, so enforcing it exactly removes one source of drift. It also changes optimization: the network only learns the departure from u₀.

**B6 [MED]. Why g(t) = t − t_k and not (t − t_k)/(window width)?**
This is the "unit" form, following Lagaris and Wang et al. (û = u₀ + t·N). It makes the derivative at the window start equal N(t_k) exactly, so the network output at the start must equal f(u₀), which has a natural O(1) scale. The "span" form is available in the code (`ic_scale`) but was not used for the reported run. No comparison of the two was reported; do not claim one is better.

**B7 [MED]. What does the residual measure and why is a small residual not the same as a small error?**
The residual r = û′ − f(û) says how much the curve violates the ODE. The error satisfies, for e = û − u, e′ = f(û) − f(u) + r, so it is driven by r and propagated by the Jacobian. A residual of 4×10⁻⁴ (RMS) over T = 13 could in the worst case produce error up to r·T·e^{L·T} with L ≈ 2, which is about 10¹¹ and vacuous; in practice, on a periodic orbit, the error is of order r·T ≈ 10⁻⁴ to 10⁻³ at worst, and measured 7×10⁻⁵. The bound is not a proof; it is a reason to report both numbers. **[recomputed constants: ‖J‖ ≤ 2.0]**

**B8 [MED]. Why is a PINN with so many parameters not overfitting between collocation points?**
Each window has 11,283 parameters against 1,536 × 3 = 4,608 residual equations, so it is over-parameterized. We tested the standard overfitting failure two ways. (1) The paper's "dense-grid" residual: 1.8725×10⁻⁷ on the 13,265 evaluation times versus 1.8739×10⁻⁷ on the training grid (ratio 0.999). Be careful with the word "dense": that grid has 3.1× *fewer* points than the 41,446 training points; it is off-grid, not denser. (2) A proper probe with `scripts/analyze_generalization_gap.py` on the saved checkpoint **[recomputed]**: residual RMS 4.329×10⁻⁴ on the training grid, 4.330×10⁻⁴ at the 41,445 midpoints between training points, and 4.330×10⁻⁴ on 414,455 off-grid points of a 10× denser mesh; maximum 4.81×10⁻³ vs 4.83×10⁻³. No gap between collocation points was found. The script exists but its output is not in the paper; add it. The solution is smooth and the collocation spacing (3.2×10⁻⁴) is far finer than the solution's variation scale (O(1)).

**B9 [MED]. Why tanh?**
Smooth, infinitely differentiable, so the autograd derivative is smooth, which the residual needs. ReLU has a zero second derivative and cannot represent smooth ODE solutions well in a PINN. `config.py` lists tanh, relu, sigmoid, gelu, swish, but only tanh was used for the reported results. No activation study is in the paper.

**B10 [MED]. Why 4 layers of 60 neurons?**
Inherited from the earlier [0, 1] study, which swept 3 to 5 layers and widths 50 to 70 (`runs/comparison.csv`). That sweep was single-seed and non-monotone in size, so it does not justify 4×60 over its neighbors; it is a reasonable default. Honest answer: "we did not tune the architecture for the long horizon."

**B11 [MED]. Why Xavier initialization with zero biases?**
It is the standard choice for tanh networks, keeping activations away from saturation at initialization.

**B12 [LOW]. What is the input to the network? Absolute time or local time?**
Absolute time t (for example 12.5 in the last window), not the local offset t − t_k. The trial solution uses the local offset, but the network sees t itself. This is a design choice we did not ablate. Local time would make all windows structurally identical and could make warm starting more exact. Concede it is untested.

**B13 [LOW]. Is the loss mean or sum?**
Mean over points (`(w * L).mean()` for the causal loss, `r.pow(2).mean()` for the plain loss). That is why the causal gate depends on the number of points (see D6).

**B14 [LOW]. What is the physics function in code?**
`lorenz1960_rhs` computes (c₀·y·z, c₁·x·z, c₂·x·y) from the same coefficients function as the SciPy baseline, so the network and the reference solve the same equation. A test feeds the SciPy solution through the residual function and checks it is near zero.

---

## Part 4. Windowing and handoff

**C1 [HIGH]. Why windows at all?**
A single network trained on the whole horizon is dragged to the equilibrium family (0, y, 0): the residual is small there (3×10⁻⁴), but the state is wrong. Late-time residuals are only meaningful if early-time states are right, and a global loss does not enforce that order. Windows turn one long problem into 27 short ones, each with a known start, so the dependence of late on early times is respected by construction.

**C2 [HIGH]. Why 27 windows, and why 0.49 each?**
T/0.5 = 26.5, and we rounded up to 27 so that each window is about 0.49, close to the 0.5 length that Wang et al. use for Lorenz-63. We did not run a sensitivity study (for example 13, 27, 54). Say: "27 is a literature-guided choice, not an optimized one."

**C3 [HIGH]. Would more windows be better? Fewer?**
Untested. Shorter windows are easier to fit but add handoffs, parameters, and sequential cost. Longer windows risk the single-network failure. There is probably an optimum, and mapping error against window count is an obvious next experiment.

**C4 [MED]. Why independent networks and not one shared network?**
We tried a shared network trained sequentially and observed that earlier windows degraded when the network moved to later ones (a forgetting effect). Independent networks cannot forget: a finished window is frozen. **[confirm with team; no saved result for the shared-network attempt, see Attack 8.]** Cost: 27× the parameters.

**C5 [HIGH]. Is it fair to compare 304,641 parameters to 11,283?**
No, and the paper says so ("27 times as many parameters"). A fairer baseline would be a single network with about 300k parameters, or windows with smaller networks. The paper concedes that "reducing the parameter count of each network might result in similar error."

**C6 [HIGH]. What does warm start do, and why is it not shared weights?**
Window k+1's weights are initialized as a copy of window k's final weights, then trained independently. It gives a good starting point because adjacent windows look alike, and the copy shortens training (most windows needed only about 100 Adam steps before the gate passed). After the copy the networks are separate. It was added in a later commit; it was not ablated against random initialization.

**C7 [MED]. The network takes absolute time. Does warm start make sense?**
It works empirically (window loss at the start of a warm-started window is about 10⁻³ to 4×10⁻³, versus about 0.06 for a cold start in window 0), but the copied network was trained on a different t range. A local-time input would make the copy more principled. Untested.

**C8 [HIGH]. What exactly is the hand-off, and why is value continuity exact?**
After window k finishes, we evaluate it at its right edge t_k+1 and write that value into window k+1 as its u₀ (`set_window_start`). The next window's trial solution equals u₀ at t_k+1, so the two pieces agree there to round-off. Measured jump: about 10⁻¹⁶.

**C9 [HIGH]. Does the handoff propagate error?**
Yes, structurally: window k+1 starts from the predicted state, not the true state. The error at the start of window k+1 equals the error at the end of window k. In practice, the saved run shows no accumulation trend (see Attack 10).

**C10 [MED]. What is the error at the start of each window?**
Not reported per joint in the paper. The repo's audit recommends reporting the endpoint-state error per window and the one-sided residuals at each joint. If asked for a number: the final endpoint error at T is about 5.5×10⁻⁵ and the maximum error anywhere is 3.1×10⁻⁴. Do not invent per-joint numbers.

**C11 [MED]. Why is the endpoint of the outgoing window added to its collocation set?**
So that the outgoing network's residual is enforced at the exact state it hands forward. Otherwise the endpoint, which belongs to the next window's bucket, would have no residual constraint from the window that produced it. This is why there are 1,535 interior points plus one endpoint per window. This was added in the repaired code; the earlier "historical" run did not have it.

**C12 [MED]. Could you use overlapping windows or a Schwarz-type scheme to enforce slope continuity?**
Yes, in principle (XPINN, cPINN, overlapping domain methods). We did not try. Our scheme is the simplest: strictly sequential, no overlap, C⁰ by construction.

**C13 [MED]. What if the first window is wrong?**
Everything after inherits it. The first window is the most critical, and in our run window 0 is one of the five windows where L-BFGS stalled (final loss 6.5×10⁻⁷, error RMS 1.1×10⁻⁴). Window 0 is also the only cold start.

**C14 [MED]. Can windows be trained in parallel?**
Not in this scheme. Window k+1 needs the endpoint of window k, so training is sequential and wall time grows linearly with the number of windows. Parallel-in-time methods (parareal-style PINNs) exist and are out of scope.

**C15 [LOW]. What happens at the right boundary of the last window?**
Nothing special. The last window has no outgoing endpoint (the final grid point is t = T and belongs to it).

**C16 [LOW]. How are evaluation times assigned to windows?**
`WindowedPINN.window_of` uses `torch.bucketize` on the interior edges (`right=True`), so a time exactly on an edge belongs to the next window. The two sides agree at an edge to round-off anyway.

**C17 [MED]. Is the piecewise function differentiable?**
Inside each window yes. At joints, no guarantee: it is C⁰ with a small slope mismatch (Attack 9). If you need a smooth global function, add an overlap or slope-matching term.

---

## Part 5. Causal training

**D1 [HIGH]. State the causal loss and explain it in one breath.**
Inside a window, sort the collocation times and compute the per-point loss L_i = mean of the three squared residuals. Weight each point by w_i = exp(−ε·Σ_{j<i} L_j), so a point only counts fully once everything before it is already fitted. Train on (1/n)·Σ w_i·L_i with the weights treated as constants (detached). Code: `causal_weights` and `causal_loss` in `functions/losses.py`.

**D2 [HIGH]. Why detach the weights?**
So the optimizer cannot lower the loss by shrinking the weights (pushing early residuals up to zero out later ones). Wang et al. also hold the weights fixed within a step. Without detaching, the gradient would include a term that rewards making early errors large.

**D3 [HIGH]. What does ε do, and why four stages 0.01, 0.1, 1, 10?**
ε sets how sharply later points are suppressed by earlier error. Small ε is nearly an unweighted loss; large ε is a strict ordering. The schedule goes from soft to strict. We took the values from the idea in Wang et al. (their schedule is similar). We did not tune them and report no ε sweep.
**Brutal honesty:** the nominal ε values are not comparable to the paper's ones. Our cumulative sum is over about 1,535 individual points; Wang et al. sum over a small number of time chunks. With the same nominal ε, our weights decay many times faster per unit of loss. Do not say "we used the same ε as Wang et al." You used the same numbers on a different scale.

**D4 [HIGH]. What does "min w > 0.99" mean in practice?**
The smallest weight is the last point's: w_last = exp(−ε·ΣL). Requiring w_last > 0.99 means ε·ΣL < 0.01005. For ε = 10 and 1,535 points: mean L < 6.5×10⁻⁷. **[recomputed]** That is a loss threshold written in causal language. Observed: the Adam phase ends at a loss between 5 and 6.6×10⁻⁷ in 20 of 27 windows (for example window 0 ends at 6.53×10⁻⁷, window 3 at 5.76×10⁻⁷, window 9 at 6.48×10⁻⁷); five windows end lower (2, 13, 14, 23, 25) and the two capped windows end higher (10, 11).

**D5 [HIGH]. So is the causal weighting doing anything?**
Honest answer: it shapes the first dozens of Adam steps in each window, when the warm-started loss is 10⁻³ to 4×10⁻³ and weights at late points are small. After that the weights are all above 0.99 and the objective is effectively the plain residual. We have no experiment showing the final accuracy depends on this early shaping. In our CPU diagnostic (Part 14) a plain-residual Adam + L-BFGS run at the same Adam budget was 4× worse than the causal run (one run, so suggestive only).

**D6 [HIGH]. Does the causal gate depend on how many collocation points you use?**
Yes. ΣL grows with the point count, so the same ε and threshold mean different mean-loss levels. Measured in the repo's audit: at ε = 10, the gate requires mean loss below about 1.96×10⁻⁶ with 512 points and 6.7×10⁻⁷ with 1,500 points. So the 512-point run passed the same nominal gate with 2.9× larger residual. Any comparison of causal runs at different densities compares different gates.

**D7 [MED]. Only 2 of 108 stages were capped. Does that show causal training works?**
No. It shows the stopping condition was met, which is a statement about the loss threshold (D4), not about accuracy. Also: the two capped stages are in windows 10 and 11 at ε = 10, with min w = 0.972 and 0.903. Window 10 is the worst window in the run.

**D8 [MED]. Where do most Adam steps go?**
93.6% of the 34,765 Adam updates are in the ε = 10 stage **[recomputed]**. Mean steps per stage: 18 (ε = 0.01), 32 (0.1), 32 (1), 1,206 (10). So three of four stages are almost trivial, and ε = 10 contains a long tail: ten windows used between 1,370 and 4,000 steps there.

**D9 [MED]. Why is the stopping condition checked after the update?**
The earlier code checked the gate on parameters from before the last Adam step, which could certify weights that were never the ones entering the next stage. The repaired code recomputes min w after the update and only then advances. The paper states this. It means the reported historical run (8.4×10⁻⁵) used older logic, and the paper's run used the repaired logic.

**D10 [MED]. Is there any evidence that causal weighting helps at all on your problem?**
Only indirect: the single network (no causal weights, plain loss) failed. But it also lacked windows, warm start, a uniform grid, a good L-BFGS budget, and so on. No isolated evidence. The paper is honest about this in the abstract's last sentence and in Discussion.

**D11 [MED]. Why does Wang et al.'s finding justify skipping the no-causal ablation?**
It does not. The Discussion says no windowed run without causal weights was saved "because Wang et al. found respecting causality gives one or two orders less error." An examiner will read that as an excuse. The correct statement is "we did not run it; it is the first follow-up". **[HIGH priority edit.]**

**D12 [LOW]. Would the causal weights matter more for a chaotic or longer problem?**
Plausibly, because early error is amplified there and ordering matters. That is the setting Wang et al. studied. On a neutrally stable periodic orbit with a warm start, the benefit may be small. Untested.

**D13 [LOW]. Why is the weight sum exclusive (j < i)?**
So the first point always has weight 1 and every point is conditioned on strictly earlier points. In code: `cumsum(L) − L`.

**D14 [LOW]. What happens at window boundaries to the causal ordering?**
The causal order is only within a window; each window restarts with weight 1 at its first point. Ordering across windows is enforced by training sequentially and passing the endpoint.

---

## Part 6. Optimization details

**E1 [HIGH]. Why Adam then L-BFGS?**
Adam is robust far from a minimum and gets close; L-BFGS (quasi-Newton, strong-Wolfe line search) converges much faster near a smooth minimum on a full batch. We use full-batch training (all 1,536 points every step), so L-BFGS is valid.

**E2 [HIGH]. Does L-BFGS do most of the work?**
In this run, yes, in most windows. Per-window residual at the end of Adam vs after L-BFGS **[recomputed from `loss_history.csv`]**:
| Window | Adam ends | L-BFGS ends |
|---|---|---|
| 3 | 5.8×10⁻⁷ | 1.0×10⁻⁸ |
| 4 | 6.2×10⁻⁷ | 3.6×10⁻⁹ |
| 11 | 4.8×10⁻⁶ | 7.2×10⁻⁹ |
| 13 | 4.0×10⁻⁷ | 1.2×10⁻⁹ |
| 0 | 6.5×10⁻⁷ | 6.5×10⁻⁷ |
| 9 | 6.5×10⁻⁷ | 6.2×10⁻⁷ |
| 10 | 1.8×10⁻⁶ | 1.9×10⁻⁶ |
| 22 | (long Adam) | 5.9×10⁻⁷ |
| 25 | 4.9×10⁻⁷ | 4.2×10⁻⁷ |

**E3 [HIGH]. Why does L-BFGS fail in windows 0, 9, 10, 22 and 25?**
We do not know. What the logs show: in those windows L-BFGS used its full budget (about 1,250 evaluations) and the loss was still creeping down by about 10⁻⁹ per 125 evaluations when it stopped, so it was budget-limited, not converged. All five stuck windows had a long Adam phase at ε = 10 (1,370 to 4,000 steps). The reverse is not true: windows 11, 20, 23, 24 and 26 also had long Adam phases but L-BFGS rescued them. Their location is not explained by the solution's speed or curvature (correlation of final loss with mean acceleration is only 0.23 **[recomputed]**). Hypotheses we can offer but have not tested: an ill-conditioned valley, a poor warm-start basin, or input-scale effects from using absolute time. Say "unexplained" if asked.

**E4 [MED]. Is L-BFGS "1,000 iterations" or "1,000 evaluations"?**
`max_iter = 1000`; PyTorch's default `max_eval` is 1.25 × max_iter = 1,250 function evaluations. Total across windows: 33,143 evaluations. The paper's text "1000 L-BFGS closure evaluations per window" is wrong; it should say "up to 1,000 iterations (about 1,250 evaluations) per window; 33,143 evaluations in total". **[fix in paper]**

**E5 [MED]. Is your L-BFGS convergence tolerance meaningful?**
`tolerance_change = 1e-16` in float64 and `tolerance_grad = 1e-12`. These are so tight that the optimizer essentially always runs to its evaluation cap. So the number of evaluations reflects the budget, not convergence.

**E6 [MED]. Adam learning-rate schedule: is it a real anneal?**
StepLR multiplies the rate by 0.9 every 1,000 steps and is restarted in each window with a fresh Adam optimizer. Most windows use fewer than 1,000 Adam steps, so their rate stays at 10⁻³; the hardest windows end at about 6.6×10⁻⁴. So Adam ends at a fairly high learning rate, which also sets a noise floor close to the gate level (D4). That floor and the 0.99 gate are probably why the long-Adam windows hover near 6×10⁻⁷. Treat this as a hypothesis.

**E7 [MED]. Are `epochs = 20000` and `lr_end = 1e-4` used?**
No. In the causal mode the number of Adam steps is controlled by the stages, the 4,000 cap and the gate; `epochs` is ignored and `lr_end` is unused because StepLR is selected. They appear in `config.json` and `run_summary.csv`, which will confuse anyone who reads the config. Mention it proactively.

**E8 [MED]. What is the total compute?**
34,765 Adam steps + 33,143 L-BFGS evaluations, each on 1,536 points with a second-order autograd graph. Wall time 881 s on a CUDA device. The Colab notebook specifies a T4 GPU **[confirm with team]**; the GPU model is not recorded in the run files. A T4 is slow for float64, which matters if someone asks for throughput.

**E9 [LOW]. Why float64?**
Residuals of 10⁻⁷ and L-BFGS curvature need more than float32 can resolve; float32 round-off is about 10⁻⁷. We did not isolate the effect of precision: the repo's `precision_floor.png` mixes different horizons and cannot support a precision claim, so do not cite it.

**E10 [LOW]. Why is a fresh optimizer created for every window?**
Adam moment estimates from a different window's objective would be stale. The fresh start is conventional but untested against carrying the state.

**E11 [LOW]. Why max 4,000 steps per stage?**
Budget guard. 2 stages hit it.

---

## Part 7. The comparison with the single network (this is where you are most exposed)

**F1 [HIGH]. What exactly is the single-network baseline?**
`run_batch.py`: one 1–60–60–60–60–3 tanh network on [0, 13.26446], float64, "unit" trial function u = u₀ + t·N(t), 39,793 Latin-hypercube points (3,000 per time unit), 40,000 Adam steps (lr 10⁻³ decaying ×0.9 every 5,000), then 5,000 L-BFGS iterations (6,251 evaluations), plain mean-squared residual, seed 0. Wall time 4,529 s. Combined RMSE 2.237, max 3.346, residual MSE 2.8×10⁻⁴.

**F2 [HIGH]. How do you know the single network did not simply need more training or a better setup?**
We do not. Its loss flattens at about 3×10⁻⁴ after the first ~10,000 steps, with spikes at about 9,000, 23,000 and 36,000 iterations in the plot, and the prediction is parked near (0, 1.7, 0), a fixed point of the ODE. But it was not tuned: no input scaling (t goes up to 13.3 and multiplies the network output in the trial function), no larger network, no Fourier or periodic features, no causal weights on the whole interval, no resampling.

**F3 [HIGH]. Is the single-network result better than a trivial predictor?**
No. RMSE 2.237 is worse than predicting the constant mean trajectory (1.44) and the constant initial state (1.91) **[recomputed]**. The relative L2 error is about 1.5. The baseline is a failed training run, not a competitor, and Table III should be framed that way.

**F4 [HIGH]. What differs between the two runs?**
Number of networks (1 vs 27), parameters (11,283 vs 304,641), time-per-network (13.26 vs 0.49), loss (plain vs staged causal then plain), grid (LHS vs uniform), point count (39,793 vs 41,446), optimizer schedule (40,000 Adam + 6,251 L-BFGS evals vs 34,765 Adam + 33,143 L-BFGS evals), warm start, endpoint-in-loss, device. The windowed run receives about 5× more L-BFGS evaluations.

**F5 [HIGH]. What would a fair single-network baseline look like?**
(a) The same causal schedule on the whole interval, (b) the same total parameters (~300k) in one network, (c) input normalization t/T, (d) a periodic or Fourier input embedding, (e) multiple seeds. Any one of these might close the gap. State that you have not done them.

**F6 [MED]. Why was the baseline's wall time 4.5× longer? Is the 881 s vs 4,529 s comparison fair?**
The single network processes ~39,793 points per step for 40,000 steps; each window processes 1,536 points per step. Per-step cost is therefore much lower in the windowed run, but it needs far more L-BFGS evaluations. The two times were recorded on possibly different CUDA devices; the GPU model is not recorded. Present wall time as "recorded", not as a speed result.

**F7 [MED]. Is the comparison "fair" in parameters?**
No (C5). Per-window the networks match; in total they do not. A reviewer may argue that capacity, not windowing, explains the gain. The expected counter-argument is the single network's loss plateau, which shows an optimization failure, not a capacity limit; but we have not shown that a bigger single network also fails.

**F8 [MED]. What was the earlier "sequential single network" experiment?**
See Attack 8. If it exists as a saved run, show it; if not, remove the number from the text.

**F9 [MED]. Did you compare to any other PINN method or published number?**
No. There is no comparison with XPINN, cPINN, time-marching PINN from Mattey and Ghosh, or published Lorenz-63 results. All claims are relative to one internal run.

**F10 [LOW]. Why does the single network drift to y ≈ 1.7?**
Any point (0, y, 0) is an equilibrium of the ODE (all three right-hand sides vanish). The network can reduce the residual by decaying x and z to zero while y settles. The initial transient where the residual is large is a small fraction of 39,793 points, so the average residual is small. This is a known failure mode for long-horizon PINNs (the trivial/fixed-point solution), and the plot is consistent with it. State it as "consistent with", not "proved".

---

## Part 8. Results, metrics and evaluation

**G1 [HIGH]. Define your metrics.**
e(t_j) = û(t_j) − u_ref(t_j), a 3-vector. Combined RMSE = sqrt(mean_j ‖e‖₂²), which equals sqrt(RMSE_x² + RMSE_y² + RMSE_z²) = sqrt(2.74² + 4.96² + 4.05²)×10⁻⁵ = 6.96×10⁻⁵. Max = max_j ‖e‖₂. Residual MSE = mean of the three squared residual components over the grid. Note: the "MSE 1.67" printed in the plateau figure is the per-component MSE; combined RMSE = sqrt(3 × 1.67) = 2.24.

**G2 [HIGH]. Is 6.97×10⁻⁵ good?**
Relative L2 error about 4.6×10⁻⁵. For a PINN that is an excellent number; for an ODE solver it is poor (DOP853 at tolerance 10⁻¹⁰ is ~10⁻¹⁰, and a fixed-step RK4 reaches this at modest step counts). The number's value is as an indicator that the training method works, not as a competitive solver.

**G3 [HIGH]. Does the checkpoint reproduce the reported number?**
Yes **[recomputed]**: loading the saved `pinn.pt` and evaluating on the 13,265-point grid gives 6.966965×10⁻⁵ combined RMSE and 3.0932×10⁻⁴ max error, matching the paper. Caveat: the checkpoint file is git-ignored and not tracked in the repository (see K9). Fix that before anyone asks.

**G4 [HIGH]. Is your evaluation on training points? Is there a train/test gap?**
The state error is measured against a reference that was never used in the loss, so it is test error by construction. The residual was checked off-grid (B8): RMS 4.33×10⁻⁴ at 41,445 midpoints and on a 10× denser mesh versus 4.33×10⁻⁴ on the training grid, so no gap between collocation points.

**G5 [HIGH]. Where is the error concentrated?**
In five windows: 0, 9, 10, 22, 25 hold 81.5% of the final residual and 73% of the squared state error **[recomputed]**. Window 10 alone (t ∈ [4.91, 5.40]) holds 36.6% of the residual; the worst off-grid residual (2.36×10⁻³) is at t = 5.167 in window 10. If those five windows had typical-window quality, the RMSE would be about 4×10⁻⁵ (rough estimate, not a result). The paper says the error shows "small peaks … particularly at t ≃ 5": that is window 10 and you should know why you do not understand it.

**G6 [MED]. Per-state errors differ: y is worst. Why?**
y has the largest amplitude (y ∈ ±1.64, versus z ∈ ±1.12 and x ∈ [0.34, 0.53]), so its absolute RMSE is the largest: y 4.96×10⁻⁵, z 4.05×10⁻⁵, x 2.74×10⁻⁵. Relative to each state's span the order flips: relative MAE is 9.8×10⁻⁵ for x, 1.2×10⁻⁵ for z and 9.0×10⁻⁶ for y (`run_summary.csv`). So x, which barely moves, is about 10× worse in relative terms. Do not say "all states are equally accurate".

**G7 [MED]. Do conserved quantities hold?**
Only to about the same level as the state error: invariant drifts of the PINN reach 2.6×10⁻⁴ and 1.7×10⁻⁴, against ≈ 3×10⁻¹⁰ for the reference. A PINN has no mechanism that preserves invariants exactly (a symplectic or invariant-preserving integrator would). The paper reports the drift; be ready to say it is a limitation.

**G8 [MED]. What is the endpoint error, given the orbit should close?**
Final-state error at T is about 5.5×10⁻⁵. The reference itself is 1.65×10⁻⁶ from the start at T, so the returned state is not exactly the initial state. The PINN's distance from the initial state at T is about 3.3×10⁻⁵ (from the older audit); do not mix "error against reference" and "distance from start".

**G9 [MED]. Why is the "dense-grid" residual called dense?**
It is the residual at 13,265 evaluation times, which is fewer than the 41,446 training points. The name misleads. Rename it "evaluation-grid residual" and cite the true 10× probe (B8).

**G10 [HIGH]. You pick the best of several runs. Is that fair?**
See Attack 6. Four causal-window runs exist; the paper reports the best, chosen after seeing reference errors. The changes between the 1500-point run (1.08×10⁻⁴) and the 1536-point run (6.97×10⁻⁵) were: point count 1,500 → 1,536 (+2.4%), L-BFGS iterations 500 → 1,000, and CPU → CUDA. The point-count change is tiny, so the L-BFGS budget is the likelier cause. Do not claim 1,536 is a special density.

**G11 [HIGH]. What is the variance across seeds?**
Unknown: one seed. Evidence of large optimization variance in your own repo: the [0,1] architecture sweep (single seed per cell) has 3×60 at 1.3×10⁻⁴ and 3×50 at 1.8×10⁻⁵, a factor of 7 for a one-step change in width, which is more consistent with optimization noise than architecture. The diagnostic runs in Part 14 gave 1.48×10⁻⁴ (seed 0) and 1.28×10⁻⁴ (seed 1) on CPU, versus 6.97×10⁻⁵ for the saved GPU run, so the spread is about 2×.

**G12 [MED]. Is the reference accurate enough?**
DOP853, rtol 10⁻¹⁰, atol 10⁻¹². The reference's own invariant drift is ≈ 3×10⁻¹⁰, three orders below the PINN's error. An independent rerun with rtol 10⁻¹², atol 10⁻¹⁴ (inside `analyze_generalization_gap.py`) gives the same max error (2.885×10⁻⁴ on the checkpoint) to the digits shown. A test also feeds the SciPy solution through the PINN residual and checks it is near zero.

**G13 [MED]. Why 13,265 evaluation points and a 10⁻³ spacing?**
`eval_per_unit = 1000` over T. Arbitrary but fine relative to the solution's variation.

**G14 [LOW]. Are the plotted trajectories on training points?**
Fig. 2 uses the final collocation snapshot (training points), not the 13,265 evaluation grid (README states this). Table II uses the evaluation grid. Say so if asked, and prefer to plot the evaluation grid.

**G15 [LOW]. Does the PINN track the phase of the orbit correctly?**
The max error of 3×10⁻⁴ over a full loop with amplitude ~1.6 means the phase drift is ≲ 2×10⁻⁴ in relative terms. We did not report a phase-error metric.

**G16 [LOW]. What about the error at the start of each window versus the end?**
Joint state error: mean 2.7×10⁻⁵, max 6.7×10⁻⁵ (joint 9) **[recomputed]**; interior error in the hard windows is higher (window 10 RMS 2.0×10⁻⁴). Within a window the error is not simply growing from the start; the hard windows show a bump in the middle.

---

## Part 9. Novelty, related work and positioning

**H1 [HIGH]. What is your contribution, in one sentence?**
A recorded, reproducible implementation and measurement of a sequential windowed PINN with exact value handoff, warm start, staged causal weighting and an L-BFGS finish on a long closed orbit of the Lorenz-1960 system, together with diagnostics that show where it succeeds and where it does not. It is an application and characterization study, not a new algorithm. Say this before the examiner does.

**H2 [HIGH]. What is new compared with Wang, Sankaran and Perdikaris (2024)?**
Honestly: little at the algorithm level. They used time-marching windows (length 0.5, 40 windows for Lorenz-63 on [0, 20], per the repo's notes in `docs/for-supervisor.md`), hard initial conditions, causal weights and an ε schedule. What differs here: a different system (non-chaotic Lorenz-1960), float64 with a 41,446-point uniform grid, warm start from the previous window, the outgoing endpoint added to each window's loss, an unweighted L-BFGS finish, and the diagnostics (gate equivalence, joint analysis, stuck-window analysis). Whether the L-BFGS finish or warm start is new relative to their code you must check; do not claim it without reading their appendix.

**H3 [HIGH]. What else is related that you did not cite?**
The reference list has seven items. Missing, and likely to be raised by a PINN referee: Krishnapriyan et al. 2021 (failure modes of PINNs, including the trivial-solution problem you see), Wang, Teng and Perdikaris on gradient pathologies, Wang et al. "An Expert's Guide to Training PINNs" (2023), Mattey and Ghosh 2022 (sequence-to-sequence / time-marching PINNs), Jagtap and Karniadakis on cPINN/XPINN domain decomposition, Meng et al. on parareal PINNs, Moseley et al. on FBPINNs, and Anagnostopoulos et al. on residual-based adaptive sampling. Also classical numerics for the closed form (Jacobi elliptic functions) and Hairer for structure-preserving integrators. The repo's own survey (`docs/pinn_long_horizon_survey.md`) already covers many of these; move the relevant ones into the paper's Related Work. **[confirm each citation exists and says what you claim before adding it]**

**H4 [MED]. Is the claim "first" anywhere?**
No, and keep it that way. Do not write "first" or "novel" or "state of the art".

**H5 [MED]. Is the problem itself a contribution?**
The paper says the Lorenz-1960 closed orbit is "a simple test". It is a benchmark selection, not a contribution. The reason it is a good test (bounded, non-stiff, exact invariants, closed form, cheap reference) is also the reason it is a weak test of generality.

**H6 [MED]. What is the single most defensible technical insight in the paper?**
The causal stopping rule, with the sum taken over individual points, reduces to a mean-residual threshold of about 6.5×10⁻⁷ at ε = 10 for 1,535 points, and the point count changes its meaning. That is checkable, general and useful to anyone copying the ε schedule. Support it with the 512-point vs 1,500-point comparison in the repo audit. This is arguably the most reusable observation, and it is currently not in the paper.

**H7 [LOW]. Is the "sequential" in the title the same as a sequential-in-time marching scheme?**
In the title, "sequential" means windows are trained one after another. The repo also had an older "sequential rollout" branch (a cumulative-sum / Euler-style walk) that was tested on [0, 1] and was less accurate (8.2×10⁻⁵ and 1.2×10⁻⁴ vs 1.7×10⁻⁵ for the batched network). Do not confuse the two in the defense.

---

## Part 10. Theory questions

**I1 [HIGH]. Does your method converge? Do you have any guarantee?**
No guarantee. The method is a heuristic optimization procedure. For any window, a zero-residual network is the exact solution of the IVP; we only reach a residual of about 10⁻⁷ to 10⁻⁹ in the good windows. The error bound from the residual is r·T·exp(L·T) with L ≈ 2, which is about 10¹¹ here and is vacuous. We report measured error instead.

**I2 [HIGH]. Why is a small residual not enough?**
Because the error satisfies e′ = f(û) − f(u) + r. The residual is a forcing term; the error is its integrated effect through the Jacobian. A network can have tiny residual on a spurious near-equilibrium (as the single network does), since the residual only measures how well the ODE holds, not which solution branch you are on. Hard initial conditions plus windows reduce the room for such branches.

**I3 [MED]. Why would the ODE's neutral stability help?**
The orbit is periodic, so the linearized flow has Floquet multipliers of modulus 1 (no exponential growth or decay). A small inherited error stays small. On a chaotic system the error would grow exponentially, and the handoff would amplify it. This is a reason the benchmark is gentle.

**I4 [MED]. Is the hard-IC window formulation a universal approximator of the solution?**
Inside a window, yes in principle: u₀ + (t − t_k)·N(t) can represent any smooth function with the right initial value if N is a universal approximator (by the Lagaris argument). Over the full horizon it is piecewise smooth only with C⁰ matching.

**I5 [MED]. Is the problem in the optimization or in the representation for the single network?**
We believe optimization (an equilibrium trap), not representation: the Lorenz-1960 solution is a smooth periodic function a 4×60 tanh network can represent in principle, though we have not shown that. The windowed run represents it with 27× parameters, so it does not settle that question either.

**I6 [MED]. What is spectral bias and does it matter?**
Networks fit low-frequency components first. A function with several oscillations over [0, 13] has frequency content a small tanh network learns slowly, which is one theoretical explanation for single-network failure. The repo has NTK figures, but its own audit says they cannot establish that explanation; do not cite them as evidence.

**I7 [MED]. Why does residual weighting by cumulative earlier error implement causality?**
A later point's weight is small unless every earlier residual is small, so gradients are dominated by the earliest unconverged region. This approximates marching in time inside the batch.

**I8 [LOW]. What is the computational complexity?**
Per window the cost is (Adam + L-BFGS evaluations) × points × network cost, constant in the window count. Total scales linearly in the number of windows, hence in T at fixed window length. Windows are sequential (no parallelism). The horizon scaling was not measured.

**I9 [LOW]. How does error scale with window count or collocation count?**
Not studied. The only variation is the 512 / 1,500 / 1,536 comparison, which changes point count, L-BFGS budget and device together.

---

## Part 11. Generality, limitations and future work

Keep the current claims inside what you ran. For the future work, say only that you plan to extend the same framework (boundary-value problems, and a formulation that trains the sub-networks jointly). Do not describe any result for them, and do not suggest they are partly done.

**J1 [HIGH]. Does it work for other initial conditions or coefficients?**
Unknown. One initial state, one (k, l). The paper says so. A different initial state gives a different orbit (different period, same invariants structure). We expect the method to carry over, but we have no evidence.

**J2 [HIGH]. Does it work for chaotic Lorenz-63?**
Not tested and not claimed. Error amplification there is exponential, so inherited handoff error would grow. Wang et al.'s work addresses that case with 40 windows; our results do not speak to it.

**J3 [MED]. Does it generalize to longer horizons?**
Not tested. Cost grows linearly with the number of windows. Whether per-window accuracy holds up depends on error propagation, which for this periodic orbit is mild.

**J4 [MED]. Could it be used for PDEs?**
In principle the same ideas (hard IC in time, causal weights along time, time marching) apply to time-dependent PDEs; that is where the literature has used them. Not tested here.

**J5 [MED]. What about boundary conditions (BVPs)?**
Current code supports an `end_state` penalty only for a single window and explicitly rejects it for windowed training (`config.py`: "global end_state is not implemented for windowed training"). BVP handling is future work. If asked what the plan is, give the one-line plan and say nothing has been measured.

**J6 [MED]. What would you do next?**
In order: (1) the no-causal windowed ablation and an L-BFGS-only ablation; (2) multiple seeds; (3) a window-count sweep; (4) a stronger single-network baseline; (5) a second initial condition; (6) extend to boundary conditions and a jointly trained formulation. Items 1 to 4 are about the existing claims and should come first.

**J7 [LOW]. Why not make the network input the local time?**
Untested design choice (B12). A local-time input would make all windows share structure and might make warm starting and eventual joint training cleaner.

**J8 [LOW]. Can the 27 networks be trained at once?**
In the current method, no: window k+1 needs the predicted endpoint of window k, so training is sequential. A joint formulation is future work and should be described as such.

---

## Part 12. Code-level audit: what a careful referee can find

Everything here was checked against the repository. "Verified" means I re-ran or re-derived it. Fix the items marked **[fix]** if you can before the defense; otherwise know the answer.

**K1 [HIGH, verified]. The causal gate is a loss threshold that depends on the point count.** See D4 and D6. `causal_weights` sums per-point losses, so ε does not mean what it means in Wang et al.

**K2 [HIGH, verified]. The Adam phase stops at the gate, not at convergence.** Adam ends at about 6.5×10⁻⁷ because the gate says so, with a learning rate still near 10⁻³ (E6). Accuracy then depends on whether L-BFGS makes progress.

**K3 [HIGH, verified]. Five windows hold most of the error and the optimizer does not fix them.** G5, E3.

**K4 [HIGH, verified]. Paper vs code: L-BFGS count.** Paper says "1000 L-BFGS closure evaluations per window"; code is max_iter 1000 with max_eval 1,250, total 33,143 evaluations. **[fix]**

**K5 [MED, verified]. `epochs` and `lr_end` in `config.json` are inactive for causal runs.** E7.

**K6 [MED, verified]. The "dense" residual is not dense.** G9. **[fix wording]**

**K7 [MED, verified]. The README for the paper states Fig. 2 uses the training-point snapshot, Table II uses the evaluation grid.** Fine if disclosed; the figure caption does not say so.

**K8 [MED, audit]. Training telemetry has known defects.** From `docs/results/2026-09-22-two-branch-code-audit.md`, partly repaired by later commits: the pre-repair `final_loss` was only the last window; progress metrics included untrained future windows; stored gradient vectors included stale gradients from finished windows; the loss curve changes objective at every stage. Do not use `gradient_stability`, `gradient_histograms`, the loss landscape, the NTK figure, the film, or `precision_floor.png` as evidence. The poster uses only trajectory, error and causal-weight figures; keep it that way.

**K9 [HIGH, verified]. The run-of-record weights are not in the repository.** `.gitignore` excludes `*.pt` except for a few listed paths; `runs/causal-window/candidate-1536pt/…/history/pinn.pt` is ignored and not tracked. The checkpoint exists only inside a local worktree (`.kilo/worktrees/merciful-timbale/…`). The poster QR code points to the GitHub repository, so anyone who clones it can read your CSVs but cannot reload the model and reproduce the number. **[fix]** `git add -f` the 2.5 MB file, or attach it to a release, and note its SHA-256 in the README.

**K10 [HIGH, verified]. No device model, git commit or library versions are recorded.** The only hint is the Colab notebook asking for a T4 and the `/content/lorenz1960-pinn/…` paths in `config.json`. **[confirm with team]** which GPU; add `torch`, `numpy`, `scipy` versions and the commit hash (`5547912…` is HEAD now, the run itself predates it).

**K11 [MED, verified]. The paper's headline run used newer code than the "historical" run.** The 1536 run includes the endpoint in the loss and a post-update gate check; Run A and the historical run B do not. Table III compares a newer windowed run with an older single-network run. State that the single-network run used the older code (it does not use the causal path, so the effect is small, but say so).

**K12 [MED]. Config arch tags omit many settings.** `Config.arch` encodes window count, causal and warm flags, but not ε, L-BFGS budget or points; the causal run directory carries a settings hash to avoid collisions. The older run directories do not.

**K13 [MED, verified]. Stale documentation contradicts the paper.** `docs/the-short-version.md` says there is no simple formula for the answer and describes t ∈ [0, 1] with 3,000 points; `docs/for-supervisor.md` argues against sequential marching on [0, 1]; `run_batch.py` has a comment claiming a 6×10⁻⁹ return that the saved data (1.65×10⁻⁶) contradict. Examiners who browse the repo will see these. Add a note at the top of each saying it describes the earlier [0, 1] study, or fix the false sentences.

**K14 [MED]. Test coverage.** 71 tests (41 in `test_pinn.py`, 30 in `test_stress.py`): hard-IC exactness, autograd vs finite differences, brute-force causal weights, handoff correctness, window leakage, reproducibility of the loss trace with the same seed, L-BFGS reducing loss, invariants of the reference. Missing: a test that the 1536 checkpoint reproduces 6.97×10⁻⁵, and any ablation. I did not re-run the full suite (training tests are slow and the CPU was busy); do that before claiming "all tests pass".

**K15 [LOW]. Determinism claim.** The code states runs are deterministic. A seed fixes initialization, and a test checks same-seed reproducibility on a small case, but GPU float64 kernels and L-BFGS line searches can differ across devices. My CPU replicate (Part 14) is a direct test: the same configuration gave 1.48×10⁻⁴ on CPU against 6.97×10⁻⁵ on the saved GPU run, so it is not reproducible to better than about 2× across devices.

**K16 [LOW]. `train.py` is long and mixes training, logging and plotting.** Maintainability, not correctness.

**K17 [LOW]. Unused options in the codebase** (soft IC, end-state penalty, other activations, LHS sampler, "span" scaling) are tested but not part of the reported result. Fine; do not mention them as features.

**K18 [LOW]. The tolerance of L-BFGS is effectively disabled (E5).**

---

## Part 13. The paper and the poster: what to fix and what you will be asked

### Inconsistencies a careful referee will quote back to you
1. **Two meanings of 2.237** (Attack 8). Abstract and Table III: full-interval network. Method §A: a shared network trained sequentially. Fix: one meaning only.
2. **L-BFGS count** (K4).
3. **"Dense-grid"** (K6).
4. **Abstract grammar and tone:** "We trained a single network on the full interval that produce…", "To adress this long horizon problem…", "In contrast, these are described with the word configurations, not isolation of the effect of causal weighting" is hard to parse.
5. **Introduction claim:** "We use both concepts, we split the time period in equal 27 windows" and "essentially an closed orbit": use "nearly closed" consistently.
6. **Duplicate label.** Both the plateau figure and the pipeline figure use `\label{fig:method}`; one of the in-text references points to the wrong figure.7. **Stage rule sentence is garbled:** "When the saved minimum weight is above 0.99 or if there is an Adam update then the code will recalculate…". Rewrite: "After every Adam update the minimum causal weight is recomputed on the updated parameters; the stage ends when it exceeds 0.99 or after 4,000 updates."
8. **Table I vs text:** "Parameters 11,283 per window" is correct; the abstract says nothing about the total (304,641); put both in Table I.
9. **Discussion excuse about Wang et al.** (D11).
10. **Overclaim words:** "It is a notable result", "passes each of the tests", "overlap completely". Replace with numbers.
11. **Title** lacks the year (Attack 4).
12. **Typos:** adress, platue, stucking, aprroximately, weigths, seperately, measuere, intial, earliar, "Thats why use we used".
13. **Unsupported sentence:** "The Lorenz-system model … was obtained from a reduced flow model" is fine, but the problem is described as "a simple test of the convergence of a learned continuous-time function"; make it a benchmark statement.
14. **Reference list is thin** (H3).
15. **Run provenance:** state the commit, device, and that the headline run is one of four candidates (G10).
16. **Equation (the sum) uses the notation j < i but the algorithm says "sort times"**: say the points are sorted by time and window weights restart at each window.

### Poster claims to double-check
- Thesis strip: "lets small per-window networks track the reference where one full-interval network plateaus." This is causal language. Say "in our runs, the windowed configuration tracks … while the single network plateaued".
- "Recorded wall time 881 s vs 4,529 s": add "recorded; GPU model not recorded".
- The metric strip shows 6.97×10⁻⁵ to three figures; add "single seed".
- The anti-overclaim block is a good habit. Keep it prominent.
- `causal_weights.png` on the poster: be ready to explain what the plot shows and that the weights are near 1 for most of training (the gate), so the picture illustrates the mechanism, not a measured benefit.

### Likely paper-level scores by an honest referee (use as your self-assessment)
- **Soundness:** moderate. The run is real and reproducible from the weights, the diagnostics are careful, but attribution is absent.
- **Novelty:** low. Components are prior work on a different system.
- **Significance:** low to moderate. A non-chaotic benchmark with a closed form; no application.
- **Clarity:** moderate; improvable.
- **Likely decision at a main PINN/SciML venue:** reject or weak reject as it stands. At a student, workshop or regional venue: weak accept if attribution (ablation, seeds) is added and claims stay modest. If you want acceptance at a good venue, the ablation, multiple seeds, a stronger baseline and a second system or initial condition are what move the decision.

---

## Part 14. Diagnostic runs I performed for this document

These are extra runs I launched on the CPU of this machine to give you honest evidence for Attacks 1, 2, 6, 7 and for the determinism question. They are **not** in the paper, they write only to a scratch directory, and they do not touch your locked results. Treat them as preliminary: CPU float64, one or two seeds, same code as the repository.

**Setup.** Same repository code, `run_causal.py` configuration, CPU (2 threads per run), float64, one process per arm, written to a scratch directory only. The driver is saved as `scripts/ablation_arms.py`. Each run took 32 to 52 minutes. Metrics are computed like the paper's (13,265-point grid against DOP853). The Adam budget of the plain arm (1,287 steps per window) equals the mean Adam budget of the causal run (34,765 / 27).

| Arm | What changes | Combined RMSE | Max error norm | Residual MSE (collocation) | Adam steps | L-BFGS evals |
|---|---|---|---|---|---|---|
| Saved paper run (GPU, seed 0) | reference | **6.97×10⁻⁵** | 3.09×10⁻⁴ | 1.87×10⁻⁷ | 34,765 | 33,143 |
| A. causal, seed 0, CPU | same config, different device | 1.48×10⁻⁴ | 3.40×10⁻⁴ | 2.60×10⁻⁷ | 32,647 | 32,809 |
| B. causal, seed 1, CPU | different seed | 1.28×10⁻⁴ | 3.14×10⁻⁴ | 1.77×10⁻⁷ | 32,831 | 32,812 |
| C. plain residual, Adam 1,287 + L-BFGS | no causal weights | 5.87×10⁻⁴ | 1.98×10⁻³ | 1.52×10⁻⁶ | 34,749 | 33,659 |
| D. L-BFGS only (1 Adam step) | no Adam, no causal | 2.60×10⁻⁴ | 1.11×10⁻³ | 1.07×10⁻⁶ | 27 | 33,075 |
| E. causal Adam only (no L-BFGS) | no L-BFGS | 2.83×10⁻³ | 7.01×10⁻³ | 8.32×10⁻⁷ | 55,942 | 0 |

**What these runs say, honestly (one run per arm, so differences below about 2× are not significant):**

1. **The headline number does not reproduce on another device.** Re-running the identical configuration on CPU gives 1.48×10⁻⁴ (seed 0) and 1.28×10⁻⁴ (seed 1), about 2× worse than 6.97×10⁻⁵. The saved run sits at the favorable end of the spread. Together with the four earlier candidates (0.7 to 1.6×10⁻⁴), the defensible claim is "about 10⁻⁴". The residuals are similar (1.8 to 2.6×10⁻⁷) while the error differs by 2×, which again shows that residual does not fix the error. This answers K15: results are not bit-reproducible across devices, and they agree within a factor of about 2.
2. **Attack 1 (causal ablation): weakly in your favor, not conclusive.** Plain Adam plus L-BFGS at the same Adam budget (arm C) is 4× worse than the causal run on the same device (5.87×10⁻⁴ vs 1.48×10⁻⁴), with a 6× higher residual (1.5×10⁻⁶ vs 2.6×10⁻⁷). That is outside the seed-to-seed spread of 1.15×. So in this setting the staged causal Adam phase seems to hand L-BFGS a better start than plain Adam does. But arm D (no Adam at all) lands at 2.6×10⁻⁴, only 1.8× worse than causal and 2.3× better than arm C. So plain Adam at learning rate 10⁻³ seems to hurt, rather than causal weighting being uniquely good. The honest reading: "causal staging beat plain Adam at an equal budget in the runs we made; whether it beats a simpler schedule such as L-BFGS from the warm start is not established." Do not claim more.
3. **Attack 7 (L-BFGS does the work): confirmed.** Without L-BFGS (arm E) the causal Adam phase alone gives 2.83×10⁻³, 19× worse than with it, and its residual stays at 8.3×10⁻⁷, the gate level, even after 55,942 steps. The paper should say that the final accuracy requires the L-BFGS phase.
4. **Window 10 is hard in every arm, so it is not a seed accident.** L-BFGS end loss in window 10: 1.9×10⁻⁶ (A), 1.6×10⁻⁶ (B), 1.6×10⁻⁶ (D), 2.9×10⁻⁶ (C), and 1.9×10⁻⁶ in the saved run. Window 9 is stuck in the causal arms (about 6×10⁻⁷) but not in arm D (4×10⁻⁸). So t ≈ 4.9 to 5.4 is intrinsically difficult for this architecture and loss, which matches the paper's error peak at t ≈ 5. The other stuck windows of the saved run (0, 22, 25) were not stuck in all arms, so those look run-specific. Say "window 10 is consistently hard; why, we do not know".
5. **One-line story the evidence supports:** the pipeline reaches 7×10⁻⁵ to 1.5×10⁻⁴ across seeds and devices; it needs L-BFGS; it probably benefits from the causal stage over plain Adam; window 10 is a consistent weak spot.

**Limits of this evidence.** One run per arm for C, D and E. CPU, not the paper's GPU. The plain arm's 1,287 Adam steps were chosen to match the mean, not tuned. No hypothesis test. Do not put these numbers into the paper as results. Rerun on the GPU with at least three seeds per arm first. Do put the question and the plan in the paper's limitations.

---

## Part 15. Process, team and curveball questions

**M1 [HIGH]. Who did what? Can each author explain every part?**
Prepare a one-line ownership for each of the five students and the supervisor. Every student should be able to explain the trial solution, the residual, the causal weights, the handoff, and one result table. **[confirm with team]**

**M2 [MED]. Did you use AI tools to write the code or the paper?**
Prepare a truthful answer and the university's policy. Whatever the answer is, every author must be able to derive and run the core pieces without assistance; examiners test exactly this.

**M3 [MED]. What was hardest, and what failed first?**
The single network stuck at the equilibrium; sharing one network across windows lost earlier windows (qualitative, see Attack 8); the first causal runs had a gate that was checked before the update; telemetry bugs found in the audit; L-BFGS stalls in five windows.

**M4 [MED]. What would you do differently?**
Run the ablations and seeds before writing, record the device and commit, store the checkpoint, and choose configurations without looking at the reference error.

**M5 [MED]. Is 27 an accident?**
It is T/0.5 rounded up (C2). Say so.

**M6 [LOW]. Why Lorenz-1960 and not your own or an applied problem?**
Chosen as a clean benchmark (A1, A6). **[confirm with team: the supervisor's rationale.]**

**M7 [LOW]. What did you learn about PINNs?**
Residual alone is not accuracy; ordering and initialization matter on long horizons; optimizer choice (L-BFGS) mattered more than expected; reporting needs ablations and seeds from the start.

**M8 [HIGH]. "If I handed you a GPU for a day, what is the one experiment you would run?"**
The windowed run with the plain residual (no causal weights) under the same Adam and L-BFGS budget, on at least three seeds, next to the causal version on the same seeds. It tests Attack 1, 6 and 7 at once.

**M9 [HIGH]. "Convince me your headline number is not a fluke."**
It reproduces exactly from the saved weights (6.966965×10⁻⁵). The residual is stable off-grid (ratio 1.00). Across four configurations the RMSE stays within 0.7–1.6×10⁻⁴. The weak point is variation: re-running the same configuration on CPU gave 1.48×10⁻⁴ and 1.28×10⁻⁴ (two seeds), about 2× worse, so quote it as about 10⁻⁴ (Part 14).

**M10 [HIGH]. "Your result might be due to float64 and L-BFGS rather than anything you propose."**
That is a fair possibility. The single-network run also used float64 and L-BFGS (5,000 iterations) and failed, which shows precision and L-BFGS alone do not rescue a long-horizon single network; but they may be necessary components of the windowed success, and Part 14 tests this: L-BFGS from the warm start with almost no Adam gave 2.6×10⁻⁴, only 1.8× worse than the causal run, so L-BFGS carries much of the result.

**M11 [MED]. "What is the practical use of 304,641 parameters to solve a 3-variable ODE?"**
None, as stated. The point is the training method on a clean benchmark.

**M12 [MED]. "Why should I believe your reference?"** G12.

**M13 [LOW]. "Does your piecewise network satisfy the ODE at the joints?"**
Residuals there are ~1.1×10⁻³, higher than average (Attack 9).

**M14 [LOW]. "Plot the energy/invariant drift over time."**
It is in the run's figures (`invariant_drift`); drift ≤ 2.6×10⁻⁴.

**M15 [LOW]. "Can you run it live?"**
Don't promise. A full run takes minutes on a GPU and about an hour on CPU. Prepare a pre-recorded or a 3-window demo. `python -m pytest src/pinn/test_stress.py -q` is quick for some tests but includes training tests; test before the day.

---

## Part 16. Honest scorecard and action list

### What is strong
- A real, working pipeline; the headline number reproduces exactly from the saved weights.
- Strict hard-IC handoff with exact C⁰ continuity (jump ~10⁻¹⁶), checked by tests.
- No generalization gap between collocation points (off-grid probe, 10× mesh).
- The reference is excluded from loss and stopping, and the paper is open about uncontrolled comparisons and one seed.
- A careful audit trail in `docs/results/` and 71 tests.

### What is weak (in priority order)
1. **No causal ablation.** Title and abstract imply an attribution that the evidence cannot support.
2. **The baseline is a failed run, not a tuned competitor.** It underperforms a constant predictor.
3. **Single seed, run selection among four candidates, three significant figures.**
4. **Accuracy comes from L-BFGS in most windows; five windows hold ~80% of the residual and are unexplained.**
5. **Novelty is low.** Components are prior work; the system is non-chaotic and has a closed form.
6. **Reproducibility gaps:** weights not in the repo, GPU and versions not recorded.
7. **Writing:** inconsistent numbers, grammar, thin related work, duplicate figure label.

### Do these before the defense, ordered by value per hour
1. **Fix the paper's inconsistencies** (Part 13): the 2.237 sentence, L-BFGS count, "dense", the Wang excuse, the title year, typos, labels. About 2 hours.
2. **Commit the checkpoint** (K9) and record device, commit hash and package versions. About 30 minutes.
3. **Run the ablation and a second seed on GPU** (`python scripts/ablation_arms.py <arm>`; arms are listed in the file; writes to `runs/ablation/`). About 1 hour of compute.
4. **Add the off-grid probe result** and the per-window loss table to the paper or a backup slide. About 1 hour.
5. **Rewrite Related Work** with the missing citations after reading them. About half a day.
6. **Prepare a two-minute answer for each of the ten attacks** (Part 1), spoken aloud, by every author.

### If an examiner asks something not in this document
Say what you know, say what you did not test, and say what experiment would answer it. "We did not test that; here is how we would" is a strong answer. A confident wrong answer is the worst outcome.

---

## Appendix A. Derivations you may be asked to write on a board

**A.1 Trial solution and its derivative.** û(t) = u₀ + (t − t_k)N(t). Then û(t_k) = u₀, and û′(t) = N(t) + (t − t_k)N′(t), so û′(t_k) = N(t_k). The residual at the window start is N(t_k) − f(u₀).

**A.2 Residual and loss.** r(t) = û′(t) − f(û(t)); L_i = (1/3)Σ_c r_c(t_i)². Plain loss = (1/n)Σ L_i. Causal loss = (1/n)Σ w_i L_i, w_i = exp(−εΣ_{j<i}L_j), weights detached.

**A.3 Gate arithmetic.** w_last > 0.99 ⇔ εΣL < −ln 0.99 = 0.01005. With ε = 10, n = 1,535: mean L < 0.001005 / 1,535 = 6.5×10⁻⁷. With n = 512: 1.96×10⁻⁶.

**A.4 Parameter count.** 1→60: 60 + 60 = 120. Three 60→60 layers: 3 × (3,600 + 60) = 10,980. 60→3: 180 + 3 = 183. Total 11,283. Times 27 = 304,641.

**A.5 Window arithmetic.** Width = 13.26446 / 27 = 0.491276. Interior points per window 1,535 plus one endpoint = 1,536; total unique points 27 × 1,535 + 1 = 41,446; spacing ≈ 0.491276 / 1,535 = 3.2×10⁻⁴.

**A.6 Conserved quantities.** d(x²)/dt = −0.2xyz, d(y²)/dt = 3.2xyz, d(z²)/dt = −1.5xyz. So x² + y²/16 is constant (−0.2 + 3.2/16 = 0) and x² − (2/15)z² is constant (−0.2 + 1.5·(2/15) = 0). The orbit is the intersection of an elliptic and a hyperbolic cylinder.

**A.7 Error equation.** e = û − u. e′ = f(û) − f(u) + r. With ‖Df‖ ≤ L ≈ 2, ‖e(t)‖ ≤ ∫‖r‖ exp(L(t − s)) ds, vacuous for T = 13.

**A.8 Why a batch equals a loop.** Row i depends only on t_i, so ∂(mean_i ℓ_i)/∂θ = mean_i ∂ℓ_i/∂θ. For the causal loss the weights are constants, so the same holds with weights.

**A.9 L-BFGS in one paragraph.** Keeps the last m = 50 curvature pairs (s_k = Δθ, y_k = Δg), builds an implicit inverse-Hessian approximation by the two-loop recursion, takes a quasi-Newton step and chooses the step length with a strong-Wolfe line search (sufficient decrease and curvature conditions), which requires extra function evaluations (hence evaluations > iterations).

---

## Appendix B. Thirty-second answers for the top questions

1. **What did you do?** Split a 13.26-unit Lorenz-1960 orbit into 27 windows, trained one small PINN per window with a hard initial condition from the previous window's predicted endpoint, staged causal weights, then L-BFGS, and measured 7×10⁻⁵ RMSE against DOP853.
2. **Why windows?** A single network sat at an equilibrium with small residual and large error.
3. **Did causal weighting help?** We do not know; no isolated ablation. It shapes the first dozens of steps in each window.
4. **Why PINN when RK solves it in milliseconds?** It is a benchmark for the training method, not a solver replacement.
5. **Is it chaotic?** No, it is integrable and periodic.
6. **Is it new?** The components are prior work; we contribute the combined recorded study and diagnostics on this system.
7. **Single seed?** Yes. Four configurations ranged 0.7–1.6×10⁻⁴.
8. **Where is the error?** Five windows (0, 9, 10, 22, 25) hold ~80% of the residual; window 10 is worst.
9. **Is the derivative continuous?** Values to 10⁻¹⁶; slopes to ~10⁻³ (0.17% of |f|).
10. **What next?** Ablation and seeds first; then boundary conditions and a joint formulation as extensions.
