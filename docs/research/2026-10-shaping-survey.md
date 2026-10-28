# Shaping the Lorenz-1960 PINN Pipeline: a Landscape Survey and a Proposal

*Working document — October 2026. Compiled from a five-perspective literature scout; every citation was verified against arXiv, publisher, or repository pages during compilation.*

---

## Abstract

The pipeline built in this repository trains physics-informed neural networks (PINNs) on the Lorenz-1960 system using batched per-window causal training, and beats a trapezoid-rollout baseline (1.7e-5 vs 8.2e-5 relative error on its headline run). The question addressed here: **what should this work become so that other researchers can use it?** Five literature perspectives (mainstream long-horizon training methods, documented failure modes, benchmarks and reproducibility, chaotic-dynamical-systems learning, and open-source tooling) converge on one specific, currently-unoccupied gap: **there is no benchmark and no systematic empirical study of causality-enforcing / windowed PINN training strategies on ordinary-differential-equation initial value problems**, and the evaluation norms used for chaotic systems in the PINN literature (pointwise error) diverge from the norms established in the dynamical-systems learning community (Lyapunov-scaled horizons, attractor statistics). The proposal is to shape the work as **a reproducible ODE-IVP benchmark and harness for long-horizon PINN training strategies**, with the existing batched-window causal scheme as one evaluated arm.

## Research questions

- **RQ1.** What is the state of the art (2021–2026) in long-horizon PINN training — causal weighting, time-marching, windowed approaches — and how solid is the evidence behind their claims?
- **RQ2.** Has anyone already done the systematic study "when does windowed/causal training help, on which problem classes"? What exactly is left open?
- **RQ3.** Where is the reproducibility/benchmark gap in this field, and could an open config-locked ODE pipeline fill it?

---

## Branch 1 — Mainstream long-horizon training methods

Three method families dominate, all responding to the same diagnosis: the standard PINN loss weights all collocation points equally, so the network fits late-time behavior before early time is resolved.

| Family | Representative works | Evidence base |
|---|---|---|
| Causal reweighting | Wang, Sankaran & Perdikaris (arXiv:2203.07404; CMAME 421:116813, 2024); Es'kin et al. loss modifications (arXiv:2304.02282; Dokl. Math. 2024) | Lorenz-63, Kuramoto-Sivashinsky, turbulent NS; claims "first successful PINN simulations" of chaotic systems |
| Time-marching / windowed | bc-PINN Mattey & Ghosh (CMAME 2022); HCS-PINN Roy & Castonguay (CMAME 2024); Ibc-PINN Lin & Chen (Chaos 2024); AT-PINN Chen et al. (2024); SI-PINN Huang et al. (PRE 2026) | Allen-Cahn, Cahn-Hilliard, KdV, NLS rogue waves, structural vibration |
| Curriculum / unified views | Krishnapriyan et al. (NeurIPS 2021); Penwarden et al. unified causal-sweeping framework (JCP 493:112464, 2023) | 1D convection/reaction/wave; systematic comparison of sweeping strategies |

Key facts that constrain any new contribution:

- **Evidence is overwhelmingly PDE-first.** ODE systems appear only as embedded benchmark cases — Lorenz-63 inside Wang et al., vibration ODEs inside AT-PINN. No dedicated ODE-focused method paper on windowed/causal training was found.
- **The field critiques itself.** SPINN (NeurIPS 2023) reports beating causal PINN on speed; Penwarden et al. show plain sequential time-marching is *not* the most accurate or scalable variant; Es'kin et al. show vanilla causal weights degrade on fine time grids; PINNacle finds causal training "problem-dependent rather than a universal fix."
- **The Wang/Perdikaris causal-training algorithm is patented by UPenn** (stated on the CausalPINNs repository); the reference code is non-commercial-licensed. The successor jaxpi2 (Apache-2.0, 2026) nonetheless ships causal training as a built-in. Any redistribution of the algorithm in a permissive library needs legal thought; academic evaluation does not.

## Branch 2 — Documented failure modes (the critics)

The strongest verified criticisms, in order of relevance to long-time ODE problems:

1. **Optimization failure is structural** (Krishnapriyan et al., NeurIPS 2021): even when the network can express the exact solution, the coupled loss landscape fails to optimize; trivially harder problems break vanilla PINNs.
2. **IVP loss landscapes are dominated by fixed points** (Rohrhofer et al., TMLR 2023): for initial value problems the loss landscape concentrates on fixed points of the RHS, biasing toward trivial solutions. This is an *ODE-specific* pathology, mostly studied on PDEs.
3. **PINNs can "cheat" on chaotic systems** (Grossmann et al., "How PINNs Cheat", arXiv:2405.17869): on the double pendulum, PINNs reach low loss and plausible trajectories while violating the equations — short-horizon fidelity certifies nothing.
4. **Weak baselines and reporting bias** (McGreivy & Hakim, Nature Machine Intelligence 2024): across ~80 ML-for-PDE papers, ~76% of claimed speedups vanish against tuned classical baselines; failures are under-reported.
5. **Pointwise prediction of chaotic systems is bounded by Lyapunov time** (Branch 4 consensus) — yet PINN papers report pointwise L2 on chaotic benchmarks, a metric that saturates within a few Lyapunov times regardless of method quality.

Critic-verdict on time-marching/causal specifically: these are *partial, cost-amplifying mitigations*, not solutions (Penwarden 2023; Mojgani et al. arXiv:2205.02902). No dedicated negative-results paper on causal/windowed training for chaotic ODEs exists.

## Branch 3 — Benchmarks and reproducibility

| Suite | Scope | ODE-IVP track? |
|---|---|---|
| PINNacle (NeurIPS 2024 D&B) | 10+ PINN variants × 20+ PDEs, seeded | **No** |
| PDEBench (NeurIPS 2022 D&B) | Data-driven neural solvers vs classical, PDEs | **No** |
| The Well (NeurIPS 2024 D&B) | 16 PDE simulation datasets | **No** |
| PDEarena (2022) | Neural operators on fluid PDEs | **No** |
| Grossmann et al. (IMA JAM 2024) | PINN vs FEM head-to-head, steady problems | **No** |

Verified finding: **no community benchmark for PINNs on ODE initial value problems exists.** The closest artifacts: Baty's stiff-ODE recipes (arXiv:2304.08289, ad hoc problem set), DeepXDE's example suite (illustrative, no fixed metrics/seeds), PinnDE (library, not benchmark). Meanwhile the reproducibility critique is quantified (McGreivy & Hakim) and the fragility mechanism is understood (ill-conditioned loss landscapes — Rathore et al., ICML 2024; NTK spectral imbalance — Wang, Yu & Perdikaris, JCP 2022), which together demand multi-seed protocols that no ODE study currently applies.

## Branch 4 — Chaotic dynamical systems (adjacent field)

- Consensus across reservoir computing (Pathak et al., Chaos 2017), RNN/LSTM forecasting (Vlachas et al., Neural Networks 2020), distributed RNNs (Linot & Graham, PNAS 2020), and weather emulation (GraphCast, Science 2023; FourCastNet): **pointwise skill saturates at O(1–10) Lyapunov times**; the meaningful long-horizon target is **attractor statistics** (invariant measure, spectra), which learned models can reproduce far beyond the trajectory horizon.
- Standard chaotic test cases are **Lorenz-63 and Lorenz-96 — not Lorenz-1960**. The Lorenz-1960 maximum-simplification barotropic vorticity model (the k/l quadratic system actually solved in this repository) is essentially absent from the ML literature. Consequence: comparisons to prior work require adding Lorenz-63; Lorenz-1960 remains a legitimately fresh problem instance.
- Nobody in the PINN literature reports Lyapunov-scaled skill or attractor statistics for PINN benchmarks — the two communities use incompatible evaluation norms.

## Branch 5 — Tooling and ecosystem

- DeepXDE (4.4k★, SIAM Review 2021): the standard, but time is just an input dimension — no causal, no windowed mode; loss-weight tuning and diagnostics are user problems (its own issue tracker confirms).
- PhysicsNeMo (NVIDIA), PINA (SISSA), NeuroDiffEq, NeuralPDE.jl, PinnDE: none ship a built-in, evaluated windowed/causal long-horizon mode. ODE/IVP support is second-class everywhere except PinnDE (small community).
- jaxpi / jaxpi2 (Wang/Perdikaris lineage): the only frameworks with built-in causal training — research-group code, JAX, GPU-only, PDE-benchmark-focused.
- Recurring practitioner complaints: seed/hyperparameter fragility, no ablation/experiment-tracking layer, no standardized diagnostics (residual-vs-time, error-vs-horizon), no plotting layer.

---

## Cross-branch synthesis: the convergent gap

Three independent literatures point at the same empty square:

1. **Method literature (Branch 1):** causal/windowed training claims rest on PDE demonstrations; ODEs are an afterthought, and the variants were never compared on equal footing on ODE systems.
2. **Critics (Branches 2–3):** the field's claims collapse without multi-seed protocols, strong baselines, and honest metrics — precisely the discipline this repository already practices (config-locked runs, ablation arms, checkpoints, generalization-gap probes, run index).
3. **Dynamical systems (Branch 4):** the correct metrics for long-horizon chaotic evaluation (Lyapunov-scaled skill, attractor statistics) exist and are standard one field over, but have never been applied to PINN training-strategy comparison.

**The gap, stated in one sentence:** nobody has produced a rigorous, multi-seed, multi-method benchmark and empirical characterization of causality-enforcing PINN training on ODE initial value problems, evaluated with dynamical-systems-appropriate metrics.

## Open problems this work could own

1. The ODE-IVP benchmark itself (problem classes: linear, oscillatory, periodic, stiff, chaotic; dimensions 1–4+).
2. The "when does it help" characterization: vanilla vs causal weighting vs sequential time-marching vs batched-window causal (the scheme built here) vs curriculum — across problem classes and horizon lengths.
3. Metric bridging: pointwise error at N Lyapunov times + attractor-statistics fidelity, reported side by side, with the honest statement that pointwise error is not the right target beyond a few Lyapunov times.
4. The stiff-ODE frontier: Baty (2023) showed vanilla PINNs fail; whether windowed-causal extends the feasible stiffness range is open and would be a genuinely new empirical result either way.
5. The IVP fixed-point pathology (Rohrhofer 2023): does windowed causal training avoid or worsen fixed-point collapse? Nobody has connected these two literatures.

## Conclusion (answers to the RQs)

- **RQ1:** The SOTA is three families of causality-enforcing training (causal weights, time-marching/windowing, curriculum), empirically validated almost exclusively on PDEs, with credible "makes feasible what was infeasible" claims but weak guarantees, known hyperparameter fragility, and self-documented variant weaknesses.
- **RQ2:** No. Systematic comparisons exist only for PDEs (Penwarden 2023; PINNacle 2024), with inconclusive problem-dependent findings. On ODEs: nothing systematic exists. The gap is real as of October 2026.
- **RQ3:** The reproducibility gap is quantified (McGreivy & Hakim: 76% of speedup claims vanish vs strong baselines) and no ODE benchmark exists. A config-locked, multi-seed, baseline-inclusive ODE harness fills a documented hole; the repository's existing discipline is most of the way there technically.

---

## The shaped problem statement (proposal)

> **A reproducible benchmark and empirical characterization of causality-enforcing training strategies for physics-informed neural networks on initial value problems.** We present an open, config-locked harness that evaluates vanilla, causally-weighted, sequential time-marching, and batched-window causal PINN training across a spectrum of ODE problem classes — linear, oscillatory, periodic, stiff, and chaotic — under multi-seed protocols, against strong classical integration baselines, and with metrics aligned to dynamical-systems theory: pointwise error at Lyapunov-scaled horizons and attractor-statistics fidelity.

What this shape preserves from the current work: the batched-window causal scheme (as a first-class method arm), the Lorenz-1960 system (as a problem instance), and the entire reproducibility discipline (configs, checkpoints, sweeps, run index, generalization-gap probes) — which becomes the harness's selling point rather than incidental engineering.

What must be added: the `ProblemSpec`/registry generalization (dim-agnostic network, generic reference solver routing — the refactor already scoped in the codebase audit), 6–10 benchmark problems including Lorenz-63 (for comparability with Wang et al.), a Van der Pol / Lotka-Volterra (oscillatory), a Robertson-type stiff system, and metric machinery (Lyapunov-time scaling, attractor statistics).

What must be honestly stated in the work itself: pointwise accuracy on chaotic systems is Lyapunov-bounded and the contribution is feasibility + comparative characterization, not superior long-horizon pointwise accuracy; causal weighting is patented by UPenn (academic evaluation unaffected, redistribution needs care); results may be "it depends" — which, given PINNacle's analogous PDE finding, is itself the honest and useful answer for ODEs.

## Reference list (verified)

1. Krishnapriyan, Gholami, Zhe, Kirby, Mahoney. *Characterizing possible failure modes in PINNs.* NeurIPS 2021. arXiv:2109.01050
2. Wang, Yu, Perdikaris. *When and why PINNs fail to train: a NTK perspective.* JCP 449:110768, 2022. arXiv:2007.14527
3. Wang, Teng, Perdikaris. *Understanding and mitigating gradient flow pathologies in PINNs.* SIAM J. Sci. Comput. 43(5), 2021. arXiv:2001.04536
4. Wang, Sankaran, Perdikaris. *Respecting causality for training PINNs.* CMAME 421:116813, 2024. arXiv:2203.07404
5. Mattey, Ghosh. *bc-PINN.* CMAME 393:114766, 2022. arXiv:2106.07606
6. Penwarden, Jagtap, Zhe, Karniadakis, Kirby. *Unified scalable framework for causal sweeping strategies.* JCP 493:112464, 2023. arXiv:2302.14227
7. Roy, Castonguay. *HCS-PINN: exact temporal continuity.* CMAME 2024. arXiv:2403.03223
8. Lin, Chen. *Ibc-PINN.* Chaos 34:033139, 2024. arXiv:2312.06715
9. Es'kin et al. *Optimal loss function under respecting causality.* arXiv:2304.02282; Dokl. Math. 2024
10. Cho et al. *SPINN.* NeurIPS 2023. arXiv:2308.08468
11. Huang et al. *SI-PINN.* Phys. Rev. E 114:025304, 2026
12. Rohrhofer, Posch, Gößnitzer, Geiger. *On the role of fixed points of dynamical systems in training PINNs.* TMLR 2023. arXiv:2203.13648
13. Baty. *Solving stiff ODEs using PINNs: simple recipes.* 2023. arXiv:2304.08289
14. Chuang, Barba. *Predictive limitations of PINNs in vortex shedding.* 2023. arXiv:2306.00230
15. Grossmann, Komorowska, Latz, Schönlieb. *Can PINNs beat the FEM?* IMA J. Appl. Math. 89(1), 2024. arXiv:2302.04107
16. Grossmann et al. *How PINNs Cheat: predicting chaotic motion of a double pendulum.* arXiv:2405.17869
17. McGreivy, Hakim. *Weak baselines and reporting biases…* Nature Machine Intelligence 6(10), 2024. arXiv:2407.07218
18. Mojgani, Balajewicz, Hassanzadeh. *Lagrangian PINNs.* 2022. arXiv:2205.02902
19. Hao et al. *PINNacle.* NeurIPS 2024 D&B. arXiv:2306.08827
20. Takamoto et al. *PDEBench.* NeurIPS 2022 D&B. arXiv:2210.07182
21. Ohana et al. *The Well.* NeurIPS 2024 D&B. arXiv:2412.00568
22. Gupta, Brandstetter. *PDEarena.* arXiv:2209.15616
23. Rathore, Lei, Frangella, Lu, Udell. *Challenges in training PINNs: a loss landscape perspective.* ICML 2024. arXiv:2402.01868
24. Lu, Meng, Mao, Karniadakis. *DeepXDE.* SIAM Review 63(1), 2021. arXiv:1907.04502
25. Matthews. *PinnDE.* arXiv:2408.10011; github.com/JB55Matthews/PinnDE
26. Pathak, Lu, Hunt, Girvan, Ott. *Using machine learning to replicate chaotic attractors.* Chaos 27:121102, 2017
27. Vlachas et al. *Backpropagation algorithms and reservoir computing…* Neural Networks 126, 2020. arXiv:1910.05266
28. Linot, Graham. *Data-driven prediction of large-scale spatiotemporal chaos.* PNAS 117(39), 2020
29. Dubois, Gomez, Planckaert, Perret. *Data-driven predictions of the Lorenz system.* Physica D 408:132495, 2020
30. Lam et al. *Learning skillful medium-range global weather forecasting (GraphCast).* Science 382, 2023
31. Lippe, Veeling, Perdikaris, Turner, Brandstetter. *PDE-Refiner.* NeurIPS 2023
32. Miñoza. *SPIKE: sparse Koopman regularization for PINNs.* CPAL 2026, PMLR 328
33. jaxpi2 — github.com/sifanexisted/jaxpi2 (Apache-2.0); CausalPINNs — github.com/PredictiveIntelligenceLab/CausalPINNs (non-commercial; UPenn patent notice)
