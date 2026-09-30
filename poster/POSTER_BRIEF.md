# Defense poster brief

## Source of truth

- Manuscript: `E:/University/FYDP/lorenz1960-pinn/writings/conference-paper/main.tex`.
- Final saved run: `runs/causal-window/candidate-1536pt/4x60_f64_unit_win27_causal_warm_f1cbeebd/`.
- Saved full-interval comparison: `runs/4x60_f64_unit/`.
- Follow manuscript terminology and its descriptive-comparison caveat. The interval is a **nearly closed orbit**, not an exact period.

## Header

- **Title:** Sequential Causal Training of Windowed PINNs for the Lorenz ODE System (typographic capitalization normalized from manuscript title).
- **Authors:** Ikramul Hasan Moral, Md. Abu Bakar, Samiur Rahman Omlan, Md. Touhidul Islam, Fariha Islam, Muhammad Nomani Kabir.
- **Affiliation:** Department of Computer Science and Engineering, United International University, Dhaka, Bangladesh.
- No event, conference, date, or QR claim; the user requested manuscript details only.

## Four measured cards

1. 27 equal time windows over `t ∈ [0, 13.26446]`.
2. `6.97 × 10⁻⁵` combined state RMSE on 13,265 evaluation times.
3. `3.09 × 10⁻⁴` maximum state-error norm.
4. `1.87 × 10⁻⁷` dense-grid residual MSE.

## Thesis strip

On one saved, single-seed Lorenz initial-value problem, the 27-window PINN reproduced the nearly closed orbit with low state error against a DOP853 reference.

## Reading order

1. **Problem:** Lorenz ODE, initial condition, nearly closed interval.
2. **Full-interval observation:** the saved single network plateaued away from the reference trajectory.
3. **Windowed method:** separate network per window, hard initial condition, endpoint and weight handoff; causal Adam stages followed by unweighted L-BFGS.
4. **Trajectory evidence:** original manuscript and candidate run figures, with readable captions.
5. **Measured results:** exact grid and comparison table, qualified as descriptive.
6. **Limits:** one seed/initial state/coefficient set; changed configuration factors; interface derivatives may jump; handoff can propagate error.

## Visual assets

- Manuscript `figures/platue-results.png`: full-interval failure observation.
- Candidate `results.png`: trajectory, signed error, and training-loss evidence.
- Candidate `figures/causal_weights.png`, `figures/joint_continuity.png`, and `figures/phase_portraits.pdf`: mechanism/continuity/shape evidence as space permits.
- The user rejected the manuscript's `method.pdf` and `trajectory_checks.pdf` visuals. The poster instead uses a source-checked HTML method flow and candidate-run evidence.

## Format and constraints

- A0 portrait, exactly `841 mm × 1189 mm`; light theme with navy, teal, and restrained amber.
- HTML source and print PDF; 150 dpi or better PNG preview. Large labels and plots, short text.
- Image-generation output is a layout reference only. All final text, figures, equations, and numbers come from the manuscript or saved run.
- Avoid isolated causal-effect language, exact-period claims, derivative-continuity claims, unsupported generalization, small plot text, decorative branding, and filler panels.

- Reference mockup review: its Lorenz-63 equations, generic labels, and unsupported wording were rejected. Only its light palette and panel hierarchy informed the final HTML.
