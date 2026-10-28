# Defense Poster Brief — A0 Portrait

Built with the Skill2Poster pipeline (`paper-poster-imagegen` → `academic-poster-builder`).
Source of truth: `writings/conference-paper/main.tex` (final paper) and
`runs/causal-window/candidate-1536pt/` (final saved run). Every number below is
verified against `metrics.csv` / `run_summary.csv` of that run.

## 1. Header

- Kicker: Final Year Design Project Defense · Physics-Informed Neural Networks
- Title: **Sequential Causal Training of Windowed PINNs for the Lorenz ODE System**
- Authors: Ikramul Hasan Moral, Md. Abu Bakar, Samiur Rahman Omlan, Md. Touhidul Islam,
  Fariha Islam, Muhammad Nomani Kabir
- Affiliation: Department of Computer Science and Engineering, United International
  University, Dhaka, Bangladesh
- Right: QR → https://github.com/ihmorol/lorenz1960-pinn ("Code & run artifacts")
- Left: UIU typographic wordmark (no logo asset available offline; none hand-drawn)

## 2. Metric strip (4 parallel cards)

| Label | Value | Comparison line |
|---|---|---|
| Combined state RMSE | 6.97×10⁻⁵ | vs DOP853 on 13,265 times · single-network run: 2.237 |
| ODE residual MSE | 1.87×10⁻⁷ | dense evaluation grid · collocation grid: 1.87×10⁻⁷ |
| Max state-error norm | 3.09×10⁻⁴ | across the full orbit t ∈ [0, 13.26446] |
| Recorded wall time | 881 s | single-network run: 4,529 s (CUDA) |

## 3. Thesis strip (one sentence)

"Splitting the orbit into 27 hard-IC windows, warm-starting each from the previous
endpoint, and training with causal residual weights lets small per-window networks
track the reference where one full-interval network plateaus."

## 4. Section plan (3 columns, center wider)

- Left — 1. Problem: the Lorenz-1960 orbit (IVP equations, nearly-closed-orbit note,
  phase portraits). 2. Why one full-interval network is not enough (plateau figure).
  3. Windowed design: hard IC + endpoint handoff (trial-solution and handoff equations).
- Center — 4. Training pipeline (paper `method.png`, full width). 5. Causal residual
  loss (per-point loss, causal weights, causal loss; four ε stages; L-BFGS polish).
  6. Causal weights in action (`causal_weights.png`).
- Right — 7. Trajectory accuracy vs DOP853 (`solution_vs_reference.png` + per-state
  error table). 8. Error diagnostics (`error_analysis.png`). 9. Two saved runs
  (descriptive comparison table). 10. Scope & takeaways.

## 5. Figures used (original paper/run assets only)

| Asset | Source |
|---|---|
| `method.png` | writings/conference-paper/figures |
| `platue-results.png` | writings/conference-paper/figures |
| `solution_vs_reference.png`, `error_analysis.png`, `causal_weights.png`, `phase_portraits.png` | runs/causal-window/candidate-1536pt/.../figures |

## 6. Anti-overclaim rules applied

- Comparison of the two runs is called **descriptive**, never an ablation of causal
  weighting (configurations differ simultaneously).
- Single seed, single initial state, single coefficient set — stated in Scope.
- Window values are exactly continuous; **derivatives may jump** at interfaces — stated.
- 2 of 108 causal stages hit the 4,000-step cap — stated.
- DOP853 reference is monitoring-only (never in loss, gradients, or stopping) — stated.
- Terminology follows the paper: "nearly closed orbit" (not "exact period"),
  "windowed PINN", "hard trial solution", "causal weights", "endpoint handoff".

## 7. Design reference mockup (paper-poster-imagegen step)

The Skill2Poster flow calls for an AI image-generation reference mockup before the
HTML rebuild. The `imagegen` built-in tool is not available in this harness and the
CLI fallback requires `OPENAI_API_KEY` (not set), so that step is **deferred**, and
the layout target from the skill's template is applied directly. If a mockup is
wanted later, use this prepared prompt (imagegen, A0 portrait):

```
Create a high-fidelity A0 portrait academic defense poster reference mockup.
Style: clean white and very light blue panels, deep navy section headers, thin
blue borders, restrained green result accents, moderate-weight sans-serif, dense
but readable, aligned three-column body with a wider center column.
Header: left "UIU · United International University, Dept. of CSE" wordmark,
center title "Sequential Causal Training of Windowed PINNs for the Lorenz ODE
System" with authors and affiliation, right one large QR block captioned "Code".
Metric strip: four parallel cards with icons — "Combined state RMSE 6.97×10⁻⁵",
"ODE residual MSE 1.87×10⁻⁷", "Max state-error norm 3.09×10⁻⁴", "Recorded wall
time 881 s" — each with a one-line comparison.
Thesis strip: one centered sentence below the metrics.
Left column: 1 Problem (Lorenz-1960 IVP equations, phase portraits figure),
2 Why one full-interval network fails (training plateau figure), 3 Windowed design
(hard trial solution + endpoint handoff equations).
Center column: 4 Training pipeline (tall flowchart figure), 5 Causal residual loss
(equation cards), 6 Causal weights plot.
Right column: 7 Trajectory accuracy vs DOP853 (solution + signed errors figure,
per-state error table), 8 Error diagnostics (4-panel figure), 9 Two saved runs
comparison table, 10 Scope & takeaways.
Constraints: place the provided figure thumbnails as framed images, do not invent
charts or numbers, no decorative footer, column bottoms aligned, readable text.
```

## 8. Build & QA

- CSS canvas: 3179 × 4494 px (A0 portrait @96dpi) — verified by Playwright.
- PNG export: 150 dpi → ≈ 4967 × 7022 px (deviceScaleFactor = 150/96).
- PDF export: 841 mm × 1189 mm, printBackground, margin 0.
- QA: no panel overflow, column bottoms aligned, poster bottom == page height.
