# Section blueprint

| Section | Role | Main judgment | Evidence IDs | Open boundary |
|---|---|---|---|---|
| Abstract | Compact result | One saved 27-window run has low measured error on this orbit | E1–E4, E7 | No claim of isolated causality |
| I. Introduction | Motivate exact test | Temporal ordering matters to the tested design; this paper reports one combined configuration | E9–E12, E1 | No universal long-horizon claim |
| II. Problem and related work | Define IVP and credit prior methods | ODE, conserved-form family, and borrowed PINN techniques | E5–E6, E9–E13 | Rounded endpoint, citation-level prior work |
| III. Method | Reproduction | Each window is trained, frozen, and handed to the next under specified loss | E1, E3, E5 | Algorithm is application of prior weighting |
| IV. Protocol | Comparison contract | Reference is logged for monitoring during training but excluded from objective and stopping; runs differ in multiple settings | E1–E2, E7, E14 | Hardware details absent |
| V. Results | Observations | Main run has low error; full-interval run has high error under its own settings | E2–E4, E7 | No causal attribution |
| VI. Discussion | Interpretation and limits | Exact value handoff does not imply slope continuity or general reliability | E3, E5–E8 | One seed and one orbit |
| VII. Conclusion | Answer question | Windowed configuration works well in this recorded test | E2, E7 | Matched ablations remain future work |
