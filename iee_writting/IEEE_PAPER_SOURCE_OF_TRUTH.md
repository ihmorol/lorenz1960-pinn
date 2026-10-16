# IEEE Conference Paper Source of Truth

## Lorenz-1960 Physics-Informed Neural Network

**Status:** Writing and planning guide  
**Scope:** IEEE conference paper only  
**Primary implementation source:** `iee/4.implementation (1).tex`  
**Formatting reference:** `iee/IEEE_Conference_Template/main.md`  
**Rule:** This document is the shared source of truth for everyone writing or reviewing the paper.

---

## 1. Purpose of This Document

This document tells the full team how to convert the existing implementation work into a concise IEEE conference paper.

The supervisor's requested writing style is:

- simple and story-driven;
- direct and technically precise;
- focused on the implemented experiment;
- limited to necessary technical terms;
- free from broad, thesis-style explanations;
- supported by measured evidence;
- honest about the scope and limitations of the experiment.

This is a writing guide, not a new research proposal. The team must describe what has already been implemented and evaluated. The team must not silently expand the claims beyond the evidence in the implementation file.

The implementation file and this guide are the basis for the manuscript. Other workspace documents may contain useful background, but they must not replace the implementation file as the source for the method, configuration, numerical results, or claims.

---

## 2. Non-Negotiable Scope

### 2.1 What the paper is about

The paper is about a **physics-informed neural network (PINN) that approximates one fixed Lorenz-1960 trajectory** for one initial-value problem.

The implemented task is:

\[
u(t) = (x(t), y(t), z(t)),
\qquad t \in [0,1],
\]

with initial state:

\[
u(0) = (0.5, 0.75, 1.0).
\]

The model receives time as input and predicts the three state variables.

### 2.2 What the paper is not about

The paper is not allowed to describe the implementation as any of the following unless new experiments are actually performed and documented:

- a general-purpose Lorenz solver;
- a solver validated across many initial conditions;
- a solver validated across many values of \(k\) and \(l\);
- a long-horizon chaotic-system solver;
- a neural operator;
- an operator-learning model;
- a state-of-the-art method;
- a universally superior alternative to numerical integration;
- a proof of PINN convergence;
- a proof of global equation satisfaction;
- a real-time solver;
- a computationally cheaper solver;
- a method that generalizes to unseen initial conditions.

The code contains configurable fields, but configuration flexibility is not the same as demonstrated experimental generalization. The reported experiment uses one configuration, one initial condition, one parameter setting, one time interval, and one random seed.

### 2.3 What must not be changed while writing

During manuscript preparation, do not change:

- `iee/4.implementation (1).tex`;
- the numerical implementation;
- the governing equations used in the reported experiment;
- the reported metric values;
- the reference-solver settings;
- the IEEE template structure or class settings.

If a numerical inconsistency is discovered, record it as a question for verification. Do not silently modify the result or rewrite the method to make the paper appear stronger.

---

## 3. Verified Implementation Facts

Every writer must use the following facts consistently.

### 3.1 Governing system

The implementation uses the Lorenz-1960 system in the reduced form:

\[
\frac{dx}{dt} = -0.10 yz,
\]

\[
\frac{dy}{dt} = 1.60 xz,
\]

\[
\frac{dz}{dt} = -0.75 xy.
\]

The coefficients come from \(k=2\) and \(l=1\):

\[
c_x=-0.10, \qquad c_y=1.60, \qquad c_z=-0.75.
\]

The implementation derives the coefficients from the configuration rather than duplicating the reduced constants in the network code.

### 3.2 Initial condition and time interval

Use exactly:

\[
x(0)=0.5, \qquad y(0)=0.75, \qquad z(0)=1.0,
\]

and:

\[
t \in [0,1].
\]

Do not change decimal formatting between sections. Use `1.0` when presenting the state value in text or tables if clarity requires it.

### 3.3 Neural-network architecture

The reported model is a fully connected multilayer perceptron:

- input dimension: 1, scalar time \(t\);
- hidden layers: 4;
- hidden units per layer: 60;
- hidden activation: `tanh`;
- output dimension: 3, corresponding to \((x,y,z)\);
- output layer: linear;
- initialization: Xavier-uniform weights and zero biases;
- trainable parameter count: verify the exact value before final submission because the implementation text contains both 11,283 and 11,263 in different places.

### 3.4 Hard initial-condition enforcement

The reported experiment uses the trial solution:

\[
u_T(t)=u_0+g(t)\mathcal{N}(t;\theta),
\]

where:

\[
g(t)=\frac{t-t_0}{t_f-t_0}.
\]

For this experiment, \(t_0=0\) and \(t_f=1\), so \(g(t)=t\). Therefore:

\[
u_T(0)=u_0
\]

for every value of the trainable parameters \(\theta\).

The initial condition is therefore satisfied by construction. It is not learned through a penalty term in the reported run.

### 3.5 Physics residual

The residual is:

\[
r(t)=\frac{du_T}{dt}-f(u_T(t)).
\]

The time derivative is computed with automatic differentiation through PyTorch. The residual is evaluated component-wise for the three Lorenz-1960 equations.

### 3.6 Training loss

The reported loss is the mean squared residual over the collocation points:

\[
\mathcal{L}(\theta)=
\frac{1}{3N_c}
\sum_{i=1}^{N_c}\|r(t_i)\|_2^2.
\]

The reported training run uses physics-only training. No solution values from the reference trajectory are included in this loss.

### 3.7 Collocation and optimization settings

Use these settings exactly unless the team performs a new verified run:

| Setting | Reported value |
|---|---|
| Collocation points | 3,000 |
| Sampling method | Latin hypercube sampling |
| Training mode | Full batch |
| Optimizer | Adam |
| Training iterations | 20,000 |
| Initial learning rate | \(10^{-3}\) |
| Final learning rate | \(10^{-4}\) |
| Learning-rate schedule | Linear decay |
| Numeric precision | float32 / single precision |
| Random seed | 0 |
| Initial-condition treatment | Hard constraint |
| L-BFGS stage | Disabled for reported results |

The collocation points are fixed before training. They are not the same as the separate evaluation grid.

### 3.8 Reference solution

The paper must call this a **high-accuracy numerical reference trajectory**, not an analytical ground truth.

The reported reference setup is:

- solver: SciPy `solve_ivp` with DOP853;
- relative tolerance: \(10^{-10}\);
- absolute tolerance: \(10^{-12}\);
- evaluation/reference grid: 1,001 uniformly spaced points over \([0,1]\).

An independent RK4 calculation with fixed step \(h=10^{-3}\) was used as a consistency check. The implementation reports:

- RMSE difference between the two reference trajectories: \(1.35\times10^{-11}\);
- maximum difference: \(6.06\times10^{-11}\).

The DOP853 trajectory is used for post-training evaluation only. It is not used as a training signal.

### 3.9 Evaluation protocol

The model is evaluated after training as follows:

1. Train for the full 20,000 Adam iterations.
2. Freeze the final network parameters.
3. Evaluate the model on a separate 1,001-point uniform grid.
4. Compare the predicted trajectory with the DOP853 reference trajectory.
5. Compute component-wise and combined error metrics.
6. Inspect the pointwise residual over the interval.

The evaluation grid must not be described as the training grid. The collocation set and evaluation grid have different purposes.

### 3.10 Reported numerical results

The current implementation reports the following values:

| State | MAE | RMSE | Maximum absolute error |
|---|---:|---:|---:|
| \(x\) | \(5.20\times10^{-6}\) | \(5.77\times10^{-6}\) | \(9.24\times10^{-6}\) |
| \(y\) | \(1.12\times10^{-5}\) | \(1.36\times10^{-5}\) | \(2.56\times10^{-5}\) |
| \(z\) | \(8.27\times10^{-6}\) | \(9.04\times10^{-6}\) | \(1.25\times10^{-5}\) |
| Combined | \(1.66\times10^{-5}\) | \(1.73\times10^{-5}\) | \(2.60\times10^{-5}\) |

The reported loss decreases from:

\[
2.17\times10^{-1}
\quad\text{to}\quad
6.82\times10^{-9}.
\]

The implementation reports the trajectory endpoint as approximately:

\[
(x,y,z)=(0.412,1.359,0.631)
\]

at \(t=1\).

Before final submission, the team must verify that the table values, parameter count, and endpoint values match the current generated result files. The paper must not contain conflicting values.

---

## 4. Required IEEE Paper Structure

Use the following section order. Do not copy the thesis chapter structure directly into the conference paper.

```text
Title
Authors and affiliations
Abstract
Index Terms

I. Introduction
II. Problem Formulation and Related Work
III. Physics-Informed Neural Network Method
IV. Experimental Setup
V. Results
VI. Discussion and Limitations
VII. Conclusion

Acknowledgment, if required
References
```

The IEEE template automatically numbers section headings. Do not manually number headings in the source text.

### Section-length priority

The exact page limit depends on the target IEEE conference. The team must check the official call for papers and author instructions. Until that limit is confirmed, write for a compact two-column paper and prioritize the following order:

1. Introduction and contribution clarity;
2. method reproducibility;
3. evaluation protocol;
4. numerical results;
5. limitations and honest comparison.

Remove repeated explanations before removing the equations, evaluation protocol, or main results.

---

## 5. Detailed Writing Instructions by Section

## Title

The title must identify the actual method and problem.

Recommended direction:

> A Physics-Informed Neural Network with Hard Initial-Condition Enforcement for the Lorenz-1960 System

The title must not claim generality, superiority, or operator learning.

Avoid titles containing:

- “universal solver”;
- “generalized framework”;
- “state-of-the-art”;
- “optimal”;
- “real-time”;
- “chaos prediction” unless the experiment actually studies chaotic prediction;
- “operator network,” because the reported model is a single-trajectory PINN.

## Abstract

Write one self-contained paragraph. Target approximately 150 to 200 words unless the conference specifies another limit.

The abstract must include:

1. the problem: approximating a Lorenz-1960 initial-value problem;
2. the method: a time-to-state PINN;
3. the key design: hard initial-condition trial solution;
4. the training: physics residual, 3,000 collocation points, 20,000 Adam iterations;
5. the evaluation: separate 1,001-point grid and DOP853 reference;
6. the main result: combined RMSE \(1.73\times10^{-5}\);
7. the scope: one fixed trajectory over \([0,1]\).

Do not put citations in the abstract unless the venue specifically permits or requires them. Do not describe implementation modules in the abstract.

## Index Terms

Use four to six terms. Recommended terms:

- Physics-informed neural network;
- Lorenz-1960 system;
- automatic differentiation;
- collocation method;
- initial-value problem;
- neural differential equation.

Define “physics-informed neural network” at first use in the main text, even if it appears in the abstract.

## I. Introduction

The introduction should be approximately three to five focused paragraphs.

### Paragraph 1: Motivation

Explain that nonlinear ordinary differential equations can be solved numerically, while PINNs provide a differentiable neural approximation constrained by the governing equations.

Start directly with the research problem. Do not begin with a broad history of artificial intelligence.

### Paragraph 2: Technical difficulty

Explain the two requirements:

- the predicted trajectory must satisfy the differential equations;
- the predicted trajectory must satisfy the initial condition.

Introduce the hard initial-condition construction as a way to satisfy the second requirement analytically.

### Paragraph 3: Prior work and gap

Discuss only the prior work needed to position this experiment:

- physics-informed neural networks;
- hard initial-condition trial solutions;
- the reference Lorenz-1960 study;
- the distinction between operator learning and single-trajectory approximation.

Do not create a broad literature survey.

### Paragraph 4: Objective and contributions

State the objective directly:

> This paper implements and evaluates a physics-informed neural network for one fixed Lorenz-1960 trajectory using a hard initial-condition constraint and a physics-only residual loss.

Use a short contribution paragraph or a short numbered list with only evidence-supported contributions:

1. A fully specified hard-constrained PINN formulation for the selected Lorenz-1960 initial-value problem.
2. A reproducible physics-only training protocol using automatic differentiation and Latin-hypercube collocation.
3. Independent quantitative evaluation against a high-accuracy DOP853 reference trajectory.
4. Component-wise and combined error reporting, together with training-loss and pointwise-residual analysis.

Do not call these contributions novel unless the literature review confirms the novelty. “We implement,” “we evaluate,” and “we report” are safer than “we introduce” when the method combines established techniques.

## II. Problem Formulation and Related Work

This section should define the exact task and briefly position it.

### A. Lorenz-1960 initial-value problem

Present the reduced equations, initial condition, interval, and coefficient origin. Define every symbol before or immediately after it is used.

The general coefficient formula may be shortened. The conference paper does not need the full source-code traceability discussion from the implementation chapter.

### B. PINNs and initial-condition enforcement

Explain that PINNs minimize a differential-equation residual at collocation points. Mention the difference between:

- soft enforcement, which adds an initial-condition penalty with a tunable weight;
- hard enforcement, which constructs the neural output to satisfy the initial condition for every parameter value.

The reported experiment uses the hard formulation.

### C. Difference from operator learning

Make the distinction explicit:

- the reference study's operator model learns a mapping involving multiple initial conditions;
- the current implementation maps time to one trajectory for one fixed initial condition.

This distinction prevents an incorrect claim that the current model reproduces an operator-learning experiment.

## III. Physics-Informed Neural Network Method

This is the central technical section. Explain the method in the order in which one prediction is produced and trained.

### A. Network architecture

Define:

\[
\mathcal{N}(t;\theta):\mathbb{R}\rightarrow\mathbb{R}^3.
\]

State the layer count, width, activation, input, output, initialization, and parameter count after verifying the exact count.

Use one architecture figure if available. Do not include multiple Python source listings in the main paper.

### B. Hard initial-condition trial solution

Present the trial-solution equation and immediately show why it satisfies the initial condition:

\[
u_T(t_0)=u_0.
\]

State that the initial-condition penalty is absent from the reported loss. The implementation may contain a soft alternative, but the soft alternative is not part of the reported experiment and should not receive equal emphasis.

### C. Automatic-differentiation residual

Present:

\[
r(t)=\frac{du_T}{dt}-f(u_T(t)).
\]

Explain that automatic differentiation computes the time derivative and preserves the computational graph needed to backpropagate the residual loss to the network parameters.

Do not explain every source-code line. Explain the mathematical operation and its purpose.

### D. Collocation loss and optimization

Present the loss equation and summarize the training configuration in one table.

State:

- 3,000 fixed Latin-hypercube collocation points;
- full-batch Adam;
- 20,000 iterations;
- linearly decayed learning rate from \(10^{-3}\) to \(10^{-4}\);
- float32 precision;
- seed 0;
- no L-BFGS polishing in the reported run.

## IV. Experimental Setup

This section explains how the result was generated and measured.

### A. Numerical reference

Describe DOP853 and its tolerances. Describe RK4 only as an independent reference-consistency check. Use “numerical reference trajectory,” not “exact solution.”

### B. Training configuration

Include one compact table of all values needed to reproduce the reported run. Avoid repeating the same values in multiple paragraphs.

At minimum, include:

- equation parameters;
- initial state;
- time interval;
- network dimensions;
- activation;
- collocation count and sampling method;
- optimizer;
- iterations;
- learning-rate range and schedule;
- precision;
- random seed;
- initial-condition treatment.

### C. Evaluation protocol and metrics

Explain the separation between training and evaluation. Define MAE, RMSE, maximum absolute error, combined error, and pointwise residual magnitude.

The paper must state that the model was evaluated on a separate 1,001-point grid that was not used as the collocation set.

## V. Results

Present results first. Interpret them only after stating the measured values.

### A. Trajectory accuracy

Use the component-wise error table. The table must be readable in the IEEE two-column layout.

The result paragraph should say what the numbers show, for example:

> The component-wise RMSE values range from \(5.77\times10^{-6}\) to \(1.36\times10^{-5}\), while the combined RMSE is \(1.73\times10^{-5}\). The largest component-wise error occurs in the \(y\) state.

Do not claim improvement over another method unless the same task, configuration, metrics, and evaluation protocol were used.

### B. Training dynamics

Use the training-dynamics figure to show the reduction from \(2.17\times10^{-1}\) to \(6.82\times10^{-9}\).

Use the wording “the loss decreased” or “the training run exhibited empirical loss reduction.” Do not use “the method converges” as a mathematical claim.

### C. Solution and residual verification

Use a solution-versus-reference figure and a pointwise-residual figure if space permits.

The solution figure should show all three state variables and their errors. The residual figure should show the measured residual magnitude across the interval.

Explain that these plots provide two complementary checks:

- agreement with the numerical reference;
- satisfaction of the governing equations across the interval.

Use “measured residual” and “empirical verification.” Do not say that the plot proves exact satisfaction.

## VI. Discussion and Limitations

This section is required for a responsible conference paper, even if it is short.

### A. Interpretation

Explain the main meaning of the result:

- hard enforcement removes the need to tune an initial-condition penalty for the reported configuration;
- physics-only training recovered the selected trajectory;
- separate evaluation prevents the result from being only a report of training-point fit;
- automatic differentiation supplies the derivative used in the residual.

### B. Comparison with the reference study

Compare designs, not raw performance claims. The two studies differ in:

- model class;
- number of initial conditions;
- learned object;
- initial-condition treatment;
- network width;
- training budget;
- accuracy reporting.

Use this conclusion:

> The two methods address different learning problems, so their accuracy values and training budgets are not directly comparable.

Do not write “our method outperforms the reference study.”

### C. Limitations

The paper must state that:

1. only one initial condition was tested;
2. only \(k=2,l=1\) was evaluated;
3. only the interval \([0,1]\) was evaluated;
4. only one random seed was used;
5. no hard-versus-soft ablation was reported;
6. no collocation-density sensitivity study was reported;
7. no architecture sensitivity study was reported;
8. no runtime or computational-cost comparison was reported;
9. the DOP853 trajectory is a numerical reference, not an analytical solution;
10. the experiment does not establish long-horizon or unseen-initial-condition generalization;
11. the reported training used single precision.

Limitations should be written as scope boundaries, not as apologies. Each limitation should explain what future experiment would address it.

### D. Future work

Future work may include:

- multiple initial conditions;
- parameter sweeps over \(k\) and \(l\);
- hard-versus-soft initial-condition ablation;
- collocation-density sensitivity;
- multiple random seeds;
- longer time intervals;
- runtime and memory measurements;
- direct comparison with numerical solvers and operator-learning models.

Do not describe these as completed work.

## VII. Conclusion

Write one short paragraph. Include:

1. the fixed problem;
2. the hard-constrained PINN method;
3. the physics-only training;
4. the separate DOP853 evaluation;
5. the combined RMSE;
6. the correct scope of the conclusion.

The conclusion must end with the demonstrated scope: accurate single-trajectory approximation for the selected parameter setting and interval.

## Acknowledgment

Include an acknowledgment only if the target conference requires it or the team has a real acknowledgment to make.

Follow the supplied IEEE template. Do not leave template placeholder text in the submitted paper.

## References

Use IEEE numbered citations in square brackets. Cite only sources actually used in the manuscript.

At minimum, verify the bibliographic information for:

- the source of the Lorenz-1960 equations and reference experiment;
- the foundational PINN formulation;
- hard initial-condition trial-solution work;
- any numerical-solver documentation or method reference that is cited.

Never invent a citation. Never claim that a result is the first published result without a literature search that verifies the claim.

---

## 6. Content to Merge, Shorten, or Exclude

The implementation chapter is much broader than the conference paper. Use the following conversion rules.

| Implementation chapter content | IEEE paper treatment |
|---|---|
| Governing equations | Keep in Problem Formulation, shortened |
| General coefficient derivation | Keep only the necessary derivation |
| Adopted-settings table | Convert into the experimental configuration table |
| Full notation table | Keep only essential notation |
| Software architecture section | Reduce to one reproducibility paragraph |
| Module map | Exclude from the main paper unless required by the venue |
| Traceability table | Exclude from the main paper; place in supplementary material if needed |
| Configuration code listing | Replace with a hyperparameter table |
| Reproduction shell listing | Replace with a code-availability statement and concise command if useful |
| Network architecture | Keep as a Method subsection |
| Initial-condition section | Keep and emphasize |
| Physics-residual section | Keep and emphasize |
| Collocation section | Merge with optimization subsection |
| Training loop listing | Replace with prose or compact pseudocode |
| Evaluation section | Keep in Experimental Setup |
| Results section | Keep, shorten, and move interpretation to Discussion |
| Comparison section | Merge into Related Work or Discussion |
| Nine-stage roadmap table | Exclude from the main paper |
| Repeated transition paragraphs | Remove |

The target paper should explain the method once, not explain the same method through equations, source listings, module tables, and repeated prose.

---

## 7. Figures and Tables

### 7.1 Recommended main-paper figures

Use no more figures than the page limit allows. The recommended set is:

1. **PINN pipeline figure:** time input, neural network, hard trial solution, automatic differentiation, residual, and loss.
2. **Solution-versus-reference figure:** predicted and reference \(x,y,z\) trajectories with signed errors.
3. **Training and residual figure:** training loss and pointwise residual, if they remain readable when combined.

If the figure becomes too dense in two-column format, use two figures rather than shrinking labels until they are unreadable.

### 7.2 Recommended main-paper tables

Use:

1. one compact training-configuration table;
2. one component-wise error table.

Move detailed traceability and module tables to supplementary material if the venue permits it.

### 7.3 IEEE figure and table rules

Follow the supplied template:

- place figures and tables near the top or bottom of columns when possible;
- cite every figure and table before it appears;
- put figure captions below figures;
- put table titles above tables;
- use `Fig.` in text for figure references;
- use readable axis labels and units where applicable;
- do not alter IEEE margins, fonts, column widths, or line spacing;
- remove every template instruction and placeholder before submission.

### 7.4 Figure accuracy rules

Every figure must have:

- correct axis labels;
- a caption that explains what is shown;
- consistent state-variable names;
- a legend that is readable at final publication size;
- no unsupported visual claim;
- values consistent with the result table.

Do not create a figure that visually suggests a comparison the experiment did not perform.

---

## 8. Terminology and Style Rules

### 8.1 Preferred terms

Use:

- “numerical reference trajectory”;
- “reported experiment”;
- “fixed initial condition”;
- “single-trajectory approximation”;
- “hard initial-condition enforcement”;
- “physics-only training”;
- “collocation points”;
- “automatic differentiation”;
- “empirical loss reduction”;
- “measured residual.”

### 8.2 Terms to avoid or use carefully

Avoid unsupported or inflated terms such as:

- revolutionary;
- novel framework;
- breakthrough;
- exact solution;
- guaranteed convergence;
- universal;
- robust in all regimes;
- state-of-the-art;
- superior;
- optimal;
- real-time;
- first-ever;
- proves.

Use “demonstrates” only for what the experiment actually tests. Use “suggests” or “indicates” for interpretations that are not mathematical guarantees.

### 8.3 Simple storytelling pattern

For each technical subsection, follow this order:

1. State the problem or operation.
2. Present the equation or design.
3. Explain why the design is used.
4. State the exact implementation setting.
5. Connect to the next stage.

Example:

> The network must satisfy the initial condition before the ODE residual is minimized. We therefore replace the raw network output with a trial solution that contains a ramp factor vanishing at the initial time. This guarantees \(u_T(0)=u_0\) for every parameter value, so the reported training objective contains only the physics residual.

This is preferable to several paragraphs explaining the same idea through source-code details.

### 8.4 Paragraph rules

Each paragraph should have one main purpose. Avoid:

- long historical introductions;
- multiple definitions of the same symbol;
- repeating table values in every section;
- explaining basic Python syntax;
- explaining the full repository when the paper is about the method;
- broad claims unsupported by the one experiment.

---

## 9. Team Working Protocol

The team should work in stages. Each stage must produce a concrete output and a review before the next stage begins.

### Stage 1: Evidence verification

**Owner:** implementation/evidence lead  
**Output:** verified fact sheet

Check every numerical value in Section 3 of this document against the implementation and generated results.

Resolve the known inconsistency in the parameter count: the implementation text states both 11,283 and 11,263. Only one verified value may appear in the final paper.

Also verify:

- exact endpoint values;
- exact combined-error definition;
- exact residual-plot quantity;
- software and package versions if they will be reported;
- hardware details if runtime or hardware claims are added.

### Stage 2: Paper outline

**Owner:** structure lead  
**Output:** section-by-section outline using the order in Section 4

The outline must assign:

- one purpose to each section;
- the equations required;
- the tables and figures required;
- the evidence supporting each main claim;
- content that will be excluded or moved to supplementary material.

### Stage 3: Literature and citation verification

**Owner:** related-work and citation lead  
**Output:** verified IEEE reference list and citation map

Verify every source through a reliable bibliographic source. Map every citation to a specific statement in the paper.

Do not add citations only to make the paper look more scholarly. Each citation must support a claim or establish prior method context.

### Stage 4: Method writing

**Owner:** method lead  
**Output:** Sections II and III

Write from the verified facts. Keep equations central and source-code listings out of the main paper unless a listing is essential for understanding the contribution.

### Stage 5: Experimental and results writing

**Owner:** experiment/results lead  
**Output:** Sections IV and V

Make the training/evaluation separation explicit. Insert only verified tables and figures. Every result paragraph must point to a measured table or figure.

### Stage 6: Discussion and scope review

**Owner:** critical reviewer  
**Output:** Sections VI and VII plus an overclaim audit

Check that the paper does not imply:

- multi-initial-condition generalization;
- long-horizon validity;
- superiority over the reference study;
- exact mathematical convergence;
- runtime improvement;
- novelty that has not been verified.

### Stage 7: IEEE formatting and final review

**Owner:** formatting lead  
**Output:** completed IEEE manuscript

Check:

- two-column IEEE formatting;
- section hierarchy;
- equation numbering;
- figure and table placement;
- citation numbering;
- readable text in all figures and tables;
- removal of all template instructions;
- page limit;
- consistent terminology;
- no unresolved placeholders.

---

## 10. Claim-Evidence Map

Use this map before approving any statement.

| Claim | Evidence required | Currently supported? |
|---|---|---|
| The model approximates one Lorenz-1960 trajectory | Defined task and predicted trajectory | Yes |
| The initial condition is satisfied by construction | Trial-solution equation | Yes |
| The training uses physics-only loss | Loss definition and training code description | Yes |
| Automatic differentiation computes the ODE derivative | Residual implementation description | Yes |
| 3,000 Latin-hypercube points are used for training | Configuration and collocation description | Yes |
| Evaluation is separate from training points | Evaluation protocol | Yes |
| DOP853 provides the numerical reference | Reference-solver description | Yes |
| The combined RMSE is \(1.73\times10^{-5}\) | Error table | Yes, subject to final result verification |
| The method generalizes to other initial conditions | Multiple-condition experiments | No |
| The method is better than the reference study | Matched-task comparison | No |
| The method is state of the art | Broad, matched benchmark study | No |
| The method converges mathematically | Formal analysis | No |
| The method is faster than DOP853 | Runtime experiment | No |
| The result is the first published result | Verified literature search | Not currently established |

If a proposed sentence has no evidence in this map, either add the required experiment/analysis or remove the sentence.

---

## 11. Known Issues to Resolve Before Final Submission

These are writing and verification issues, not permission to change the implementation silently.

### 11.1 Parameter-count inconsistency

The implementation text reports both:

- 11,283 trainable parameters;
- 11,263 parameters.

Calculate or inspect the actual network parameter count and use one verified value everywhere.

### 11.2 Combined metric definition

The implementation describes the combined row as a Euclidean magnitude but also reports MAE, RMSE, and maximum absolute error. The final paper must define exactly how the combined row is calculated.

Do not use “combined \(L_2\)” without a mathematical definition.

### 11.3 Residual metric definition

The paper must state whether the residual figure shows:

- the Euclidean norm of the three-component residual;
- component-wise residuals;
- a maximum component residual;
- or another quantity.

Use the same definition in the method and results sections.

### 11.4 Reference terminology

Replace “ground truth” with “high-accuracy numerical reference” unless an analytical solution is available.

### 11.5 Comparison claim

The comparison with Matthews and Bihlo is not a matched benchmark comparison. Present it as a difference in learning problem and method design.

### 11.6 Priority claim

Do not claim that the error table is the first published table unless the team performs and documents a proper literature search.

---

## 12. Final Pre-Submission Checklist

### Scope and correctness

- [ ] The paper clearly says that the experiment solves one fixed trajectory.
- [ ] The equations, coefficients, initial condition, and interval are correct.
- [ ] The hard trial solution is presented correctly.
- [ ] The loss contains only the physics residual for the reported run.
- [ ] The reference trajectory is described as numerical, not analytical.
- [ ] The training and evaluation grids are clearly distinguished.
- [ ] The parameter count has been verified and is consistent.
- [ ] The combined metric has a precise definition.
- [ ] The residual metric has a precise definition.

### Evidence and claims

- [ ] Every numerical result matches the verified result files.
- [ ] Every result claim points to a table or figure.
- [ ] No unsupported state-of-the-art claim appears.
- [ ] No unsupported generalization claim appears.
- [ ] No unsupported runtime claim appears.
- [ ] No unsupported novelty or first-publication claim appears.
- [ ] The comparison with the reference study does not claim superiority.
- [ ] Limitations are stated clearly.

### Writing quality

- [ ] The introduction reaches the problem quickly.
- [ ] Each paragraph has one main purpose.
- [ ] Technical terms are defined at first use.
- [ ] Repeated explanations have been removed.
- [ ] Source-code listings have been minimized.
- [ ] The prose is simple, direct, and technical.
- [ ] The paper does not read like a thesis chapter or software manual.

### IEEE formatting

- [ ] The official IEEE template is used without changing its layout settings.
- [ ] Section headings are not manually numbered.
- [ ] Equations are numbered consecutively when referenced.
- [ ] Figures are cited before they appear.
- [ ] Figure captions are below figures.
- [ ] Table titles are above tables.
- [ ] Figure labels remain readable at publication size.
- [ ] IEEE citation style is used consistently.
- [ ] All template instructions and placeholders have been removed.
- [ ] The paper satisfies the target conference page limit.

### Reproducibility

- [ ] The training configuration is complete.
- [ ] The reference-solver settings are reported.
- [ ] The evaluation grid size is reported.
- [ ] The code repository or reproducibility location is included if permitted.
- [ ] Software versions are reported if they affect reproducibility.
- [ ] Hardware details are reported if runtime or resource claims are made.

---

## 13. One-Sentence Paper Story

Every team member should be able to summarize the paper using this sentence:

> We train a time-to-state physics-informed neural network on the Lorenz-1960 equations, enforce the fixed initial condition analytically through a trial solution, and obtain a combined RMSE of approximately \(1.73\times10^{-5}\) against a separate DOP853 reference grid for the selected trajectory.

If a proposed section, figure, claim, or experiment does not support this story, the team must decide whether it is necessary before adding it to the paper.
