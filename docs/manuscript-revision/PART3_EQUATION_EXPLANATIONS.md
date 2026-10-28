# Part 3: explanations for the updated equations

Reviewed 8 October 2026 against `E:/University/FYDP/lorenz1960-pinn/writings/conference-paper/main.tex` after the uploaded-source layout fixes. This guide does not edit the manuscript. Keep numerical claims intact; the additional table and its accompanying prose are deferred.

Line numbers refer to the current source and will shift. Use the quoted text and equation labels as searchable anchors. Paste the fenced blocks as visible LaTeX prose, not percent-sign comments. Apply replacements rather than appending them to the old paragraphs.

## Equation analysis

Coefficient substitution gives -0.1, 1.6, and -0.75. The nonlinear vector field correctly collects the right-hand sides in x,y,z order. The parameter counts are 11,283 per network and 304,641 across 27 networks. The trial solution satisfies its starting state, and its product-rule derivative is correct for the unit time factor used in the implementation. Residual signs, the component-average point loss, detached causal weights, weighted objective, unweighted objective, and combined RMSE identity match the inspected implementation.

The first causal weight is one because the earlier-point sum is empty. There are 41,446 unique global points and 41,472 total point uses across window training. The causal objective divides by the point count, not the sum of weights. Combined state RMSE does not divide by three; residual MSE does. These checks concern formulas and implementation, not fresh verification of saved run outcomes.

The remaining gap is mainly visible definitions and introductions. Batch-symbol definitions and the autograd/vector-field distinction currently exist as comments, so readers cannot see them. The dense residual-MSE display from Part 1 is still absent. Physical dimensions or units are not established by the array-shape declarations.

## II-A: Lorenz Equations

### Physical parameters

Insert before `eq:lorenz1960`, after the sentence ending “general form ... is:” (around line 75).

```latex
Here $x(t)$, $y(t)$, and $z(t)$ are the three state components,
and $\kappa$ and $\ell$ are the physical wavenumber parameters.
We use $k$ separately to index the time windows introduced below.
```

### Coefficient substitution

Replace the sentence beginning “Suppose the state vector is given by” at line 95, before the coefficient breakdown.

```latex
We write the state as $u(t)=[x(t),y(t),z(t)]^\mathsf{T}$.
Substituting $\kappa=2$ and $\ell=1$ into
\eqref{eq:lorenz1960} gives the following coefficients.
```

Insert after the coefficient breakdown and before `eq:ode`.

```latex
These coefficients multiply $yz$, $xz$, and $xy$, respectively,
giving the reduced system used in this experiment.
```

### Vector field

Insert after `eq:ode`, before the display beginning `u=[x,y,z]` (around line 116).

```latex
Collecting the three right-hand sides gives the vector form
$du/dt=f(u)$, where $f$ is the prescribed ODE vector field.
```

Insert after that vector-field display and before “For our experiment we used the coefficients” at line 131.

```latex
The input to $f$ is a three-component state vector, and its output
contains the prescribed rates of change of $x$, $y$, and $z$ in
that order. During training, we evaluate this same function at
the predicted state $\hat{u}_k(t)$.
```

Keep the existing coefficient values, initial condition, final time, and nearly-closed-orbit statements.

## III-A: Network Structure

### Window index and boundaries

Insert after “We split the total time ... and gave each window its own network”, before the partition display beginning `t_k=kT/27` (around line 150).

```latex
The index $k=0,\ldots,26$ identifies the 27 windows, while
$t_0,\ldots,t_{27}$ are their 28 boundary times. Each window
$W_k$ has width $\Delta t=T/27$ and its own network parameters
$\theta_k$.
```

### Correction output

Replace the architecture sentence beginning “Each window has its own network” at line 164. Keep the separate initialization paragraph.

```latex
Each window uses a network $N_k(t;\theta_k)$ with one scalar time
input, four hidden layers of 60 tanh units, and three output
components. These outputs form a correction vector; the
predicted state $\hat{u}_k(t)$ is obtained by applying the trial
solution in \eqref{eq:trial}.
```

### Parameter calculation

Insert before the display beginning `P_{\mathrm{window}}` (around line 168).

```latex
The input layer, three hidden-to-hidden connections, and output
layer contribute 120, $3\times3660$, and 183 parameters,
respectively, including biases. The total below counts all 27
stored networks.
```

## III-B: Window Handoff

### Trial-solution terms

Replace the opening sentence “As Lorenz system is an initial value problem”, before `eq:trial` (around line 177).

```latex
For window $k=0,\ldots,26$, let $t_k$ be its starting time and
$u_{0,k}\in\mathbb{R}^3$ its fixed starting state. We enforce
this state through the following trial solution, where
$N_k(t;\theta_k)\in\mathbb{R}^3$ is the network correction.
```

### Initial-state identity

Replace the entire paragraph beginning `where $k=0,\ldots,26$` at lines 197–200, after the initial-state identity and before `eq:handoff`.

```latex
At $t=t_k$, the scalar factor $t-t_k$ is zero, so the predicted
state equals $u_{0,k}$ independently of the network parameters
\cite{lagaris1998}. The first window uses the prescribed initial
state $u_{0,0}=u(0)$; each later window uses the predicted endpoint
of its predecessor, as specified below.
```

### State continuity versus warm start

Replace the entire paragraph beginning `$\theta_{k+1}\leftarrow\theta_k` after the continuity identity and before the pipeline figure (around line 211).

```latex
The endpoint transfer in \eqref{eq:handoff}, together with the
hard trial solution, makes the predicted states continuous at
shared boundaries. Separately, we initialize the next network by
copying $\theta_{k+1}\leftarrow\theta_k$ and then train its
parameters independently. Since transferred states are
predictions, errors from earlier windows can propagate forward.
```

Retain the derivative-continuity qualification when relocating Discussion and Limitations later. Parameter copying does not itself prove state continuity.

## III-C: Training Loss

### Collocation points

Replace the first sentence after `\subsection{Training Loss}` at line 218, before the collocation/count display.

```latex
Each window is trained on $n_k=1536$ uniformly spaced times
$t_{k,i}$, ordered by $i=1,\ldots,n_k$ and including both endpoints.
```

Replace the complete paragraph beginning “Ideally 1535 points” at line 233, after that display.

```latex
Adjacent windows share boundary times, giving 41,446 unique
global points but 41,472 point uses across window training.
The implementation assigns each shared boundary to the next
window on the global grid and also includes it when training
the outgoing window. The last window includes the final endpoint.
```

### Time derivative and residual function

Replace the complete paragraph beginning “Then we apply the trial solution” at line 234, before the derivative display.

```latex
We apply the trial solution to the network outputs and compute
its time derivative component by component using PyTorch
automatic differentiation. Because $u_{0,k}$ is fixed during
training, differentiating \eqref{eq:trial} gives the product-rule
expression below.
```

Insert after the derivative display and before `eq:residual` (around line 245). The two comments there can remain, but this explanation must be visible prose.

```latex
The vector-valued ODE residual function compares this learned
trajectory derivative with $f(\hat{u}_k(t))$, the original ODE
right-hand side evaluated at the predicted state. Both terms
are three-component rate vectors; their difference is $r_k(t)$.
```

### Batch symbols and dimensions

Insert after the residual-component equations and before the display beginning `\mathbf{t}_k` (around line 269).

```latex
For a batch of $n_k$ times, each row represents one collocation
point and the three columns follow the $x,y,z$ order.
We denote the network outputs, trial states, time derivatives,
ODE right-hand sides, and residuals by $\mathbf{N}_k$,
$\hat{\mathbf{U}}_k$, $\dot{\hat{\mathbf{U}}}_k$,
$\mathbf{F}_k$, and $\mathbf{R}_k$, respectively.
```

Optional, after the batch-shape displays and before the point-loss introduction:

```latex
Row $i$ of $\mathbf{F}_k$ is
$f(\hat{u}_k(t_{k,i}))^\mathsf{T}$, and row $i$ of
$\mathbf{R}_k$ is $r_k(t_{k,i})^\mathsf{T}$.
```

### Point loss

Replace “For each ordered collocation time ...” at line 283, before `eq:pointloss`.

```latex
At each time $t_{k,i}$, we square the three residual components
and average them to obtain the nonnegative scalar point loss
$L_{k,i}$. This reduces each residual row from three components
to one loss value.
```

### Prefix and causal weight

Keep the existing causal-training motivation. Insert immediately before the display beginning `S_{k,i}` (around line 301).

```latex
The prefix $S_{k,i}$ sums point losses at earlier times in the
same window. Its sum is empty at the first point, so
$S_{k,1}=0$ and the first causal weight is one.
```

### Scalar window objective

Insert after the weight-bound/epsilon display, immediately before `eq:causalloss`.

```latex
We multiply each point loss by its causal weight and average over
all $n_k$ points. The resulting $\mathcal{L}_{k,\epsilon}$ is one
scalar objective for the current window and epsilon stage at the
current parameter values.
```

Replace the complete paragraph beginning `Here $n_k=1536$` at line 340, after the loss/weight-vector displays.

```latex
Here $k$ identifies the window, $i$ its current point, and $j<i$
an earlier point. The vectors $\mathbf{L}_k$ and $\mathbf{w}_k$
contain one scalar loss and one scalar weight per point.
The parameter $\epsilon$ controls how strongly accumulated earlier
losses suppress later weights. We use the ordered stages
$0.01$, $0.1$, $1$, and $10$. As earlier losses decrease, later
weights rise toward one. We recompute the weights during training
and detach them when calculating the parameter gradient.
```

The denominator is the point count, not the sum of weights. Avoid implying a normalized weighted-average convention.

### Repeated mechanics

In the paragraph beginning “Only using ODE residual” at line 342, retain the existing motivation through “possibly the later error can be reduced as well.” Replace only the remainder beginning “To address this, we first calculate” with:

```latex
Equations~\eqref{eq:pointloss}--\eqref{eq:causalloss} implement
this ordering through point losses and causal weights.
The following subsection gives the stage stopping rule and
unweighted L-BFGS refinement.
```

This preserves the existing scientific motivation while avoiding repeated equation explanations.

## III-D: Training Procedure

Replace the sentence beginning “If the minimum causal weight from the entire window among 1535 points” in the opening paragraph (lines 346 onward).

```latex
If the verified post-update minimum over all 1536 training-point
weights exceeds $0.99$, training advances to the next
$\epsilon$ stage.
```

Keep the cap statement and numerical budgets. Distinguish a capped stage from a failed trajectory in the later failure-case discussion.

Insert immediately before the display beginning `\mathcal{L}^{\mathrm{raw}}_k`.

```latex
For L-BFGS refinement, we remove the causal weights and minimize
the unweighted mean of the point losses. Equivalently, this is
the mean of all $3n_k$ squared residual components.
```

Use “iterations” for the configured L-BFGS budget instead of “epoch” in the preceding sentence. Preserve the distinction between closure evaluations and optimizer iterations in Run Comparison.

## IV-B: Evaluation

Insert immediately after `\subsection{Evaluation}` at line 431, before the error-vector display.

```latex
At evaluation time $t_j$, we subtract the numerical reference
state from the predicted state to obtain the error vector
$e(t_j)\in\mathbb{R}^3$, with $j=1,\ldots,n_{\mathrm{eval}}$.
This state-reference error is distinct from the ODE residual.
```

Insert between that display and `eq:rmse` (around line 440).

```latex
The combined RMSE averages the squared Euclidean error norms over
evaluation times and then takes the square root. Its equivalent
component form is the square root of the sum of the three squared
component RMSEs.
```

Insert after `eq:rmse`, before the display beginning `\operatorname{MAE}_c` (around line 455).

```latex
For $c\in\{x,y,z\}$, we report componentwise MAE, RMSE, and
maximum absolute error. We also report the mean and maximum
Euclidean error norms, denoted by $E_{\mathrm{mean}}$ and
$E_{\mathrm{max}}$.
```

Remove the isolated sentence beginning “while the combined maximum error is” at line 467 because the display and new introduction already define it. The quantity remains present.

Replace only the sentence beginning “The dense-grid residual mean square” at line 468. Keep the following comparison sentences and claims.

```latex
The dense-grid residual MSE averages the squared ODE residuals
over all evaluation times and all three components. It therefore
includes a factor of $1/(3n_{\mathrm{eval}})$, whereas the squared
combined state RMSE averages vector norms over times without
dividing by three.
```

If the missing dense residual-MSE display is added in a later equation pass, put it immediately after this explanation.

## V-A: Prediction Accuracy

Insert near the opening reference to Table II, before the table.

```latex
In the state-norm row, the MAE-column entry is
$E_{\mathrm{mean}}$, the RMSE entry is $\operatorname{RMSE}_2$,
and the maximum entry is $E_{\mathrm{max}}$.
```

This explains the existing cells without changing them. The reference-drift sentence also needs to identify its actual diagnostic and normalization from the corresponding artifact. Do not invent those details or remove the claim.

## Integration and space

Use replacements wherever specified. Keep definitions next to their equations; avoid splitting the error definition from Equation 9 or the derivative from Equation 5 when arranging floats. Keep the shorter headings. Remove announcement-only paragraphs in the later writing pass as previously agreed. Preserve distinct scientific statements when relocating Discussion and Limitations content.

The additional table, its final caption, and failure-case prose are deferred. The present six-page PDF remains the baseline; this guide does not claim a five-page build. Compile and inspect the final layout after applying the writing replacements.
