# Deep Comparative Analysis: Causal Windowed PINN vs. Batch-Wise PINN on Lorenz-1960

**Date**: September 22, 2026  
**Scope**: Full architectural, mathematical, gradient, and telemetry evaluation of `feat/batch-precision-run` (Run A) and `feat/causal-window-training` (Run B) on the closed periodic orbit ($T = 13.26446$).  
**Repository**: `ihmorol/lorenz1960-pinn`

---

## 1. Executive Summary & Benchmark Comparison

The objective of this research is solving the 3-dimensional conservative Lorenz-1960 nonlinear oscillator over one complete closed periodic orbit:
$$\dot{x} = -c_1 y z, \quad \dot{y} = c_2 x z, \quad \dot{z} = -c_3 x y$$
with coefficients $k=2, l=1$ yielding $(c_1, c_2, c_3) = (0.10, 1.60, 0.75)$, and initial condition $u_0 = (0.50, 0.75, 1.00)$ over the time horizon $t \in [0, T_{loop}]$ where $T_{loop} = 13.26446$ ($|u(T) - u_0| \approx 6 \times 10^{-9}$).

Two distinct paradigms are implemented in the codebase across two branches:
1. **Run A (`feat/batch-precision-run`)**: Single monolithic neural network covering the entire domain $t \in [0, 13.26446]$ with Latin Hypercube Sampling (LHS) and standard Mean Squared Physics Residual loss.
2. **Run B (`feat/causal-window-training`)**: 27 sequential time windows ($\Delta t \approx 0.491$) with temporal causality weighting (Wang, Sankaran & Perdikaris, 2024), warm-start weight inheritance, hard endpoint state hand-off, and per-window L-BFGS fine-tuning.

### Benchmark Results Table

| Performance Dimension | Run A (Batch Monolithic) | Run B (Causal Windows) | Gain / Difference |
| :--- | :--- | :--- | :--- |
| **Branch** | `feat/batch-precision-run` | `feat/causal-window-training` | Domain Decomposition + Causality |
| **Model Architecture** | 1 network (4x60, tanh) | 27 networks (each 4x60, tanh) | 11,283 params vs. 304,641 params |
| **Precision** | `torch.float64` | `torch.float64` | Both high precision |
| **Collocation Scheme** | 39,793 LHS points | 39,793 Uniform points (1,474/win) | Causal ordering requires uniform |
| **Optimization Budget** | 40,000 Adam + 5,000 L-BFGS | $\le 4,000$ Adam/stage + 1,500 L-BFGS/win | 45k total vs. 78,677 window-steps |
| **Wall-Clock Time** | 4,528.6 s (~75.5 min) | 928.6 s (~15.5 min) | **4.88x faster** |
| **Final Loss** | $2.80 \times 10^{-4}$ (plateau) | $1.49 \times 10^{-8}$ (orbit mean) | **$18,800\times$ lower loss** |
| **Combined $L_2$ RMSE** | **2.237** | **$8.43 \times 10^{-5}$** | **$26,500\times$ higher accuracy** |
| **Combined $L_2$ Max Error** | **3.346** | **$2.35 \times 10^{-4}$** | **$14,200\times$ lower max error** |
| **$R^2$ Scores ($x, y, z$)** | -30.89, -1.88, 0.026 | 0.9999998, 0.999999998, 0.999999995 | True trajectory recovery |
| **Invariant 1 Max Drift** | $1.00 \times 10^{-1}$ (10.0% drift) | $2.51 \times 10^{-4}$ (0.025% drift) | Preserves first quadratic invariant |
| **Invariant 2 Max Drift** | $1.14 \times 10^{-1}$ (11.4% drift) | $1.92 \times 10^{-4}$ (0.019% drift) | Preserves second quadratic invariant |
| **Outcome** | **Catastrophic Failure** (False Basin) | **Complete Success** (Closed Orbit) | SOTA accuracy on chaotic loop |

---

## 2. In-Depth Bug & Implementation Defect Audit

Through line-by-line inspection of `src/pinn/pinn.py`, `src/pinn/train.py`, `src/pinn/functions/losses.py`, `src/pinn/functions/trial.py`, `src/pinn/functions/derivative.py`, and `src/pinn/functions/optimizers.py`, several critical bugs, mathematical defects, and scaling flaws were uncovered.

### Defect 1: Collocation Density Scaling Defect in Causal Weights (CRITICAL)
- **Location**: [`src/pinn/functions/losses.py:9-19`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/functions/losses.py#L9-L19)
- **The Code**:
  ```python
  def causal_weights(L_sorted: Tensor, eps: float) -> Tensor:
      earlier = torch.cumsum(L_sorted, 0) - L_sorted
      return torch.exp(-eps * earlier).detach()

  def causal_loss(r: Tensor, t: Tensor, eps: float) -> tuple[Tensor, Tensor]:
      order = torch.argsort(t.reshape(-1))
      L = r[order].pow(2).mean(dim=1)
      w = causal_weights(L, eps)
      return (w * L).mean(), w
  ```
- **The Mathematical Bug**:
  In Wang, Sankaran & Perdikaris (2024), the temporal domain is partitioned into $M$ discrete intervals (time slabs), where $\mathcal{L}_k$ is the interval-average loss, approximating the continuous integral $\int_0^{t_i} \mathcal{L}(\tau) d\tau \approx \sum_{k<i} \mathcal{L}_k \Delta t$.
  In this implementation, **every single collocation point** is treated as its own interval. Thus, `earlier[i]` is the raw unnormalized sum over all prior collocation points:
  $$\sum_{k=0}^{i-1} L_k \approx i \cdot \bar{L}$$
  As the number of collocation points $N_c$ per window increases (here $N_c \approx 1,474$), the accumulated sum grows proportionally with $N_c$.
- **The Resulting Pathology**:
  The criterion to advance an $\epsilon$-stage is $\min_i w_i > \delta = 0.99$.
  Taking the natural logarithm:
  $$w_{\min} = \exp\left( -\epsilon \sum_{k=0}^{N_c-1} L_k \right) > 0.99 \implies \sum_{k=0}^{N_c-1} L_k < \frac{-\ln(0.99)}{\epsilon} \approx \frac{0.01005}{\epsilon}$$
  At stage $\epsilon = 10.0$, this requires:
  $$\sum_{k=0}^{N_c-1} L_k < 0.001005 \implies \bar{L} < \frac{0.001005}{1474} \approx 6.82 \times 10^{-7}$$
  The mean per-point residual loss $\bar{L}$ must fall below $6.82 \times 10^{-7}$ for Adam to advance!
  At a learning rate of $10^{-3}$, Adam's stochastic descent noise floor sits around $10^{-5}$ to $10^{-6}$.
  **Consequence**: In windows 9–11 and 23–25 (the highest-velocity stretches of the orbit), Adam was tasked with reaching an unnormalized threshold below its optimization floor. It hovered at $\min w \in [0.75, 0.90]$, hit the 4,000-iteration cap every time, and caused the loss to *rise* from $10^{-6}$ to $10^{-5}$ due to accumulator chatter before L-BFGS repaired it.
- **The Solution**: Normalize the cumulative loss by the point count or multiply by $\Delta t$:
  $$\text{earlier} = \frac{1}{N_c} (\text{cumsum}(L) - L) \quad \text{or} \quad \Delta t \sum L_k$$

---

### Defect 2: Lack of First-Derivative ($C^1$) Continuity Across Window Joints
- **Location**: [`src/pinn/functions/trial.py:4-7`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/functions/trial.py#L4-L7) and [`src/pinn/pinn.py:59-62`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/pinn.py#L59-L62)
- **The Code**:
  ```python
  def hard_initial_condition(t: Tensor, n: Tensor, u0: Tensor, t0: float, tf: float, form: str) -> Tensor:
      g = (t - t0) / (tf - t0) if form == "span" else (t - t0)
      return u0 + g * n
  ```
- **The Mathematical Bug**:
  The trial function guarantees exact value continuity at the initial boundary $t = t_0$:
  $$u(t_0) = u_0 + 0 \cdot \mathcal{N}(t_0) = u_0 \quad (\text{exact machine precision } \sim 10^{-16})$$
  However, computing the time derivative analytically:
  $$\frac{du}{dt} = \frac{dg}{dt} \mathcal{N}(t) + g(t) \frac{d\mathcal{N}}{dt} = 1 \cdot \mathcal{N}(t) + (t - t_0) \frac{d\mathcal{N}}{dt}$$
  Evaluating at the initial boundary $t = t_0$:
  $$\left. \frac{du}{dt} \right|_{t=t_0} = \mathcal{N}(t_0)$$
  Notice that $\mathcal{N}(t_0)$ is purely the arbitrary output of the neural network!
  The true derivative required by the ODE physics is:
  $$\left. \frac{du}{dt} \right|_{t=t_0} = f(u_0)$$
  Therefore, at $t = t_0$, the physics residual is:
  $$r(t_0) = \left. \frac{du}{dt} \right|_{t=t_0} - f(u(t_0)) = \mathcal{N}(t_0) - f(u_0)$$
- **The Resulting Pathology**:
  The neural network is forced to numerically *learn* to output $\mathcal{N}(t_0) \approx f(u_0)$. In every window, this creates an unforced residual error spike of order $10^{-3}$ at the left joint (evident in `figures/physics_residual.png` and `figures/joint_continuity.png`).
- **The Solution**: Construct a $C^1$-continuous trial function:
  $$u_T(t) = u_0 + (t - t_0) f(u_0) + (t - t_0)^2 \mathcal{N}(t)$$
  Differentiating yields:
  $$\dot{u}_T(t) = f(u_0) + 2(t - t_0) \mathcal{N}(t) + (t - t_0)^2 \dot{\mathcal{N}}(t) \implies \dot{u}_T(t_0) = f(u_0) \quad \text{identically!}$$
  This makes the physics residual at the joint exactly zero by construction ($r(t_0) \equiv 0$), eliminating the $10^{-3}$ joint residual spikes entirely!

---

### Defect 3: Unweighted Loss Discrepancy During L-BFGS Phase
- **Location**: [`src/pinn/train.py:211-216`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/train.py#L211-L216)
- **The Code**:
  ```python
  if cfg.lbfgs_iters > 0:
      def closure() -> Tensor:
          loss = pinn_loss(sub, pts.clone().requires_grad_(True))
          history.loss.append(loss.item())
          return loss
      run_lbfgs(sub.parameters(), closure, cfg.lbfgs_iters, cfg.torch_dtype)
  ```
- **The Issue**:
  During Adam training, `train_windows` enforces the causally weighted loss `causal_loss(parts.r, t, eps)`.
  However, in the L-BFGS polishing stage, the closure calls `pinn_loss(sub, pts)`, which evaluates the **unweighted** mean squared residual `mean_squared_residual(parts.r)`.
  If Adam terminates normally with $\min w > 0.99$, all weights are close to 1, so the unweighted residual is nearly consistent. But in the 6 capped windows where Adam terminated early with $\min w \approx 0.75-0.90$, causality was abruptly abandoned upon switching to L-BFGS.

---

### Defect 4: Dtype Hardcoding in `PINN.__init__`
- **Location**: [`src/pinn/pinn.py:52-53`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/pinn.py#L52-L53)
- **The Code**:
  ```python
  self.register_buffer("u0", torch.tensor([cfg.initial_state], dtype=torch.float32))
  self.register_buffer("coeffs", torch.as_tensor(cfg.coefficients, dtype=torch.float32).reshape(1, 3))
  ```
- **The Issue**:
  `u0` and `coeffs` are explicitly registered as `float32`. While `train()` invokes `model.to(dtype=cfg.torch_dtype)`, standalone instantiation `PINN(Config(dtype="float64"))` retains `float32` buffers. When evaluating `model(t)` with float64 `t`, PyTorch throws a dtype mismatch runtime error during buffer addition.
- **The Fix**:
  ```python
  self.register_buffer("u0", torch.tensor([cfg.initial_state], dtype=cfg.torch_dtype))
  self.register_buffer("coeffs", torch.as_tensor(cfg.coefficients, dtype=cfg.torch_dtype).reshape(1, 3))
  ```

---

### Defect 5: Inefficient Autograd Loop in `time_derivative`
- **Location**: [`src/pinn/functions/derivative.py:5-7`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/functions/derivative.py#L5-L7)
- **The Code**:
  ```python
  def time_derivative(u: Tensor, t: Tensor) -> Tensor:
      cols = [torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0][:, 0] for j in range(u.shape[1])]
      return torch.stack(cols, dim=1)
  ```
- **The Issue**:
  A list comprehension executes 3 distinct reverse-mode autodiff graph traversals (one for each state variable $x, y, z$). Because `create_graph=True` builds the second-order graph for the optimization backward pass, running 3 separate graph traversals triples the autograd memory and execution overhead.
- **The Fix**:
  In PyTorch, vectorizing across outputs via `torch.autograd.grad` with a combined vector or utilizing `torch.func.vjp` / `torch.func.jvp` yields forward-mode or single-pass derivative evaluation.

---

### Defect 6: Ineffective Learning Rate Scheduler in Windowed Mode
- **Location**: [`src/pinn/functions/optimizers.py:13`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/functions/optimizers.py#L13) and [`src/pinn/train.py:177`](file:///e:/University/FYDP/lorenz1960-pinn/src/pinn/train.py#L177)
- **The Issue**:
  `StepLR` is initialized with `lr_decay_every = 5000`. In `train_windows`, the maximum iterations per window across all $\epsilon$-stages is at most $4 \times 4000 = 16,000$, and in practice 21 out of 27 windows finish in under 300 Adam steps.
  Because the optimizer and scheduler are re-instantiated from scratch at the start of each window (`adam, sched = adam_with_decay(...)`), the step count resets to 0. Consequently, the scheduler never reached 5,000 steps, and **learning rate decay was never applied** during Run B.

---

## 3. Results, Gradients & Loss Landscape Analysis

### 3.1 Loss Landscapes (`figures/loss_landscape.png`)
- **Run A (Batch Monolithic)**:
  Projecting the weight trajectory onto the top two PCA directions reveals four discrete clusters situated at the exact same loss contour ($\log_{10} \text{loss} \approx -3.55$). Between these clusters are long ballistic jumps.
  The global batch objective possesses a degenerate valley corresponding to a trivial pseudo-equilibrium: $x(t) \to 0, y(t) \to 1.7, z(t) \to 0$. Because $\dot{x} = -0.1 y z = 0$, $\dot{y} = 1.6 x z = 0$, and $\dot{z} = -0.75 x y = 0$, this constant line produces an ODE residual of near zero over 90% of the trajectory! The network minimizes the mean squared residual over the full horizon by collapsing into this wrong attractor.
- **Run B (Causal Windows)**:
  The landscape displays a clean, monotonic funnel plunging down to $\log_{10} \text{loss} \approx -6.8$. By decomposing the domain into short temporal slabs and imposing the causal exponential filter, the network is strictly prevented from optimizing future times before resolving earlier times.

### 3.2 Gradient Dynamics & Optimizer Instabilities (`figures/gradient_stability.png`, `layer_grad_norms.png`)
- **The Blow-Up Mechanism in Run A**:
  In Run A, the cosine similarity between successive snapshot gradients remained pegged at $+1.0$ for thousands of epochs while the gradient norm decayed from $10^4$ down to $10^{-4}$.
  This confirms that the optimizer was sliding along a nearly flat valley.
  In Adam, parameter updates are computed as:
  $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon_{opt}} m_t$$
  When gradients remain at $10^{-4}$ for thousands of iterations, the uncentered second moment $v_t$ decays to $10^{-8}$. As soon as the network encounters a minor curvature change, the gradient nudges slightly, but dividing by $\sqrt{v_t} \approx 10^{-4}$ scales the step size up by $10^3$!
  This triggered the three explosive parameter blow-ups observed at iterations ~9,000, ~23,000, and ~36,000. Following each blow-up, Adam descended right back into the exact same $-3.55$ plateau.
- **Gradient Balance in Run B**:
  In Run B, all four layer gradient norms remained tightly clustered within a factor of $5\times$ ($\sim 10^{-2}$ to $10^{-3}$ during Adam, dropping smoothly to $10^{-7}$ during L-BFGS). No layer suffered from vanishing or exploding gradients.

### 3.3 Neural Tangent Kernel (NTK) Spectrum (`figures/ntk_spectrum.png`)
- The NTK matrix $K = J J^T$ characterizes the convergence speed of residual modes:
  $$\frac{d r}{d t_{opt}} \approx - K r$$
- In both Run A and Run B, the eigenvalues of $K$ span 15 orders of magnitude ($10^2$ to $10^{-13}$) across ~600 modes.
- Crucially, the NTK eigenspectra are virtually identical at initialization and remain fixed throughout training.
- **Scientific Conclusion**: The failure of the batch PINN is **not due to spectral bias or network expressivity**. The network is fully capable of representing the solution. The failure is entirely attributable to the non-convex optimization geometry of the monolithic space-time loss.

### 3.4 Residual Evolution & Convergence Front (`figures/residual_evolution.png`)
- **Run A**: Appears as horizontal bands across time. The residual improves or worsens uniformly everywhere simultaneously.
- **Run B**: Shows a strict **causal staircase**. The residual drops to $10^{-5}$ sequentially from left to right. Once a window converges, its state is frozen and handed off, completely preventing backwards error contamination.

---

## 4. Addressing User Inquiries (`feedback.md`)

Below are detailed technical explanations for the five points raised in `feedback.md`.

### Question 1: Formulation of Trial Function in Windowing
> *"- $u_T(t) = u_0 + g(t)\mathcal{N}(t;\theta)$ --- ekhane $u_0$ er jaigai $u(t)$ hobe ki na, $u_{i+1}(t) = u_i(t) + g(t)\mathcal{N}(t;\theta)$"*

**Clarification**:
In multi-window time-domain decomposition (time-marching):
For window $i$ defined on $[t_i, t_{i+1}]$, the correct trial solution is:
$$u_T^{(i)}(t) = u_i(t_i) + g(t) \mathcal{N}^{(i)}(t; \theta_i)$$
where $u_i(t_i) \in \mathbb{R}^3$ is the **frozen terminal state vector** of the preceding window $i-1$ evaluated at time $t_i$.
It is a constant 3-dimensional vector, **not** the continuous function $u(t)$.
- If one were to write $u_{i+1}(t) = u_i(t) + g(t)\mathcal{N}(t)$, that corresponds to **residual boosting / Picard iteration** across the *entire* domain (training network $i+1$ to learn the residual correction of network $i$).
- For domain decomposition, each window has its own local time domain $[t_i, t_{i+1}]$, and the initial condition for that window is the discrete endpoint of the previous window.

---

### Question 2: Automatic Differentiation of Trial Solution
> *"- $r(t_i) = \dot{u}_T(t_i) - f(u_T(t_i))$ ekhane $\dot{u}_T(t_i)$ ei derivative ta kibhabhe hocche, as $u_T(t_i)$ itself just $x,y,z$ value."*

**Clarification**:
Although $u_T(t_i)$ evaluates numerically to values $(x, y, z)$, in PyTorch it is **not** an isolated numeric tuple; it is a node in a dynamic Directed Acyclic Graph (DAG) whose leaf is the tensor $t$ (with `requires_grad=True`).
1. Forward pass: $t \to \mathcal{N}(t) \to u_T(t) = u_0 + g(t)\mathcal{N}(t)$.
2. Reverse-mode automatic differentiation:
   ```python
   dudt = torch.autograd.grad(u[:, j].sum(), t, create_graph=True)[0]
   ```
3. Autograd executes the exact chain rule backward through the network layers:
   $$\frac{\partial u_j}{\partial t} = \frac{\partial g}{\partial t} \mathcal{N}_j(t) + g(t) \sum_{l} W^{(L)}_{j,l} \left( \sigma'(\dots) \dots W^{(1)} \cdot 1 \right)$$
4. Setting `create_graph=True` preserves the operations used to compute $\dot{u}_T$, enabling PyTorch to differentiate the residual loss $L_{res}(\theta) = (\dot{u}_T - f(u_T))^2$ with respect to the network weights $\theta$ during `loss.backward()`.

---

### Question 3: The Factor of $1/3$ in the Residual Loss
> *"- Residual loss er total loss e keno $u(t_i)$ 1/3 diye gun kora hocche? amra ki 1 - Nc loop e 3 kore increase korchi naki."*

**Clarification**:
The factor of $1/3$ is **not** an index increment of 3. It is the **arithmetic mean over the 3 ODE dimensions** $(x, y, z)$.
The Lorenz-1960 system consists of 3 coupled equations:
$$r_x = \dot{x} - f_x(x,y,z), \quad r_y = \dot{y} - f_y(x,y,z), \quad r_z = \dot{z} - f_z(x,y,z)$$
For $N_c$ collocation points, the residual tensor $r$ has shape $(N_c, 3)$.
In PyTorch:
$$\text{loss} = \text{mean\_squared\_residual}(r) = \frac{1}{N_c \times 3} \sum_{i=1}^{N_c} \left( r_{x,i}^2 + r_{y,i}^2 + r_{z,i}^2 \right) = \frac{1}{N_c} \sum_{i=1}^{N_c} \left[ \frac{1}{3} (r_{x,i}^2 + r_{y,i}^2 + r_{z,i}^2) \right]$$
The factor $1/3$ simply normalizes the sum of squares across the three physical variables so that the loss represents the mean squared error per equation component.

---

### Question 4: Architecture Formulation ($1 \to 60 \to 60 \to 60 \to 60 \to 3$)
> *"- 1 -> 60 -> 60 -> 60 ->60 -> 3 , ei formulation change kore dekhte, aro different combination kirokom result dei, plus figure eo fix kore dite hobe"*

**Clarification**:
The current network architecture is a Multi-Layer Perceptron (MLP):
- **Input layer**: 1 neuron ($t$).
- **Hidden layers**: 4 layers of 60 neurons each with Tanh activations.
- **Output layer**: 3 neurons ($x, y, z$).
- **Total parameters**: $1\times 60 + 60 + 3 \times (60 \times 60 + 60) + 60 \times 3 + 3 = 11,283$ weights and biases.

In the architecture grid sweep on $[0, 1]$ (`runs/comparison.csv`), depth $\in \{3, 4, 5\}$ and width $\in \{50, 60, 70\}$ were evaluated:
- Depth 4, Width 50: RMSE $9.62 \times 10^{-6}$ (7,903 params)
- Depth 4, Width 60: RMSE $1.73 \times 10^{-5}$ (11,283 params)
- Depth 5, Width 70: RMSE $1.45 \times 10^{-5}$ (20,233 params)
On short horizons, all these architectures achieve $\sim 10^{-5}$ accuracy.
On the long horizon $T = 13.26446$, increasing depth or width in the batch mode did not help at all (Run A failed completely even with 11,283 parameters). In windowed mode, 4x60 per window proved more than sufficient because each window only spans $\Delta t \approx 0.49$.

---

### Question 5: Computational Graph & Collocation Batching
> *"- Diagram modification (include more steps): first 3000 collocation ekta loop e run hobe till amra r(ti) peye jai for the full batch, than residual loss e move korbe. er ag porjonto loop cholbe."*

**Clarification**:
In vectorized PyTorch execution, collocation points are processed simultaneously as a single batch of size $(N_c, 1)$, rather than an explicit Python loop:
1. Generate Collocation Grid: $T_{colloc} = [t_1, t_2, \dots, t_{N_c}]^T \in \mathbb{R}^{N_c \times 1}$.
2. Forward Evaluation: Compute $N = \text{MLP}(T_{colloc}) \in \mathbb{R}^{N_c \times 3}$.
3. Hard IC Application: $u = u_0 + (T_{colloc} - t_0) \odot N \in \mathbb{R}^{N_c \times 3}$.
4. Batch Autograd Derivative: Compute $\dot{u} = \frac{\partial u}{\partial T_{colloc}} \in \mathbb{R}^{N_c \times 3}$.
5. Physical RHS Evaluation: Compute $f(u) \in \mathbb{R}^{N_c \times 3}$.
6. Residual Tensor Construction: $r = \dot{u} - f(u) \in \mathbb{R}^{N_c \times 3}$.
7. Causal Weighting: Sort $T_{colloc}$, compute temporal weights $w \in \mathbb{R}^{N_c}$, and calculate scalar loss $\mathcal{L} = \frac{1}{N_c} \sum_{i=1}^{N_c} w_i \left( \frac{1}{3} \sum_{c=1}^3 r_{i,c}^2 \right)$.
8. Optimization Step: Compute $\nabla_\theta \mathcal{L}$ and update weights via Adam/L-BFGS.

---

## 5. State-of-the-Art (SOTA) PINN Implementation Strategy

To push solution accuracy from $10^{-4}$ to $10^{-7} - 10^{-9}$ and eliminate all observed defects, the following seven enhancements should be implemented:

### 1. Enforce $C^1$ Continuity via Physics-Preserving Trial Functions
Replace the first-order trial function with the second-order Taylor-augmented trial form:
$$u_T(t) = u_0 + (t - t_0) f(u_0) + (t - t_0)^2 \mathcal{N}(t)$$
- At $t = t_0$: $u_T(t_0) = u_0$ (Exact Initial State).
- At $t = t_0$: $\dot{u}_T(t_0) = f(u_0)$ (Exact Initial Velocity / Physics Derivative).
- Residual at boundary: $r(t_0) = \dot{u}_T(t_0) - f(u_T(t_0)) = f(u_0) - f(u_0) \equiv 0$.
This structurally eliminates the $10^{-3}$ residual spike at the start of every window.

### 2. Implement Normalized Causal Loss
Normalize the cumulative loss by the point count $N_c$ or interval length $\Delta t$:
$$w_i = \exp\left( -\frac{\epsilon}{N_c} \sum_{k=1}^{i-1} L_k \right)$$
This removes the density-dependence bug, ensures that the stopping condition $\min w > 0.99$ reflects the true physical residual error, and eliminates the 4,000-iteration stalling behavior in high-velocity windows.

### 3. Incorporate Conserved Quadratic Invariant Loss Terms
Lorenz-1960 conserves two independent quadratic forms:
$$I_1(x,y,z) = \alpha_1 x^2 + \beta_1 y^2 + \gamma_1 z^2, \quad I_2(x,y,z) = \alpha_2 x^2 + \beta_2 y^2 + \gamma_2 z^2$$
Add an invariant penalty to the loss function:
$$\mathcal{L}_{total} = \mathcal{L}_{res} + \lambda_{inv} \sum_{j=1}^2 \frac{1}{N_c} \sum_{i=1}^{N_c} \left( \frac{I_j(u(t_i)) - I_j(u_0)}{I_j(u_0)} \right)^2$$
Enforcing invariant conservation acts as a Riemannian manifold regularizer, preventing the network from decaying into false off-manifold attractors.

### 4. Random Fourier Feature (RFF) Embeddings
To overcome spectral bias in high-frequency and chaotic dynamics (Tancik et al., 2020; Wang et al., 2021), embed the scalar time input $t$ into a Fourier feature space:
$$\gamma(t) = \left[ \cos(2\pi B t), \sin(2\pi B t) \right]^T$$
where $B \sim \mathcal{N}(0, \sigma^2)$. This flattens the NTK eigenvalue decay and enables rapid learning of oscillatory modes.

### 5. Multiplicative Gating Architecture (Modified MLP)
Adopt the Wang, Yu & Perdikaris (2021) architecture with two transformer-style gating encoders:
$$U = \phi(t W_u + b_u), \quad V = \phi(t W_v + b_v)$$
$$H^{(l+1)} = (1 - \phi(H^{(l)} W_l + b_l)) \odot U + \phi(H^{(l)} W_l + b_l) \odot V$$
This architecture prevents gradient vanishing in second-order autograd graphs and facilitates deeper models without optimization collapse.

### 6. Residual-Based Adaptive Refinement (RAR)
Instead of uniform collocation points, employ adaptive sampling:
1. Sample a fine candidate grid of 100,000 points.
2. Evaluate residual $|r(t)|$.
3. Dynamically append the top $K$ points with the largest residuals to the training batch.
This concentrates network capacity precisely where the dynamical velocities and curvatures are highest.

### 7. Overlapping Schwarz / XPINN Domain Coupling
Rather than pure sequential hand-offs where errors can accumulate over long multi-orbit horizons ($T > 50$), implement Overlapping Domain Decomposition (XPINN / Schwarz coupling) with continuity penalties in the overlap zones $\Omega_k \cap \Omega_{k+1}$:
$$\mathcal{L}_{interface} = \|u^{(k)}(t) - u^{(k+1)}(t)\|^2 + \|\dot{u}^{(k)}(t) - \dot{u}^{(k+1)}(t)\|^2 \quad \text{for } t \in \Omega_k \cap \Omega_{k+1}$$
This allows bidirectional error communication and guarantees $C^1$ smoothness globally across the entire trajectory.

---

## 6. Summary of Action Items

1. **Bug Fix 1**: Update `causal_loss` in `losses.py` to normalize by $N_c$ (resolving the Adam stall in windows 9–11 and 23–25).
2. **Bug Fix 2**: Upgrade `hard_initial_condition` in `trial.py` to include $(t - t_0) f(u_0)$ (eliminating boundary residual spikes).
3. **Bug Fix 3**: Update `closure()` in `train_windows` to pass the final causally weighted loss or ensure causality is verified prior to L-BFGS.
4. **Bug Fix 4**: Fix `u0` and `coeffs` buffer dtypes in `PINN.__init__` to match `cfg.torch_dtype`.
5. **Optimization 5**: Vectorize `time_derivative` to eliminate redundant autograd backward passes.
6. **Feature 6**: Add invariant conservation loss $\mathcal{L}_{inv}$ to guarantee physical fidelity.
