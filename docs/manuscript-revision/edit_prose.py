from pathlib import Path
p=Path('writings/conference-paper/main.tex');s=p.read_text()
def tex(x):return x.replace('§',chr(92))
def replace(old,new):
 global s
 old,new=tex(old),tex(new)
 assert old in s,old[:90]
 s=s.replace(old,new,1)
def para(start,new):
 global s
 start=tex(start);a=s.index(start);b=s.index('\n\n',a)
 s=s[:a]+tex(new)+s[b:]
# Keep original narrative except planned definitions, repeated announcements, and evidence corrections.
for start in ['In this section we introduce','This section explains the implementation','This section explain the experimental setup','This section reports the measurements']:
 para(start,'')
replace('Suppose the state vector is given by $u=[x,y,z]^§mathsf{T}$.',r'''Here $§kappa$ and $§ell$ are physical wavenumbers; $k$ below indexes windows. We write the state as $u=[x,y,z]^§mathsf{T}$. Substituting $§kappa=2$ and $§ell=1$ into §eqref{eq:lorenz1960} gives the three coefficients.''')
replace('§label{eq:ode}\n§end{equation}',r'''§label{eq:ode}
§end{equation}
Collecting these right-hand sides gives the prescribed vector field $f$, which maps a three-component state to its three rates of change:''')
para('Each window has its own network',r'''Each window has its own network $N_k(t;§theta_k)$ with one time input, four hidden layers of 60 tanh units, and three output components. These outputs form a correction vector; §eqref{eq:trial} gives the predicted state. Including biases, the input, hidden-to-hidden, and output connections give the following parameter counts.''')
para('As Lorenz system is an initial value problem',r'''For window $k=0,§ldots,26$, $t_k$ is its starting time and $u_{0,k}§in§mathbb{R}^3$ its fixed starting state. As the Lorenz system is an initial value problem, we enforce this state through a trial solution.''')
para('where $k=0,',r'''At $t=t_k$, the scalar factor $t-t_k$ is zero, so $§hat{u}_k(t_k)=u_{0,k}$ regardless of the parameters~§cite{lagaris1998}. The first window uses the prescribed initial state; every later window uses the predicted endpoint of the previous window.''')
para('$§theta_{k+1}§leftarrow',r'''The endpoint transfer in §eqref{eq:handoff} makes the predicted values continuous at window boundaries, though derivatives can change there. Separately, we copy $§theta_{k+1}§leftarrow§theta_k$ and train the next network independently. As later windows start from predictions, errors from earlier windows can propagate forward.''')
replace('§caption{Complete implementation pipeline}',r'''§caption{Sequential training and endpoint transfer. The figure numbers windows 1--27; equations use 0--26. Reference values monitor error but do not enter optimization or stopping.}''')
para('To predict and train the networks',r'''We train each window on $n_k=1536$ uniform collocation times $t_{k,i}$, ordered by $i=1,§ldots,n_k$ and including both endpoints.''')
# This paragraph contains the next paragraph without a blank separator.
para('Ideally 1535 points',r'''Adjacent windows share boundary times. This gives 41,446 unique points and 41,472 point uses during training. Each batch has shape $(1536,1)$; the network correction and trial state have shape $(1536,3)$. We apply the trial solution and use PyTorch automatic differentiation to compute its derivative component by component. With $u_{0,k}$ fixed, the product rule gives''')
replace('§begin{equation}\nr_k(t)',r'''The ODE residual function compares this derivative with the original vector field $f(§hat{u}_k(t))$ evaluated at the predicted state. Both terms are three-component vectors:
§begin{equation}
r_k(t)''')
replace('§[\n§mathbf{t}_k',r'''For a batch, rows represent times and columns follow the $x,y,z$ order. Network corrections, trial states, derivatives, vector-field values, and residuals are denoted by $§mathbf{N}_k$, $§hat{§mathbf{U}}_k$, $§dot{§hat{§mathbf{U}}}_k$, $§mathbf{F}_k$, and $§mathbf{R}_k$:
§[
§mathbf{t}_k''')
para('For each ordered collocation time',r'''For each ordered time $t_{k,i}$, averaging the three squared residual components gives one nonnegative scalar point loss:''')
para('To solve a long horizon time dependent',r'''To give earlier times priority during training, we use causal weights~§cite{wang2024causal}. The prefix $S_{k,i}$ sums the point losses before time $i$ in window $k$. At the first point this sum is empty, so its weight is one:''')
replace('§begin{equation}\n§mathcal{L}_{k,§epsilon}',r'''We multiply each point loss by its weight and divide by the number of points. This gives one scalar objective for window $k$ at the current $§epsilon$ stage:
§begin{equation}
§mathcal{L}_{k,§epsilon}''')
para('Here $n_k=1536$.',r'''Here $j<i$ indexes earlier points. Each of $§mathbf{L}_k$ and $§mathbf{w}_k$ contains 1536 scalar values. The parameter $§epsilon$ controls how strongly earlier losses suppress later weights. As earlier losses decrease, later weights rise toward one. We recompute weights during training and detach them when calculating parameter gradients.''')
para("Only using ODE residual isn't enough",r'''Later states depend on earlier states, and the windowed model cannot backtrack after a window is frozen. Reducing earlier error can therefore help later predictions. Equations~§eqref{eq:pointloss}--§eqref{eq:causalloss} implement this ordering; the ablations below test its effect for this orbit.''')
para('The causal loss defined in above',r'''The causal loss is minimized separately in each window. Figure~§ref{fig:pipeline} and Algorithm~§ref{alg:training} show the sequence. Each window starts a new Adam optimizer and StepLR scheduler. The learning rate starts at $10^{-3}$ and is multiplied by 0.9 every 1,000 updates, across the four $§epsilon$ stages. Each stage has a 4,000-step limit. The code verifies the post-update minimum over all 1536 weights; if it exceeds $0.99$, training advances to the next stage. Otherwise the stage ends at its cap and this is recorded. A capped stage does not by itself mean the trajectory failed. After all four stages, L-BFGS minimizes the unweighted residual mean for at most 1,000 iterations:''')
para('The code saves the parameters',r'''The code saves each window's parameters, transfers its endpoint state, and copies parameters to the next trainable window. Checkpoints store configuration, optimizer state, training history, and random states for resuming training. Incompatible configurations are rejected. DOP853 values monitor error but do not enter the loss, gradient, or stopping test. After training, the piecewise model is evaluated on collocation and evaluation grids, and the saved arrays provide error and invariant measures.''')
replace('and Distances away from training grid.', 'on a separate evaluation grid that shares endpoints with the training grid.')
replace('a absolute tolerance', 'an absolute tolerance')
replace('§subsection{Evaluation}\n',r'''§subsection{Evaluation}
At evaluation time $t_j$, we subtract the reference state from the prediction. This state error differs from the ODE residual:
''')
replace('§begin{equation}\n§begin{aligned}\n§operatorname{RMSE}_2',r'''The combined RMSE averages squared error norms over times, then takes the square root:
§begin{equation}
§begin{aligned}
§operatorname{RMSE}_2''')
replace('while the combined maximum error is $§max_j§|e(t_j)§|_2$.',r'''Here $c$ identifies a state component. The state-norm row of Table~§ref{tab:errors} reports $E_{§mathrm{mean}}$, $§operatorname{RMSE}_2$, and $E_{§mathrm{max}}$.''')
replace('The dense-grid residual mean square is the average of the three residual components squared, taken over the times of the evaluation.',r'''Unlike squared combined RMSE, residual MSE also divides by the three components:''')
replace('§input{ablation-table}',r'''§subsection{Ablation Protocol}
We use the completed suite in Table~§ref{tab:ablations}. All cases use the same initial state, coefficients, precision, interval, and reference grid. The candidate has five seeds, including the earlier seed-0 run with identical settings. Removing causal weighting uses 1,300 Adam steps per window (35,100 total), close to the candidate mean of 34,768. Removing warm start keeps the candidate schedule. The shared-network case uses the no-causal schedule but repeatedly updates one network. The causal single-network case uses the full-interval grid and budget: 39,793 LHS points, 40,000 Adam steps, and at most 5,000 L-BFGS iterations. Runs mix CPU and GPU execution, so we compare Adam steps rather than wall time.

§subsection{Failure Cases}
We test whether one shared network retains earlier windows and whether causal weighting alone resolves full-interval error. All reported runs completed; poor final prediction is distinct from interrupted training or a capped stage. Table~§ref{tab:ablations} reports the full range, including the two candidate seeds above $10^{-4}$. No success cutoff was used to select runs.

§input{ablation-table}
The candidate median RMSE is $8.13§times10^{-5}$, with three of five seeds below $10^{-4}$. Without causal weighting the median is 2.3 times larger, but that configuration is better for seed 2. Without warm start, all three matched seeds have larger RMSE and the mean Adam count rises to 102,989. Reusing one network gives RMSE 0.350--0.648, versus $1.75§times10^{-4}$--$5.16§times10^{-4}$ for separate networks with the same no-causal schedule. Causal single-network training gives 0.275 in its one run. These observations support the tested combination; they do not establish a universal causal-weighting gain.''')
replace('§caption{Windowed-run error against DOP853 on 13,265 times}',r'''§caption{Seed-0 error against DOP853 on 13,265 times. This is the lowest-RMSE candidate seed and the run shown in the figures.}''')
replace('The DOP853 reference drifted less than $4§times10^{-10}$ for perspective.',r'''For the conserved quantity $E=16x^2+y^2$, $dE/dt=32x(-0.1yz)+2y(1.6xz)=0$. The saved physics check gives maximum absolute drift $4.05§times10^{-4}$ for seed 0 and $1.76§times10^{-10}$ for DOP853.''')
replace('All of the above are uncontrolled types of ablation in the concept of causal weighting §cite{wang2024causal}, windowing §cite{penwarden2023} and warm starts as well as precision and even L-BFGS.', 'This full-interval comparison changes several settings together; Table~§ref{tab:ablations} separates selected changes.')
replace('The 27 run windows had 34,765 Adam updates and 1000 L-BFGS closure evaluations per window.', 'The seed-0 run had 34,765 Adam updates and 33,143 L-BFGS closure evaluations across 27 windows.')
a=s.index(tex('§section{Discussion and Limitations}'));b=s.index(tex('§section{Conclusion}'),a)
s=s[:a]+tex(r'''The experiments cover one initial state and one coefficient set. The candidate stores 27 times as many parameters as the single-network cases, and a capacity-matched full-interval network has not been tested. Causal weights also depend on collocation count and spacing, which remain fixed in the windowed ablations. Other states, coefficients, and grids may behave differently.

''')+s[b:]
replace('This results in significant increase in loss with RMSE of 2.237.',r'''The shared-network runs give RMSE 0.350--0.648 in Table~§ref{tab:ablations}; 2.237 belongs to the full-interval run.''')
replace('We reported, in a single-seed run', 'In the lowest-error of five candidate seeds')
replace('The full-interval and windowed runs also vary',r'''Across five seeds, the median RMSE is $8.13§times10^{-5}$ (range $6.97§times10^{-5}$--$2.60§times10^{-4}$). The full-interval and windowed runs also vary''')
replace('Future research will focus on matched configurations and initial states', 'Future research will focus on capacity-matched configurations and additional initial states')
p.write_text(s)
