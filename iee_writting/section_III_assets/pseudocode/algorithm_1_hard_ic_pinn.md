# Algorithm 1: Hard-Initial-Condition Lorenz-1960 PINN

Use this as a compact, language-neutral pseudocode block in Section III. It is a
method aid, not a source-code listing. Keep the final typeset version to roughly
18--25 lines so it fits one IEEE column.

```text
Input: k=2, l=1; u0=(0.5, 0.75, 1.0); [t0, tf]=[0, 1]
       Nc=3000; E=20000; eta0=1e-3; etaf=1e-4; seed=0
Compute (cx, cy, cz) from the Lorenz-1960 coefficient formulas.
Sample and fix {ti}_(i=1)^Nc with a Latin-hypercube design on [t0, tf].
Initialize the 1-60-60-60-60-3 tanh MLP N(t; theta)
with Xavier-uniform weights and zero biases.
for epoch = 1,...,E do
    uT(ti) <- u0 + ((ti-t0)/(tf-t0)) N(ti; theta)
    dudt(ti) <- automatic_derivative(uT(ti), ti)
    f(ti) <- (cx*yT*zT, cy*xT*zT, cz*xT*yT)
    r(ti) <- dudt(ti) - f(ti)
    L(theta) <- (1/(3*Nc)) sum_i ||r(ti)||_2^2
    theta <- AdamUpdate(theta, grad_theta L, eta(epoch))
    eta(epoch) <- linear_decay(eta0, etaf, epoch, E)
end for
Evaluate uT on a separate 1001-point uniform grid.
Compute component-wise and combined errors against the DOP853 reference.
Report the dense-grid residual norm ||r(t)||_2 separately from the training loss.
```

Do not include the DOP853 comparison inside the optimization loop: in this
project the numerical reference is post-training evaluation only. Do not add an
L-BFGS stage to the reported algorithm because `lbfgs_iters=0` in the run of
record.
