# The whole project, in plain words

This file explains what we built and what we found. Short sentences. No jargon
where we can avoid it. For the full line-by-line story, see `CODE_EXPLAINED.md`.

## The problem

Lorenz wrote down a small weather model in 1960. It has three numbers: x, y, z.
They change over time. Each one's speed depends on the other two:

- x changes at a rate of -0.10 times y times z
- y changes at a rate of 1.60 times x times z
- z changes at a rate of -0.75 times x times y

We start at x = 0.5, y = 0.75, z = 1.0. We want to know the values from time 0
to time 1. There is no simple formula for the answer. You have to compute it.

## Two ways to solve it

The classic way: step through time in tiny steps. Our baseline does this with
RK4 and with SciPy's DOP853 solver. Both agree with each other. We treat the
SciPy answer as the truth.

The new way: train a neural network. This is the PINN, short for
Physics-Informed Neural Network. The trick is that we never show it the answer.
We only tell it the rules above. The network guesses a solution, we measure how
badly the guess breaks the rules, and we nudge the network until the rules hold.

That rule-breaking score is called the residual. Training means making the
residual small at 3000 random time points.

One more trick. We do not ask the network to learn the starting point. We build
the starting point into the output formula, so it is always exactly right. The
network only learns how the curve moves away from it.

## What each file does

- `src/baseline/lorenz1960_baseline.py` holds the equations and the trusted
  solver. One source of truth. The PINN imports from it and never edits it.
- `src/pinn/config.py` is the settings sheet. Network size, training length,
  everything. Change a setting there, no code edits needed.
- `src/pinn/pinn.py` is the network itself, plus the residual and the loss.
- `src/pinn/train.py` runs training: 20000 rounds of Adam, learning rate
  sliding from 1e-3 down to 1e-4. L-BFGS polish exists but is off by default.
- `src/pinn/history.py` writes down what happened during training, so we can
  make plots later without training again.
- `src/pinn/figures.py` draws all the figures.
- `src/pinn/test_pinn.py` has 9 tests. They check the start point is exact,
  the physics is coded right, training works, and every figure gets written.

## What we found

The network solves the system well. Compared to the trusted SciPy answer:

- average error per variable is about 5 to 11 millionths (1e-6 to 1e-5)
- the worst single error anywhere is about 26 millionths (2.6e-5)
- the training loss falls from around 1 to below 1e-8

For scale: the values of x, y, z are around 0.5 to 1.6. So the network is right
to about five decimal places, using only the rules, never the answer.

The curves in `src/pinn/results/` show the same story. The PINN line sits on
top of the reference line. You cannot see a gap by eye. The error plot is the
interesting one, because the gap only shows up when you zoom in a million times.

Two caveats, so nobody oversells this:

- The network uses 32-bit numbers. That likely sets the floor around 1e-5 to
  1e-6. More training would not help much past that.
- This is one starting point and one time window. A new starting point means
  training again. The classic solver does not have that problem. On this small
  problem the classic solver is also much faster. The point here was to show
  the method works and to have a solid base for the next phase, where we
  compare network designs.

## How to see it yourself

```
pip install -r requirements.txt
python -m pytest          # checks, takes about half a minute
python run_pinn.py        # full training run, writes all results and figures
```

Results land in `src/pinn/results/`. The training record and the trained
network land in `src/pinn/history/`.
