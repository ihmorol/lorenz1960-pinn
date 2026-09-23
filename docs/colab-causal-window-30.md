# Colab launch: Lorenz-1960 [0, 15]

This is a **long-horizon** 30-window experiment, not a closed-orbit run. The
code is on `feat/causal-window-30`; the notebook pins code commit
`57172f6e00f6b537ca3b78f27f5dc21b271b7f21` and checks the commit after
checkout. Do not run training, numerical tests, evaluation, or figures locally.

1. Open [`colab/causal_window_30.ipynb`](../colab/causal_window_30.ipynb) in
   Google Colab from the pushed branch.
2. Select **Runtime → Change runtime type → T4 GPU** (or another available GPU).
3. Run the cells in order. Authorize the Drive mount to store the repository and
   checkpoints in `MyDrive/lorenz1960-window30/repo`.
4. The `--verify-only` check and experiment-relevant numerical tests must pass
   before the long training cell. The full repository test suite is separate;
   it includes sweep and plotting checks outside this experiment.
   The preflight checks all 30 counts against 1,536 effective training points, including
   each nonfinal outgoing endpoint. It also verifies 128 fixed independent
   Gauss diagnostic times per window.
5. If Colab disconnects, reconnect with the same GPU model and rerun the cells.
   The runner restores `history/progress.pt`, including the active window,
   optimizer and scheduler states, RNG states, counters, and compact diagnostics.
   Up to 999 Adam updates or 24 L-BFGS iterations can be lost between checkpoints.
6. After completion, the final cell writes
   `MyDrive/lorenz1960-window30/causal-window-30-results.zip`. Copy the final
   checkpoint, `history/config.json`, CSVs, `metadata.json`, `report.md`, and
   `figures/` from that ZIP into
   `runs/causal-window/long-horizon-30/seed0/` on this branch. Commit only after
   checking the recorded commit, device, versions, and segment metrics.

The 15,001-point evaluation grid contains the exact historical boundary
13.26446: 13,265 times on `[0, 13.26446]`, and 1,736 further times on
`(13.26446, 15]`. The historical checkpoint and the new model are reevaluated
at the **same prefix times**. The generated `region_summary.csv` reports prefix,
extension, and whole-domain RMSE, maximum state error, dense residual MSE, and
invariant drift separately. The reference solution is absent from the Adam and
L-BFGS objectives and causal stopping rule.

The run is one bundled configuration. Differences from historical runs cannot
be assigned to the changed horizon, density, window count, or optimizer budget
individually.
