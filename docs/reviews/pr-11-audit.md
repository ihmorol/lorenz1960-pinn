# PR #11: causal-window pipeline audit

Reviewed source: [PR #11](https://github.com/ihmorol/lorenz1960-pinn/pull/11),
head `969a296066c437ad1efe3a151430b49d209864e1` against `main`.
This report distinguishes what the saved run proves from what its training
configuration suggests. The PR is integrated into the review branch, not main.

## Claims checked

| Claim | Evidence and verdict |
|---|---|
| 27 independent warm-started 4x60 tanh networks over `t=[0,13.26446]` | Confirmed by config, model structure, and final checkpoint. |
| 1,536 effective uniform points in each window | Confirmed: 41,446 shared-grid records produce 1,535 interior points plus the outgoing endpoint in windows 0-25, and 1,536 points in the last window. |
| Epsilon `(0.01, 0.1, 1, 10)`, threshold `0.99`, 4,000 Adam steps per stage, up to 1,000 L-BFGS iterations per window | Confirmed by config and saved stage/history records. The two capped stages below did not meet the threshold. |
| Selected run has the lowest *combined RMSE* among the historical, 1,500-point, and 1,536-point saved candidates | Confirmed: `8.4346e-5`, `1.0784e-4`, and `6.96696531e-5`, respectively. The historical run has a smaller recorded maximum error than the selected run. |
| Final checkpoint reproduces the saved trajectory | Confirmed by reloading and evaluating against a 13,265-point DOP853 reference: combined RMSE `6.96696531e-5`, maximum combined state error `3.09323169e-4`. The saved `results.png` matches the user-supplied final plot byte-for-byte. |
| Completed run can resume intermediate training and regenerate all training-time figures from committed files | Not claimed by the PR. Its `progress.pt`, parameter trail, and per-step breakdown were absent, so only final checkpoint reload is supported by those files. |

The plot's `1.62e-9` MSE is a component-averaged squared error; it is not the
combined L2 RMSE. Low physics residual alone is not evidence of an accurate
trajectory. The reloaded trajectory-to-reference error is the decisive test.

## Important limitations

- Windows 10 and 11 reached the 4,000-step Adam cap at epsilon 10 with saved
  minimum causal weights `0.9718467` and `0.9031337`, below `0.99`.
  L-BFGS finished afterward, so the run is complete but those stages did not
  satisfy the causal advancement criterion.
- The 1,500- and 1,536-point candidates changed both point count and L-BFGS
  budget (500 versus 1,000) and used different devices. Their result does not
  isolate a point-count benefit or support a controlled speed comparison.
- The saved result is one seed and one orbit. It does not establish robustness
  across seeds, initial states, horizons, or hardware.
- The original PR had absolute Colab storage paths in `config.json`, so
  `run_causal.py` failed its completed-run reuse check after relocation. The
  review branch compares only scientific settings for reuse, rebases loaded
  progress snapshots, and tests relocation explicitly.
- The original PR mixed historical runners, generated caches, and multiple
  experiments with the selected pipeline. The review branch keeps the final
  entry point and separates three key runs from a historical archive. Missing
  active-run progress and sequential-walk checkpoint files were not invented.

## Verification on the review branch

The complete Python suite passed (43 tests). The final and shared-network
checkpoints were evaluated from their own saved artifacts. The shared-network
result reproduces its recorded combined RMSE `0.398124876` within `5.26e-11`.
All archived checkpoints were reloaded, notebook JSON parsed, Python modules
compiled, and the staged diff passed whitespace checks. Full Colab GPU
retraining was not performed.
