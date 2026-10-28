# Verification notes, 1 October 2026

Code SHA: 0c5f9c86c597bd27897632fb917d6d17548eeffc.
Checkpoint SHA-256: d58fbc51d2f48a8af9cd21090378f7fdb8a8ef29bce80a11911ba379532b1456.
Environment: Python 3.13.2, PyTorch 2.9.1+cpu, SciPy 1.16.2, Windows. Audit scripts use CPU and do not retrain.

## Completed evidence

- scripts/check_saved_causal.py: exit 0; metric agreement, local/global grid sizes, continuity asserted; final weights and joint derivatives measured. Output checkpoint-verification.json.
- scripts/check_defense_reference.py: exit 0; manually specified RHS compared with tighter DOP853, Radau and RK4 refinement; explicit invariants checked. Output reference-verification.json.
- Complete existing pytest suite: 41 collected, 40 passed, 1 failed. Command used torch.set_num_threads(1) to reduce CPU oversubscription. This is verification environment configuration, not a training change.
- Subagent training audit: 16 targeted numerical/window regressions passed.
- Saved results.png visually inspected: metric labels and training-plot interpretations reviewed.
- Actual ignored main.tex, bibliography, TEAM_WRITING_GUIDE and EVIDENCE_MAP read; no edits made there.

## Existing failure

src/pinn/test_pinn.py:233, test_sweep_writes_runs_and_comparison:

```text
assert len(list((run / "breakdown").glob("epoch_*.csv"))) == 3
AssertionError: assert 4 == 3
```

For epochs=12 and snapshot_every=6, source writes epoch 0, epoch 6, last pre-update epoch 11, and final post-update epoch 12. The three-file expectation does not match the current four-state behavior. Do not delete a final snapshot merely to obtain a green suite; reconcile the intended snapshot contract in a separate fix.

Seaborn emitted a pending-deprecation warning. Joint figure code also emitted a requires-grad tensor-to-scalar warning; this is not a failed numerical assertion. No project lint configuration was found in pyproject.toml (only pytest settings).

## Review scope

Only audit reports, JSON evidence and two standalone check scripts were added. Training source, tests, checkpoints, manuscript and external project checkout were not changed. No new training, paid-model experiment, publication or external message was performed. Missing archived breakdown/param-trail/progress state prevents complete historical visual reconstruction.
