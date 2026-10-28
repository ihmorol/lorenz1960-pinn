# Causal 1536-point verification

The previous batch-case viewer is replaced by the canonical 1536-point case.
27 networks and 41,446 unique collocation times (1536 points per window including
shared endpoints). Evaluated saved checkpoint in float64 against the baseline
reference on 13,265 dense points. Measured combined state RMSE reproduces the
saved value 6.96696531e-5 within relative tolerance 1e-4 and absolute tolerance
1e-9. All 26 endpoint handoffs match within absolute tolerance 1e-12.

Verified 27 increasing window completion marks and actual reference-error history
coverage for every window. Frozen final-network prefixes are reconstructed;
there are no local in-window trajectory or per-parameter snapshots for this run.
The summary reports 43 snapshots, but their CSVs are not present locally.
Layer view shows final parameter values only. No invented gradients or updates.

Browser observed the successful full trajectory by default, RMSE 6.967e-5,
window-27 recorded learning history, and window-1 final weights following the
window selection. No training code or original run data changed. Twelve separate
reference equation cases retain invariant and amplitude/time-scaling checks.
No full training suite, frame-rate benchmark, or complete browser matrix run.


## Window learning revision

Validated all 67,908 original objective records without dropping optimizer phases.
Each of 27 windows has local indices, Adam counts matching stages.csv, contiguous
L-BFGS evaluations, and reference samples checked against its coverage interval.
The model checkpoint and dense RMSE check remain unchanged.

Browser checks: large 3D view with side controls and direct color legend; hide
reference independently; selected-window zoom; local evaluation 3000 reports
Adam update 3001 with epsilon 10 in window 27; local evaluation 4500 reports
L-BFGS evaluation 481. Window 1 has 1457 Adam updates and 1250 L-BFGS evaluations.
120-second playback advances and pauses; all underlying objective rows remain
in the embedded data. Playback cursor moves between recorded indices using
elapsed time; no synthetic objective values or trajectory frames are generated.

Window 27 explicitly shows 12.77318 to 13.26446; total interval is independently
shown as 0 to 13.26446. Recorded RMSE states its last sampling index; full-run
final checkpoint RMSE is separate. Narrow viewport uses a collapsible side drawer;
desktop uses a fixed sidebar. Final rendering had no recorded console errors.
Intermediate per-window parameter checkpoints are still unavailable locally.
