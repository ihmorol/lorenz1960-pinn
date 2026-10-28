# Window learning playback revision

1. [complete] Confirm canonical causal 1536 PT history: all loss rows, optimizer phases,
   stage counts, and availability of intermediate parameter checkpoints.
2. [complete] Add each window's full loss history and locally indexed reference samples;
   validate counts against stages.csv and increasing completion boundaries.
3. [complete] Enlarge 3D rendering; add visible reference/prediction labels and independent
   visibility controls. Add selected-window zoom and explicit final-state label.
4. [complete] Add a per-window learning cursor with 30/60/120-second playback, phase labels,
   epsilon boundaries, and latest recorded metrics without interpolation.
5. [complete] Numerical self-check and browser QA: labels, plotting size, window/phase
   navigation, pause/speed, cursor boundaries, and absence of invented geometry.

Unavailable: intermediate parameter/trajectory snapshots. The 3D view shows
completed states; only recorded scalar histories replay during optimization.
