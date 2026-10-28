# Training film

Palette: Classic 3B1B (background #1C1C1C; reference grey at 0.4 opacity; network BLUE to RED by error; gradient arrow YELLOW).

Act 1, LearningTheLoop (~30 s): the reference orbit is drawn once, then the network's curve is redrawn at each snapshot epoch with each point coloured by log10 error while the camera orbits slowly. What to watch: the curve collapsing onto the fixed point, then snapping onto the loop.

Act 2, DescendingTheSurface (~30 s): the PCA loss surface as a 3-D mesh; the Adam path traced epoch by epoch with a yellow arrow for the projected step direction. What to watch: the arrow shrinking and turning as the path crosses the ridge.

Act 3, TheCausalFront (branch B, ~20 s): the temporal weights w(t) drawn as one curve per snapshot, stacked along the iteration axis, sweeping forward as eps advances.
