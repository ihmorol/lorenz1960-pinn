# Review of previous edits

The previous DOP853-only description, seed groups, snapshot interval, invariant definitions, exploratory Mann-Whitney test, corrected references and scoped conclusions are retained. No supported result, numbered equation, table or algorithm is removed or restored selectively. Claims removed for lack of evidence are not restored to fill space.

A targeted code check found that the preceding prose overstated the frequency of post-update causal-weight certification. src/pinn/train.py:310-317 certifies updated weights before accepting a stage transition (when the pre-update condition or cap triggers that check), rather than unconditionally after every update. The new wording describes the verified transition criterion and capped outcome without that incorrect frequency claim. All 1,536 window points, including the outgoing endpoint, participate; no sampling change is made.

The regenerated pipeline uses k=0,...,26 consistently with the equations and algorithm, replacing its old one-based display caption. New figure geometry is authorized by the latest user instruction; prior figure-geometry constraints are superseded for this revision.

The later compact method diagram was rejected by the author. Its original detailed branching structure is now restored at greater height; no prior scientific correction was undone. See method-restoration.md.
