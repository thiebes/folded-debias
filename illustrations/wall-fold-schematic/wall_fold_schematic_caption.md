# Caption -- wall-and-fold schematic

Low contrast bites: a one-directional bias.

Left -- the fold at the wall. A fitted width cannot be negative, so at low CNR the
fitted-variance distribution is bounded below at zero. The forbidden negative tail
(variance < 0) folds back across the wall as a reflected tail; the realizable
estimate and the reflected tail pile up near the wall, so the distribution of fits
(their sum) is right-skewed and its mean sits above the true variance. That gap is
the per-frame bias.

Right -- the tilted slope. The per-frame bias grows as CNR falls across the late
frames, lifting the late-time mean-squared displacement above the true line. The
inflated fit has a steeper slope, so the extracted D is overestimated. Crucially it
is a bias, not scatter: averaging repeated measurements does not remove it.

Pure schematic.
