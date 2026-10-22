# Caption: real-data panel (MoSSe alloy)

Material: **MoSSe alloy** (Mo(S,Se)2; Enrique Arevalo Rodriguez's stroboSCAT
sample `250716_MoSSe_mystery`, `mixed_tmdc/26_32/250716`). Not WSe2.

The truncation-bias correction on a real diffusion measurement, in physical
units. Each of the two interleaved delay sweeps is re-zeroed to its own t0
(de-parity), the merged MSD is converted to um^2 with the 0.060 um pixel at the
sample (0.0036 um^2/px^2, from Enrique Arevalo Rodriguez), and the per-frame
widths are averaged
in inverse-variance-weighted time bins over the early diffusive window
(t <= 2 ns). Points are the binned naive fitted sigma^2 (red, open) and the
corrected sigma^2 (blue, filled) at the same bin time; error bars carry the
within-bin scatter. Binning removes the per-frame scatter and the even/odd parity
wobble -- the systematic bias survives averaging, the noise does not -- so the
correction shows on every binned point.

Result: **naive D = 0.0116, corrected D = 0.0113 um^2/ns, ratio 0.97** -- the
correction pulls D down by ~3% on this real measurement, consistently across the
binned points (over the full retained range to ~7 ns the pull is ~5%). Fit over
the 16 frames with per-frame CNR >= 1.3 and t <= 2 ns (t = -0.1..1.8 ns, CNR
2.5..18). Below-CNR-floor late frames (where the correction over-subtracts -- the
documented limitation) are excluded.

Message: on real data the correction removes a small, systematic upward bias in D
that averaging alone cannot -- "replication reduces variance, not bias." The
correction is modest here because the reliable frames sit at moderate CNR; the
large bias in the deep-low-CNR regime, where truth is known, is shown by the
simulation recovery demo (`diagnostics/_frame_ladder/0_simulated_benefit.png`,
`0b_simulated_distribution.png`).

The pixel scale is not taken from the ratio of Enrique's fitted sigma (um) to the
px fit: that ratio depends on the fit model and window and is not a calibration.
The ratio 0.97 does not depend on the scale.

Style: Spanish-flag palette (flag red = naive/biased, flag blue = corrected/
reference), transparent background. Rendered by `render_realdata_panel.py`.
