# Caption: real-data CNR frame ladder (benefits and limitations)

Individual per-frame profile overlays (raw 1-D profile + naive Gaussian +
corrected Gaussian) from the real stroboSCAT scans, ordered by per-frame CNR, to
show where the correction helps and where it fails:

- `1_cnr57.0_*` -- high CNR: no bias, naive and corrected coincide (the
  correction leaves clean data untouched).
- `2_cnr04.0_*` -- moderate CNR: small real bias, small correction.
- `3_cnr01.4_*` -- near the floor: large correction (naive sigma^2 12.7 ->
  corrected 4.7); the data are noisy but a peak is discernible.
- `4_cnr01.3_*` -- at the floor: strong correction, edge of usefulness.
- `5_cnr00.5_*` -- below the floor: the profile is pure noise, the corrected
  sigma^2 goes negative (over-subtraction) -- the failure mode, and why the floor
  exists.

Rendered by `realdata_frame_ladder.py` (which reads each measurement's
`manifest.csv`). On real data the correction's effect is small where the data is
trustworthy and the bias only becomes large at low CNR where the data is noisy;
the clean "recovers the true width" claim is carried by the simulation figures
in this folder.
