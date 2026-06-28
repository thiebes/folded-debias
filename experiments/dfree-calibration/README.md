# dfree-calibration

## Purpose

Establish that the bias calibration is **D-free**: built at `D = 0`, keyed on
per-frame CNR alone, so the application's true D is never used. Includes the
dense calibration + four-scheme / three-shape test backing the corrected-bias
curves.

## Status

Validated. Poster-scope (source for Panel 5 corrected-bias curves and the
MAS-vs-folded recommendation).

## Contents

- `validate_d_zero.py [profile_model]` -- D-independence test: one calibration
  built at `D = 0`, applied across `D in {0.005, 0.01, 0.02, 0.05, 0.1}` on
  Gaussian / Voigt / generalized-Gaussian.
- `dense_calibration_and_test.py` -- dense (100 log-spaced CNR0) D-free
  calibration then a random log-distributed test, all four schemes (`ols`,
  `inverse_variance`, `folded_normal`, `mas`) on all three shapes. Wall time
  ~3 h.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py
.venv/Scripts/python.exe experiments/dfree-calibration/dense_calibration_and_test.py
```

Set `WEIGHTING_WALL_WORKERS=0` for the dense run. Outputs to repo-root `output/`;
committed snapshots are in this folder.

## Key result + figure

In the poster regime (`D >= 0.02`) the corrected mean `|bias-1|` is 0.01-0.10 per
shape. Dense folded-normal overall mean: gaussian 0.119, voigt 0.175, gengauss
0.155; `folded_normal` is the robust default (`mas` overshoots at `CNR0 < ~4`).
Result tables: `validate_d_zero_<model>.txt`, `dense_calibration_and_test.txt`.

## Depends on

`weighting_wall`. No sibling-folder helpers.

## See also

`docs/STATUS.md`, `docs/DERIVATION.md`, `docs/poster/POSTER_PLAN.md` Panel 5.
