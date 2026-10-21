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
  Gaussian / Voigt / generalized-Gaussian (`generalized_gaussian:1.5`,
  `generalized_gaussian:4`; the bare name is p = 2). Prints each cell's Monte
  Carlo SE.
- `dense_calibration_and_test.py` -- dense (100 log-spaced CNR0) D-free
  calibration then a random log-distributed test, all four schemes (`ols`,
  `inverse_variance`, `folded_normal`, `mas`). The default runs Gaussian,
  Voigt, and p = 2; pass specs to run others. Wall time ~3 h per shape.
- `calibration_se_check.py generate|analyse` -- tests `calibration_D_se`
  against 16 independent calibrations on the fft-cnr paper's grid (48 CNR0,
  1500 runs) applied to one fixed evaluation set; separates the bias and
  weight terms, checks per-point SE and within-ensemble correlation, and ranks
  the paper's three seeds. `generate` is about 5 h on 8 cores (cached in
  `output/calibration-se-check/`).
- `render_bias_panels.py` -- renders the poster bias panels from this set's
  `dense_calibration_and_test.txt`: Panel 3 (`poster_panel3_bias.png`, the two
  standard fits fail) and Panel 5 (`poster_panel5_corrected_bias.png`, corrected,
  three shapes). Uses `scripts/poster_style.py`.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/dfree-calibration/validate_d_zero.py
.venv/Scripts/python.exe experiments/dfree-calibration/dense_calibration_and_test.py
```

Set `FOLDED_DEBIAS_WORKERS=0` for the dense run. Outputs to repo-root `output/`;
committed snapshots are in this folder.

## Key result + figure

`validate_d_zero` (300-run calibration, 300 evaluation runs per cell) prints
each cell's evaluation SE, the calibration's own propagated SE
(`calibration_D_se`), their combination, and |bias - 1| in combined SE. The
corrected `mean(D_hat)/D_ref` is consistent with 1 within two combined SE in 136
of 140 cells (Gaussian, Voigt, p = 1.5, p = 4; D 0.005 to 0.1; CNR0 5 to 90),
with the other four at 2.0 to 3.4 SE, about the number expected by chance: no
resolved bias. Small D is precision-limited, not biased: combined SE 0.02 to
7.9 at `D <= 0.01`, 0.02 to 1.05 at `D = 0.02`, at most 0.03 only for
`D >= 0.05` at `CNR0 >= 34`.

`calibration_se_check` (`calibration_se_check.txt`): the propagated
calibration SE is correct. Observed / predicted variance of the cell mean over
18 draws is 0.98; per-point SE matches to 0.91 to 1.07; the weight term is
below 0.06 of the prediction from CNR0 5 up. The fft-cnr paper's ratio of 2.0
comes from an unlucky triple of seeds (top 1.0 % of triples). See
`docs/STATUS.md` (2026-10-20).

Dense sweep (5000-run calibration), overall mean `|bias-1|`, folded-normal vs
MAS: Gaussian 0.119 vs 0.438, Voigt 0.175 vs 0.167, p = 2 0.155 vs 0.151,
p = 1.5 0.178 vs 0.815, p = 4 0.179 vs 0.509. Folded-normal is the default: it
wins clearly where the deep-CNR bias is large (Gaussian, p = 1.5) and ties
elsewhere, and MAS fails upward at `CNR0 < ~4`.

Tables: `validate_d_zero_{gaussian,voigt,genGauss_p1.5,genGauss_p4}.txt`,
`dense_calibration_and_test.txt` (Gaussian, Voigt, p = 2),
`dense_calibration_and_test_genGauss_p1.5_p4.txt`. `validate_d_zero_genGauss.txt`
is an older p = 2 draw. The poster's Panel 5 "three shapes" are Gaussian,
Voigt, and p = 2.

## Depends on

`folded_debias`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md`, `docs/poster/POSTER_PLAN.md` Panel 5.
