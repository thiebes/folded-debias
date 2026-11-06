# fisher-calibration

## Purpose

Decide whether `calibrate_fisher` (exact Fisher leading term plus a smoothed
Monte Carlo residual) should replace `calibrate_d_free` (raw per-cell lookup) as
the recommended calibration, and test the two physical claims it rests on: the
leading-order bias equals the width's Fisher variance, and the bias at a fixed
pixel grid and peak CNR scales with the width as the Fisher width table says.

## Status

Run 2026-10-28. `calibrate_fisher` did not pass the switch criterion (it passed
on the Gaussian only), so `calibrate_d_free` stays the recommended path. Scope:
paper. The switch criterion below was written before any full-budget result
existed and is not changed after the run. It was amended once before the run
(see Amendment); the criteria and their bands are unchanged.

## Contents

- `compare_calibrations.py generate|analyse` -- head-to-head at an equal Monte
  Carlo budget (25,200 fits per calibration draw) on validate_d_zero's grid:
  four shapes (Gaussian, Voigt, generalized-Gaussian p = 1.5 and p = 4), D =
  0.005, 0.01, 0.02, 0.05, 0.1 sigma_0^2/tau, CNR_0 5 to 90, 1000 evaluation
  realizations per cell, independent calibration draws: 8 raw, 64 Fisher. Seeds
  have separate roots per shape, method, and draw. Arms: `raw` (calibrate_d_free),
  `fisher` (calibrate_fisher, no width scaling), `fisher_w` (with the width
  table; reported only). Each arm is scored on all frames and on frames with
  per-frame CNR >= 1.
- `fisher_check.py generate|analyse` -- part 1: Fisher bias scale against the
  Monte Carlo spread at CNR >= 8 for the four shapes; part 2: single-frame cells
  at true variance 0.25, 0.5, 1, 2, 4, 9 sigma_0^2 (Gaussian and Voigt) compared
  with three predictions from the sigma_0 calibration (unscaled, leading-order
  width ratio, and the width-scaled lookup).
- `diagnose_criterion3.py` -- after the run: splits the Fisher arm's
  observed/predicted calibration variance into the part from per-draw slope
  weights and the part from the per-draw choice of the smoothing parameter.

## Data

None. Fully synthetic.

## Run

```powershell
$env:WIDTH_DEBIAS_WORKERS = 0
.venv/Scripts/python.exe experiments/fisher-calibration/compare_calibrations.py generate
.venv/Scripts/python.exe experiments/fisher-calibration/fisher_check.py generate
.venv/Scripts/python.exe experiments/fisher-calibration/compare_calibrations.py analyse
.venv/Scripts/python.exe experiments/fisher-calibration/fisher_check.py analyse
.venv/Scripts/python.exe experiments/fisher-calibration/diagnose_criterion3.py
```

The full run took 4 h 20 min on 8 cores (evaluation sets 8 to 11 min per shape).

`generate` caches to repo-root `output/fisher-calibration/` and resumes where it
stopped. `analyse` writes `compare_calibrations.txt` and `fisher_check.txt` there;
the committed snapshots in this folder are the record.

## Switch criterion (pre-registered)

Scored on frames with per-frame CNR >= 1, the regime the method claims.
`calibrate_fisher` becomes the recommended path only if every one of the four
shapes passes all of:

1. **Accuracy.** (a) The pooled corrected bias over the 35 cells (inverse-
   variance weights; calibration error taken from the draw-to-draw scatter, so
   the correlation between cells that share a calibration is included) is
   within 2 SE of zero, or no farther from zero than the raw path's. (b) The
   number of cells whose bias, averaged over that method's draws, lies beyond
   2 SE is at most max(4, the raw path's count). The second clause keeps a more precise
   calibration from being penalised for resolving biases the raw path's larger
   SE hides.
2. **Calibration precision.** The median over cells of the predicted
   calibration SE of D, Fisher over raw, is at most 0.7, and no cell with
   D >= 0.02 sigma_0^2/tau is above 1.1.
3. **Honest calibration SE.** For `fisher`, the observed draw-to-draw variance of
   the cell means over the predicted calibration variance, averaged over cells,
   lies in [0.7, 1.4].
4. **Coverage.** In every cell the `fisher` 95% interval (fit SE and calibration
   SE combined) covers the reference at a rate no more than 0.02 below the raw
   path's, and the median coverage over cells with D <= 0.02 is at least 0.93.
   (At D = 0.1 the raw path's own coverage is 0.89 to 0.94, because the D-free
   weights are set at sigma_0, so an absolute bound there would test the
   weights, not the calibration.)

Width scaling (`fisher_w`) is not part of this decision. Turning it on by
default is a separate decision, after part 2 of `fisher_check.py` and the
high-N Br-I rerun (`experiments/halide-mixing-highn/key_sensitivity.py`).

## Amendment (2026-10-27, before the full run)

A Gaussian-only mid-size check (D 0.02 and 0.1 sigma_0^2/tau, CNR_0 8, 21, 55;
300 evaluation realizations; 8 raw and up to 64 Fisher draws; seed root 0, which
the full run does not use) was run before the full run, to test whether the
Fisher path's predicted calibration SE is honest. It found the following, and
the protocol and the method were changed before any full-run noise was drawn:

1. **Criterion 3 had no power at 8 draws.** One observed/predicted variance
   ratio from 8 draws has 7 degrees of freedom: its 95% range is 0.24 to 2.29
   for an exactly honest SE. The 35 cells share each draw's calibration curve,
   so averaging over them adds little. An honest SE would have fallen outside
   [0.7, 1.4] with probability 0.53 per shape (fully correlated cells). The
   Fisher method now has 64 draws: 63 degrees of freedom, 95% range 0.68 to
   1.38, probability 0.05 per shape of falling outside the band. The raw path keeps 8: it is the comparator, and its SE is
   not under test. The budget per draw is unchanged and equal.
2. **Calibration SE: the smoothed curve's covariance.** The P-spline reported
   its Bayesian covariance, which adds an allowance for smoothing bias that
   draw-to-draw scatter cannot show; over 32 draws it overstated the scatter
   near the knee (CNR 4 to 5) about twofold in variance. It now reports the
   sampling (sandwich) covariance, which criterion 3 measures; smoothing bias is
   tested by criteria 1 and 4.
3. **Calibration SE: the cell errors.** The cell SEs came from a smooth of log
   SD, a geometric mean, which understates the variance of a heavy-tailed cell
   mean near CNR 1 two- to fourfold. They now come from a smooth of the sample
   variance itself. Over all 84 cells and 32 draws the raw cell scatter matched
   s^2/n (mean ratio 0.95 to 1.00 per CNR band; fits within a cell
   uncorrelated, lag-1 autocorrelation below 0.01).
4. **Seeds.** The full run uses seed root 1 (`SEED_ROOT` in
   `compare_calibrations.py`), so none of its data was examined when these
   changes were made.

The criteria and their bands are as first written.

## Key result + figure

No figure; the record is `compare_calibrations.txt`, `fisher_check.txt`, and
`diagnose_criterion3.txt` (seed root 1; 8 raw and 64 Fisher draws of 25,200 fits
each; 1000 evaluation realizations per cell).

**Switch criterion: not passed.** Frames with per-frame CNR >= 1:

| Shape | Pooled bias, Fisher (raw), D/D_ref | Cells beyond 2 SE, Fisher (raw) | Median cal-SE ratio Fisher/raw | Obs/pred cal variance, Fisher | Verdict |
| ------- | ------- | ------- | ------- | ------- | ------- |
| Gaussian | +0.0009 +/- 0.0009 (+0.0008 +/- 0.0013) | 1 (0) | 0.32 | 0.99 | pass |
| Voigt | -0.0003 +/- 0.0011 (-0.0008 +/- 0.0016) | 1 (0) | 0.25 | 1.75 | fail (3) |
| p = 1.5 | +0.0006 +/- 0.0012 (-0.0003 +/- 0.0016) | 6 (4) | 0.30 | 0.89 | fail (1b) |
| p = 4 | +0.0003 +/- 0.0005 (+0.0000 +/- 0.0006) | 0 (1) | 0.20 | 2.71 | fail (3) |

Every shape passes criteria 1a, 2a, 2b, 4a, and 4b. Accuracy is the same as the
raw path's and the predicted calibration SE is 0.20 to 0.32 of it (median over
cells).

- **p = 1.5, criterion 1b.** The cells beyond 2 SE are mostly the same cells
  in both arms, with nearly equal biases, for example (D 0.1, CNR_0 55) at
  -0.008 in both. Their deviations come mostly from the shared evaluation set,
  and the Fisher arm's smaller calibration SE moves two more over 2 SE. Both
  arms exceed the 1.6 cells expected by chance, so p = 1.5 carries evaluation
  noise or a small systematic common to both methods.
- **Voigt and p = 4, criterion 3: the predicted calibration SE is too small.**
  The ratio rises with CNR_0 (p = 4: 1.7 at CNR_0 5 to 5.4 at 90).
  `diagnose_criterion3.txt` locates the missing term. Holding the slope weights
  fixed changes nothing (1.77 and 2.68). Refitting every draw at one smoothing
  parameter brings the ratio to 1.01 (Voigt) and 1.23 (p = 4), flat in CNR_0.
  Generalized cross-validation (GCV) picks the smoothing parameter per draw
  over a wide range (5th to 95th percentile 0.06 to 56 for Voigt, 0.02 to 1000
  for p = 4, against 0.01 to 1 for the Gaussian). The reported covariance
  treats that choice as fixed, so it leaves out the scatter the choice adds.
  The fixed value in the diagnostic is the median over the same 64 draws, so
  the diagnostic shows where the error is, not a validated fix.

Turning this into a passing calibration would need a stable smoothing choice
(for example restricted maximum likelihood, or one value per shape fixed from
a large pooled calibration) or a covariance that includes the
smoothing-parameter uncertainty, and then a new pre-registered run on fresh
seeds. Even with its predicted SE scaled up by the observed ratio, the Fisher
calibration error stays below half of the raw path's.

**Fisher bias scale (`fisher_check.txt`, part 1).** The Fisher value of
A = CNR^2 Var(sigma_hat) matches the Monte Carlo spread at CNR >= 8 pooled over
64 draws: ratio 0.999, 1.003, 0.998, 0.998 (Gaussian, Voigt, p = 1.5, p = 4);
the closed form (2/sqrt(pi)) dx gives 0.05642 sigma_0^2 for the Gaussian.

**Width scaling (`fisher_check.txt`, part 2).** Not established. Over the 13
cells with CNR >= 1:

- At true variance 9 sigma_0^2 no prediction fits the measured bias (chi^2/dof
  2.3 to 6.4, Gaussian and Voigt).
- At 0.25 sigma_0^2 only the width-scaled lookup fits, and only for Voigt
  (1.36); for the Gaussian all three are 4.4 to 5.7.
- Between 0.5 and 4 sigma_0^2 the width-scaled lookup has chi^2/dof 0.78 to
  1.77, against 1.00 to 2.61 unscaled. That is not a clear separation, because
  above CNR 3 the bias is small against its Monte Carlo error.

The ratio of measured bias to the leading term does not collapse onto one curve
in S (at S of 5 to 7 it reads 0.57 to 2.06 across widths for the Gaussian, each
+/- 0.3 to 0.4). Width scaling stays off.

## Depends on

`width_debias` (`calibrate_fisher` components: `calibrate_cells`,
`leading_order_terms`, `BiasLookup.from_cells`; `calibrate_d_free`).
`fisher_check.py` imports paths and seeds from `compare_calibrations.py`.

## See also

`docs/DERIVATION.md` (sections 2, 8, 9); `docs/STATUS.md` 2026-10-27;
`experiments/correction-floor/analytical_floor.py` (the earlier, Monte Carlo
estimate of the bias scale); `experiments/dfree-calibration/` (the raw path's
validation).
