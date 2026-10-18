# coverage-interval

## Purpose

Does the reported confidence interval on D cover D_true at the nominal rate, not
just the mean? Compares the covariance-based CI, a parametric bootstrap, and an
analytical attribution for the de-biased folded-normal estimator.

## Status

Validated. Paper-scope (not a poster panel).

## Contents

- `coverage_check.py` -- sweeps CNR0 and reports empirical 95% coverage for
  `folded_normal` (and the biased `inverse_variance` as the failing contrast),
  plus the `corr_inflation = sqrt(Var_full/Var_diag)` attribution.

## Data

None. Fully synthetic. Reuses the cached `calibration_subdiffusive.npz`.

## Run

```bash
.venv/Scripts/python.exe experiments/coverage-interval/coverage_check.py
```

Set `FOLDED_DEBIAS_WORKERS=0`; ~5 min. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

The shared sigma_hat^2_0 is common-mode (shifts the intercept, not the slope), so
it is not the cause of under-coverage (inflation 0.91-0.93 < 1). The covariance
CI under-covers erratically (0.74-0.79) because the D-free calibration
mis-estimates the absolute per-point variance; the bootstrap covers ~0.94.
Recommendation: report a bootstrap/profile interval, not the covariance CI.
Figure: `coverage_check.png`; table `coverage_check.txt`.

## Depends on

`folded_debias` (`fit.fit_D` returns `D_se`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (the folded-normal correction).
