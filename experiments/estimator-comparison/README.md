# estimator-comparison

## Purpose

Answer "is the correction even needed, or does a different estimator avoid the
bias?" Compares the folded-normal correction against the non-parametric moment
estimator and an unbounded (positivity-wall-removed) width fit.

## Status

Validated. Poster aside / paper-scope (the "removing the positivity constraint
does not substitute for the correction" leg).

## Contents

- `moment_estimator_failure.py` -- five-way comparison; the `MomentModel`
  (second-moment estimator) is unstable at low CNR / low D (wild, often negative
  D).
- `unbounded_fit_comparison.py` -- folded-normal vs an unbounded width fit (the
  Gaussian fitted in precision `b = 1/sigma^2`, so the fitted variance can take
  any sign), with bounded+OLS and the moment estimator as references.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/estimator-comparison/moment_estimator_failure.py
.venv/Scripts/python.exe experiments/estimator-comparison/unbounded_fit_comparison.py
```

Set `FOLDED_DEBIAS_WORKERS=0`. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

Above per-frame CNR ~1.2 the bounded and unbounded fits are bit-for-bit identical
(the wall never binds); below it, removing the wall only partly debiases and
explodes the variance, because a nonlinear shape-matched fit intrinsically
resists negative variances. Only folded-normal is both unbiased and precise
across CNR0. Figures: `method_comparison.png`, `unbounded_fit_comparison.png`;
table `unbounded_fit_comparison.txt`.

## Depends on

`folded_debias` (`MomentModel`, `UnboundedGaussianModel`). No sibling-folder
helpers.

## See also

`docs/DERIVATION.md` section 7.
