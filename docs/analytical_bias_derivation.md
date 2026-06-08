# Closed-form bias of the four width estimators

Goal: an explicit closed form for the bias in the recovered profile width
`b = E[sigma_hat^2] - sigma_true^2` (and the width bias) as a function of all
experimental parameters, for each of four estimators -- Gaussian fit, Voigt
fit, generalized-Gaussian fit, and the second-moment estimator -- validated
against the repository Monte Carlo. This replaces the project's numerical
`var_bias(CNR)` calibration with an analytic expression.

Stop criterion (settled with the operator, iter 3): the **variance form** is
the binding test -- `|mean(var_corrected) - var_true| <= 0.02 var_true` at
per-frame CNR = 1.3, across the covered parameter ranges, for every estimator.
The variance is the quantity that de-biases D (the MSD slope), so it is the
physically meaningful target. The width is reported as a derived diagnostic,
quoted as `sqrt(mean(var_corrected))` (the width of the corrected mean
variance, which is what sets the reported diffusion length).

**Why the per-frame width form was dropped.** At CNR = 1.3 the estimator spread
is large, so `mean(sqrt(v_hat)) != sqrt(mean(v_hat))` (Jensen). The raw
per-frame mean width is only +0.18% biased while the raw variance is +5.24%
biased -- a 29x ratio, not the 2x that linearisation assumes. A correction that
makes the variance unbiased necessarily makes the *per-frame-averaged* width
biased low (-2.36% at CNR 1.3), because `sqrt` of an unbiased variance
estimator is a downward-biased width estimator. The two per-frame forms are
mutually unsatisfiable at CNR 1.3; the variance form is the one that matters for
D, and `sqrt(mean(v_corrected))` recovers the width to -0.19% there.

## Progress log (newest first)

- **Iter 3 (2026-06-08).** Ran the stop-condition test (multiplicative
  correction on isolated-frame MC). Surfaced and resolved the width-vs-variance
  ambiguity: at CNR 1.3, Jensen makes the per-frame width and variance criteria
  mutually unsatisfiable. Operator chose the **variance form** as binding
  (width is a derived diagnostic, quoted as `sqrt(mean v_corr)`). Under it the
  Gaussian **two-term form passes at base geometry**: variance residual -0.38%
  (criterion 2%), derived-width residual -0.19% (Box-only is +1.47%, inside 2%
  but near the edge). Built the unified numeric Box predictor
  (`box_bias_numeric`), validated to reproduce the analytic Gaussian to 4
  places, and computed Voigt / GenGauss coefficients (Voigt grows with gamma;
  GenGauss p=1.5 flips sign). Next: pin `C4` geometry scaling via the geometry
  sweep so the two-term form holds across the grid (not just base geometry),
  validate the variance criterion across dx/W/sigma at CNR 1.3, then MC-validate
  Voigt/GenGauss and the moment estimator.
- **Iter 2 (2026-06-08).** MC-validated the Gaussian Box coefficient. At CNR
  8..12 the clean isolated-frame mean bias matches `b_Box` to within MC error
  (`mean/Box = 0.98-1.02`), confirming the `2/sqrt(pi)` coefficient and
  settling the discrepancy: `analytical_floor.py`'s `0.37` was `var_max` tail
  contamination of the mean in the full diffusion sweep, not the true bias. The
  median bias is `~= -b_Box` (right-skew), so the mean is the correct target.
  Below CNR ~ 3 a positive excess appears (`mean/Box` = 1.10 at CNR 3, 1.57 at
  CNR 1.3); `(mean - b_Box) * CNR^4 ~= 0.055 sigma0^2` is roughly constant, a
  clean **second term**: `b = b2/CNR^2 + C4/CNR^4`, `C4 ~= 0.055 sigma0^2` at
  base geometry. At CNR 1.3 the Box-only variance residual is 0.019 (1.9% of
  sigma^2); the two-term form removes it. Next: (i) confirm the stop criterion
  on the corrected WIDTH (not just variance), (ii) pin `C4`'s geometry scaling,
  (iii) run the moment-estimator and geometry sweeps.
- **Iter 1 (2026-06-08).** Set up branch `analytical-bias-derivation`,
  deliverable skeleton, parameter grid. Derived the high-CNR additive bias for
  the Gaussian fit from Box's (1971) nonlinear-least-squares second-order bias
  formula. Known-amplitude, known-centre continuum result:
  `b = (2/sqrt(pi)) * sigma_n^2 * dx * sigma / A^2 = (2/sqrt(pi)) (dx sigma) / CNR^2`.
  Multi-parameter version (free A, free centre) in progress; MC validation of
  the leading coefficient launched. Next: confirm the coefficient, then add the
  free-A/free-centre nuisance inflation.

## Notation and conventions

- `x` pixel grid, spacing `dx`, count `N`, half-window `W` (so the axis spans
  `[-W, W]`, `W = (N-1) dx / 2`). Lengths in units of `sigma0` unless stated.
- `sigma`, `v = sigma^2`: true profile width and variance at the frame.
- `A`: noise-free peak amplitude. `sigma_n`: white pixel-noise RMS.
- Per-frame `CNR = A / sigma_n` (peak over noise RMS), matching
  `simulate.noise_sigma_for_cnr0` / `montecarlo.run_ensemble`'s `cnr_frame`.
- `sigma_hat^2`: the estimator's reported variance for one noisy frame.
- `b = E[sigma_hat^2] - v`: the additive variance bias (the target). The width
  bias is `E[sigma_hat] - sigma`; near the truth `b ~= 2 sigma (E[sigma_hat] -
  sigma) + Var(sigma_hat)`, so a variance criterion of 2% maps to a width
  criterion of ~1%.

## The numerical oracle

For a chosen `(CNR, dx, N, W, gamma, p, sigma)` we generate noisy profiles,
apply the estimator, and measure the empirical `E[sigma_hat^2] - v` with its MC
standard error. Several thousand realizations per cell for the bounded fits;
more for the moment estimator near its instability. The MEAN is the target
statistic; the median is reported only as a diagnostic of
tail contamination.

`scripts/analytical_bias_validate.py` computes the closed form and the MC bias
across the grid and reports the post-correction residual.

## Parameter grid for validation

The validation sweeps each axis around the poster-relevant base point while
holding the others fixed, then a joint check at the stop point CNR = 1.3.

| Axis | Base | Sweep | Holds |
| ------- | ------------- | ------------- | ------------- |
| CNR | 8 | {1.3, 2, 3, 5, 8, 12, 20, 30, 50} | dx=0.05, N=241, W=6, sigma=1 |
| dx (via N) | 0.05 | N in {121, 161, 241, 321, 481} | W=6, sigma=1, CNR=8 |
| W (via half_window) | 6 | {3, 4, 6, 8, 10} | dx=0.05, sigma=1, CNR=8 |
| sigma | 1 | {0.5, 0.7, 1.0, 1.4, 2.0} | dx=0.05, N=241, W=6, CNR=8 |
| gamma (Voigt) | 0.3 | {0.1, 0.2, 0.3, 0.5} | base geometry, CNR=8 |
| p (GenGauss) | 2 | {1.5, 1.8, 2.0, 2.5, 3.0} | base geometry, CNR=8 |

`sigma` and `W` are coupled through the dimensionless window `W/sigma`; the
relevant geometric ratios are `sigma/dx` (pixels per width) and `W/sigma`
(window in widths). The closed form is expressed in those ratios where possible.

## 1. Gaussian fit -- high-CNR additive bias (Box second-order NLS bias)

The least-squares Gaussian fit with Gaussian pixel noise is the maximum-
likelihood estimator. Its leading bias (order `sigma_n^2`, i.e. `1/CNR^2`) is
given by Box's (1971) second-order bias formula for nonlinear least squares.
For a scalar parameter `theta` estimated from `y_i = f_i(theta) + eps_i`,
`eps ~ N(0, sigma_n^2)`,

    b(theta_hat) = -(sigma_n^2 / 2) * [ sum_i f_i' f_i'' ] / [ sum_i f_i'^2 ]^2,

with `f_i' = df_i/dtheta`, `f_i'' = d^2 f_i/dtheta^2`. (The folded-normal "fold"
is a separate, lower-CNR effect; at high CNR the estimate sits many `s_est`
above the wall and the fold is exponentially small. The polynomial `1/CNR^2`
bias the calibration sees at high CNR is this Box term, not the fold.)

### 1.1 Known amplitude, known centre

Parameter `theta = v` (variance), `f(x) = A exp(-x^2 / 2v)`. Then

    f' = f * x^2 / (2 v^2),
    f'' = f * [ (x^2/2v^2)^2 - x^2/v^3 ].

Pass to the continuum (`sum_i -> (1/dx) integral dx`) and use
`integral exp(-x^2/v) x^(2k) dx = sqrt(pi v) (2k-1)!! (v/2)^k`:

    sum f'^2     =  3 sqrt(pi) A^2 / (16 dx v^(3/2)),
    sum f' f''   = -9 sqrt(pi) A^2 / (64 dx v^(5/2)).

Substituting,

    b = (2 / sqrt(pi)) * sigma_n^2 dx sqrt(v) / A^2
      = (2 / sqrt(pi)) * (dx sigma) / CNR^2.          (Gaussian, known A & centre)

So the leading bias is additive, positive, scales as `1/CNR^2`, grows linearly
in the pixel size `dx` and in the width `sigma`, and is independent of the
window `W` and pixel count `N` once the profile is well inside the window (the
Gaussian tails that fall outside are exponentially small).

Note `sum f'^2 = I_v sigma_n^2` is the Fisher information, so the known-A
variance CRLB is `CRLB_v = sigma_n^2 / sum f'^2 = (16/3) dx v^(3/2) sigma_n^2 /
(sqrt(pi) A^2)`. The bias is **not** simply `CRLB_v/(4 sigma^2)` (that heuristic
gives `4/3` instead of `2` for the coefficient): Box's formula correctly
combines the Jensen variance term and the width-MLE second-order bias.

### 1.2 Free amplitude and centre

The fit estimates `(v, A, c)` jointly. Box's multi-parameter formula is

    b(v_hat) = -(sigma_n^2/2) sum_b (F^TF)^-1_{v,b} sum_i F_{i,b} d_i,
    d_i = sum_{c,d} (F^TF)^-1_{cd} (d^2 f_i / dtheta_c dtheta_d),

with theta = (v, A, c). The Jacobian columns are even (v, A) and odd (c) in x,
so `F^TF` is block-diagonal: a 2x2 (v, A) block and a scalar c block. Working
the continuum integrals (all reduce to Gaussian moments
`integral exp(-x^2/v) x^(2k) dx = sqrt(pi v)(2k-1)!!(v/2)^k`):

    (F^TF)^-1_{vv} = 8 dx v^(3/2) / (sqrt(pi) A^2)   [= CRLB_v/sigma_n^2, matches crlb.py]
    (F^TF)^-1_{vA} = -2 dx v^(1/2) / (sqrt(pi) A)
    (F^TF)^-1_{cc} =  2 dx v^(1/2) / (sqrt(pi) A^2)

The d-vector collapses neatly -- the `vA` and `cc` contributions cancel the
`x^2/v^2` pieces -- to

    d(x) = (2 kappa / v) [ x^4/v^2 - 4 x^2/v - 1 ],   kappa = dx sqrt(v)/(sqrt(pi) A^2).

Carrying the two projections through (the `v` row gives `-13/8`, the `A` row
gives `-9/2`, weighted by `(F^TF)^-1_{vv}` and `(F^TF)^-1_{vA}` they combine as
`-13 + 9 = -4`), the result is

    b = (2 / sqrt(pi)) sigma_n^2 dx sqrt(v) / A^2 = (2/sqrt(pi)) (dx sigma)/CNR^2.

**The free-(A, c) nuisances do not change the leading bias coefficient** -- it
is identical to the known-amplitude `2/sqrt(pi)` of Sec. 1.1, even though
fitting A inflates `Var(v_hat)` by 3/2. This is the high-CNR closed form for the
Gaussian fit:

    b_Gauss(CNR, dx, sigma) = (2/sqrt(pi)) (dx sigma) / CNR^2,   coefficient 1.128.

Independent of window `W` and pixel count `N` once the profile is interior
(`W/sigma >~ 4`); the dependence enters only through `dx = 2W/(N-1)`.

**MC validation (iter 2).** Isolated-frame mean bias (20000 fits/cell, base
geometry) against `b_Box`:

| CNR | b_MC_mean (sigma0^2) | b_Box | mean/Box | med/Box |
| ----- | ------------- | ------------- | ------- | ------- |
| 12 | 3.84e-4 | 3.92e-4 | 0.98 | -1.08 |
| 8 | 8.97e-4 | 8.82e-4 | 1.02 | -0.92 |
| 5 | 2.38e-3 | 2.26e-3 | 1.05 | -1.06 |
| 3 | 6.92e-3 | 6.27e-3 | 1.10 | -0.96 |
| 2 | 1.67e-2 | 1.41e-2 | 1.19 | -0.98 |
| 1.3 | 5.24e-2 | 3.34e-2 | 1.57 | -1.03 |

The coefficient is confirmed; `analytical_floor.py`'s `0.37` (mean over deep-CNR
frames of a full diffusion sweep) was `var_max`-tail-inflated. The median bias
is `~= -b_Box`: the estimator distribution is right-skewed, so the mean lies a
Box-bias above truth and the median a Box-bias below -- which is why the mean is
the de-biasing target and the median must not be substituted for it.

### 1.3 Low-CNR second term

Below CNR ~ 3 the mean exceeds `b_Box`. The excess `b_MC - b_Box` times `CNR^4`
is roughly constant (0.04-0.075 sigma0^2 across CNR 1.3..8), i.e. a clean
next-order term:

    b_Gauss(CNR) = (2/sqrt(pi)) (dx sigma)/CNR^2 + C4/CNR^4,   C4 ~= 0.055 sigma0^2.

This is the third-order NLS-bias correction (non-Gaussianity of the width MLE as
CNR falls), *not* a wall effect: at CNR 1.3 the lower wall (sigma_min ~ 1 pixel)
sits ~5 width-noise sigma below the mean and the `var_max` ceiling is far above,
so neither fold nor saturation is active yet -- the excess is the asymptotic
series continuing. `C4`'s geometry scaling (does it track `(dx sigma)`, or
`(dx sigma)^2`, or carry a `W` dependence?) is pinned by the geometry sweep
next. With both terms the CNR 1.3 variance residual falls from 1.9% to MC noise.

## 2. Gaussian fit -- low-CNR corrections (pending)

The fold (folded-normal mean shift) and the `var_max` saturation ceiling
(Sec. 9-10 of `DERIVATION.md`) enter below CNR ~ a few. To reach the CNR = 1.3
stop point these must be added to the Box term. Pending.

## 3. Voigt and generalized-Gaussian fits -- unified numeric Box predictor

The Box machinery is not Gaussian-specific. Parametrise any profile by the
diffusive variance `v` (the reported quantity) plus amplitude and centre; by
MLE reparameterisation equivariance the Box bias for `v` is the variance bias,
whatever the profile shape. `box_bias_numeric` finite-differences the profile
w.r.t. `(v, A, c)` on the pixel grid and evaluates the full multi-parameter Box
formula (no block-diagonal assumption). It reproduces the analytic Gaussian
coefficient to 4 places (ratio 1.0000), so it is trustworthy for shapes whose
integrals are not closed form.

High-CNR coefficient `b * CNR^2` (sigma0^2 units, base geometry dx = 0.05,
sigma = 1, CNR = 8), from the numeric predictor (MC validation pending):

| Shape | parameter | b * CNR^2 | vs Gaussian 0.0564 |
| ------- | ------------- | ------------- | ------------- |
| Gaussian | -- | 0.0564 | 1.00 |
| Voigt | gamma = 0.1 | 0.0619 | 1.10 |
| Voigt | gamma = 0.3 | 0.0740 | 1.31 |
| Voigt | gamma = 0.5 | 0.0875 | 1.55 |
| GenGauss | p = 1.5 (heavy tail) | -0.189 | -3.34 |
| GenGauss | p = 2.0 | 0.0564 | 1.00 |
| GenGauss | p = 3.0 (flat top) | 0.0287 | 0.51 |

Readings: (i) the Voigt coefficient grows with the fixed Lorentzian width
`gamma` -- a broader instrument PSF dilutes the Gaussian-width information, so
the same noise produces a larger width bias; (ii) GenGauss p = 2 recovers the
Gaussian exactly (consistency check); flat-topped profiles (p > 2) bias less;
**heavy-tailed profiles (p < 2) flip the sign** -- the curvature term
`sum f' f''` changes sign, so the leading bias is negative. The p = 1.5 sign
flip is a concrete prediction to confirm against MC.

These coefficients (0.06-0.09 for Voigt) are again ~5-6x below
`analytical_floor.py`'s `0.42` (Voigt) / `0.37` (GenGauss), consistent with the
same `var_max`-tail inflation of the mean identified for the Gaussian.

## 5. Moment estimator (pending)

Different mechanism: not bounded, no fold. Bias from (a) the additive
noise-floor term in the second moment (white noise of variance `sigma_n^2`
weighted by `(x - xbar)^2`, scaling with the window extent and inversely with
total signal) and (b) the total-signal-through-zero instability that corrupts
the sample mean at low CNR. To be derived as a ratio-of-bilinear-forms mean.
