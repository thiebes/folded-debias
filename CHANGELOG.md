# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-03

### Added

- `TruncationModel.frame_variance` (with `framevar_grid` in cached 1-D
  calibrations) gives the per-frame width variance used for the slope weights.
- `folded_debias.fisher` computes the leading-order bias of the fitted variance,
  the width's Fisher-information variance, from the profile model and pixel
  grid alone (`leading_order_terms`, `var_of_fitted_variance` with Cramer-Rao
  or least-squares-sandwich forms). The whole-sequence Fisher matrix and
  `crlb_D` move here from `experiments/crlb-precision`.
- `calibrate_fisher` builds a D-free 1-D calibration from that exact leading
  term and a P-spline smooth of the higher-order residual over single-frame
  Monte Carlo cells (`calibrate_cells`, `CalibrationCells`,
  `TruncationModel.from_cells`). The smoothed lookup carries the sampling
  covariance of the smooth as a factor (`TruncationModel.var_bias_factor`,
  `var_bias_cov`), and `calibration_D_se` uses it. With `width_scaling=True` a width table rescales
  the bias to each frame's width (it grows as sigma at a fixed pixel grid and
  peak CNR). It is not yet the recommended path.
- `msd_and_weights`, `corrected_variance`, and `calibration_D_se` take an
  optional `key_variance`, a measurement-level width key; the weights are
  width-scaled only with such a key.
- `TruncationModel.save` and `TruncationModel.load`.

### Changed

- The weighted schemes weight every frame, t = 0 included, by
  `1 / Var(sigma_hat_i^2)`. The MSD fit is then the same as fitting the
  corrected widths against time and reaches the Cramer-Rao bound on D (0.98 to
  1.08 x sqrt(CRLB) from initial CNR 21 on four profile shapes). A 1-D model
  saved without `framevar_grid` keeps the MSD-point weights with frame 0 at
  weight 0.
- The `cnr` extra requires `fft-cnr>=0.3.3`, the first release that refuses
  degenerate generalized-Gaussian fits below CNR 1.
- `license` uses the SPDX expression form (`"MIT"`) with `license-files`.

## [0.3.0] - 2026-10-21

The code the fft-cnr paper ran (its cases 27 to 30). The first tagged release.

### Added

- Profile models (`GaussianModel`, `VoigtModel`, `GeneralizedGaussianModel`,
  `MomentModel`, `UnboundedGaussianModel`), the Monte Carlo ensemble
  (`run_ensemble`), and the MSD slope fit (`fit_D`).
- Four slope-fit schemes in `msd_and_weights`: `ols`, `inverse_variance`,
  `folded_normal`, and `mas`. `folded_normal` subtracts the calibrated per-frame
  variance bias `var_bias(CNR)` and weights each MSD point by its inverse
  variance, with frame 0 at weight 0.
- `calibrate_d_free`, the D-free 1-D calibration: the standard sweep at D = 0,
  so the lookup is keyed on per-frame CNR alone. `calibrate_cnr_sigma2` builds
  the experimental joint (CNR, sigma^2) lookup.
- `calibration_D_se` (in `weights`) propagates the calibration's own Monte Carlo
  error into the corrected D, through the slope fit's linear map.
  `TruncationModel.var_bias_se` gives the standard error of each looked-up bias,
  `frame_stats` records it, and `SchemeResult.cal_se` reports it per cell. The
  propagated value matches the scatter across 16 independent calibrations
  (observed/predicted 0.98).
- `model_from_spec` and `spec_tag` build a profile model from a string such as
  `"generalized_gaussian:1.5"`, so scripts and worker processes can name the
  generalized-Gaussian shape exponent.
- Packaging: the version is defined once, in `folded_debias.__version__`;
  published dependencies carry compatible upper bounds, and `uv.lock` records
  the environment that produced the committed results. MIT license.

### Documentation

- `TruncationModel.var_bias` and `calibrate_cnr_sigma2` document the choice of
  key for the joint (CNR, sigma^2) lookup: keying each frame on its own fitted
  variance under-corrects when the width grows several-fold over the
  measurement.
