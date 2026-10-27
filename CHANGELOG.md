# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `calibration_D_se` (in `weights`) propagates the calibration's own Monte Carlo
  error into the corrected D, through the slope fit's linear map.
  `TruncationModel.var_bias_se` gives the standard error of each looked-up bias,
  `frame_stats` records it, and `SchemeResult.cal_se` reports it per cell. The
  propagated value matches the scatter across 16 independent calibrations
  (observed/predicted 0.98).
- `model_from_spec` and `spec_tag` build a profile model from a string such as
  `"generalized_gaussian:1.5"`, so scripts and worker processes can name the
  generalized-Gaussian shape exponent.
- `TruncationModel.frame_variance` (with `framevar_grid` in cached 1-D
  calibrations) gives the per-frame width variance used for the slope weights.

### Changed

- The weighted schemes weight every frame, t = 0 included, by
  `1 / Var(sigma_hat_i^2)`. The MSD fit is then the same as fitting the
  corrected widths against time and reaches the Cramer-Rao bound on D (0.98 to
  1.08 x sqrt(CRLB) from initial CNR 21 on four profile shapes). A 1-D model
  saved without `framevar_grid` keeps the MSD-point weights with frame 0 at
  weight 0.
- The `cnr` extra requires `fft-cnr>=0.3.3`, the first release that refuses
  degenerate generalized-Gaussian fits below CNR 1.
- Published dependencies carry compatible upper bounds; `uv.lock` records the
  environment that produced the committed results.
- The version is defined once, in `folded_debias.__version__`.
- `license` uses the SPDX expression form (`"MIT"`) with `license-files`.

### Documentation

- `TruncationModel.var_bias` and `calibrate_cnr_sigma2` document the choice of
  key for the joint (CNR, sigma^2) lookup: keying each frame on its own fitted
  variance under-corrects when the width grows several-fold over the
  measurement.
