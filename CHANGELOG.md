# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-06

The first release.

### Added

- `BiasLookup.frame_variance` (with `framevar_grid` in cached 1-D
  calibrations) gives the per-frame width variance used for the slope weights.
- `width_debias.fisher` computes the leading-order bias of the fitted variance,
  the width's Fisher-information variance, from the profile model and pixel
  grid alone (`leading_order_terms`, `var_of_fitted_variance` with Cramer-Rao
  or least-squares-sandwich forms). The whole-sequence Fisher matrix and
  `crlb_D` move here from `experiments/crlb-precision`.
- `calibrate_fisher` builds a D-free 1-D calibration from that exact leading
  term and a P-spline smooth of the higher-order residual over single-frame
  Monte Carlo cells (`calibrate_cells`, `CalibrationCells`,
  `BiasLookup.from_cells`). The smoothed lookup carries the sampling
  covariance of the smooth as a factor (`BiasLookup.var_bias_factor`,
  `var_bias_cov`), and `calibration_D_se` uses it. With `width_scaling=True` a width table rescales
  the bias to each frame's width (it grows as sigma at a fixed pixel grid and
  peak CNR). It is not yet the recommended path.
- `msd_and_weights`, `corrected_variance`, and `calibration_D_se` take an
  optional `key_variance`, a measurement-level width key. A 1-D model with a
  width table scales its weights only with such a key; a joint (CNR, sigma^2)
  model keys its weights on each frame's own fitted variance.
- `BiasLookup.save` and `BiasLookup.load`.

### Changed

- The package is named for what it corrects, the noise bias in each fitted
  profile width, from which the MSD and D corrections follow: the distribution
  `folded-debias` is `width-debias` (import `width_debias`), the
  `folded_normal` scheme is `subtract`, the `truncation` module and
  `TruncationModel` are `bias_lookup` and `BiasLookup`, and
  `FOLDED_DEBIAS_WORKERS` is `WIDTH_DEBIAS_WORKERS`. In the regime where the
  correction holds, the fit's lower bound does not bind, and the bias is the
  width's estimation variance; the folded-normal model is one derivation of it.
- The weighted schemes weight every frame, t = 0 included, by
  `1 / Var(sigma_hat_i^2)`. The MSD fit is then the same as fitting the
  corrected widths against time and reaches the Cramer-Rao bound on D (0.98 to
  1.08 x sqrt(CRLB) from initial CNR 21 on four profile shapes). A 1-D model
  saved without `framevar_grid` keeps the MSD-point weights with frame 0 at
  weight 0.
- The `cnr` extra requires `fft-cnr>=0.3.3`, the first release that refuses
  degenerate generalized-Gaussian fits below CNR 1.
- `license` uses the SPDX expression form (`"MIT"`) with `license-files`.

### Documentation

- The README gives the installation from PyPI, a worked example on a
  measurement, and the conditions for real data: a calibration that matches the
  data's geometry and units, the per-frame CNR floor, and the narrower range
  over which an estimated CNR holds.
- `docs/DERIVATION.md` states the leading-order bias as growing with sigma at a
  fixed pixel grid and peak CNR, matching `width_debias.fisher`.

## [0.3.0] - 2026-10-21

Not tagged or published; 0.4.0 is the first release. The code at commit
2836e6c, on which the fft-cnr paper's cases 27 to 30 were first run.

### Added

- Profile models (`GaussianModel`, `VoigtModel`, `GeneralizedGaussianModel`,
  `MomentModel`, `UnboundedGaussianModel`), the Monte Carlo ensemble
  (`run_ensemble`), and the MSD slope fit (`fit_D`, with the covariance
  standard error `DFit.D_se`).
- Simulation beds: `Experiment`, `AnomalousExperiment` (subdiffusive spreading),
  and `MultiExpExperiment` (multi-exponential decay), with the noise laws in
  `folded_debias.noise` (`WhiteNoise`, `ShotNoise`).
- `evaluate` (bias curves per scheme over a CNR sweep), `per_frame_diagnostic`,
  and `cnr.per_frame_cnr` (per-frame CNR from measured profiles through
  `fft-cnr`).
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
