# `scripts/` — what each script is for

All scripts run from the repository root, with `.venv/Scripts/python.exe`
on Windows. Outputs go to `output/` with matching base names.

## Canonical scripts

These are the scripts the poster and methods paper rely on. Keep them
working; re-run when calibration or validation logic changes.

### Bias mechanism and the standard pipeline's behaviour

- **`poc_demo.py [profile_model]`** — Two-panel proof-of-concept figure
  for a single profile model (default `gaussian`). Panel (a): the
  truncation diagnostic (fraction below zero, sample skewness) vs CNR.
  Panel (b): bias curves `mean(D̂)/D_true` vs CNR₀ for OLS,
  inverse-variance, and the derived folded-normal scheme. *Source for
  poster Panels 2 and 5.*
  Outputs: `output/poc_demo[_<model>].png`, `output/poc_demo[_<model>].txt`.

- **`diagnose_boundary.py [profile_model]`** —
  Clipping-vs-reflection diagnostic. Established that the fit
  **reflects** rather than clips at the wall at zero, so the
  **folded-normal** route applies (not a Tobit / clipped model). One run
  per profile model. *Historical / framing reference; cited in
  DERIVATION.md §2.*
  Outputs: `output/boundary_diagnostic[_<model>].png`.

### Calibration without knowledge of D, and the headline validation

- **`validate_d_zero.py [profile_model]`** — D-independence validation
  of `calibrate_d_free()`: one calibration built at `D = 0`, applied
  across `D ∈ {0.005, 0.01, 0.02, 0.05, 0.1} σ₀²/τ`, on Gaussian, Voigt,
  or generalized-Gaussian profile shapes. *The headline calibration
  claim's evidence.*
  Outputs: `output/validate_d_zero_<model>.txt`.

- **`dense_calibration_and_test.py`** — Dense parallel
  `D`-free calibration (100 log-spaced CNR₀, 1000 runs/cell) followed
  by a random log-distributed test (100 CNR₀ values, 1000 runs/cell),
  for all four schemes (`ols`, `inverse_variance`, `folded_normal`,
  `mas`) on all three profile shapes. *Source for the MAS-vs-folded
  recommendation reversal and for poster Panel 5 corrected bias curves.*
  Wall time ≈ 3 hours on a typical machine.
  Output: `output/dense_calibration_and_test.txt`.

### Where the correction works

- **`per_profile_floor.py [profile_model]`** — Isolated single-profile
  sweep that places the correction's practical floor. For each test CNR,
  simulates 50,000 noisy profiles at `σ² = σ₀²`, applies the
  calibration's `var_bias(CNR)` subtraction, reports the residual
  `mean(σ̃²) − σ²_true` with MC standard error. *The cleanest test of
  where the correction works; source for poster Figure F.*
  Wall time ≈ 45 min (Gaussian) to ≈ 3 h (Voigt).
  Outputs: `output/per_profile_floor_<model>.txt`.

  **Note on the script's auto-floor reporting**: the "Per-profile floor
  (deepest CNR where |z| ≤ 2)" line at the bottom uses a strict |z|
  criterion that triggers spuriously at very high CNR (where residuals
  are tiny but SE is even tinier). Read the table directly: the
  practical floor is where `|residual| / σ²_true` first exceeds, say,
  0.05.

- **`analytical_floor.py`** — Power-law fit of
  `var_bias(CNR) ≈ A / CNR²` σ₀² to the calibration at CNR > 8, then
  comparison to the empirical curve at low CNR. Identifies where the
  power-law breaks (≈ 0.1 across all three shapes; the σ̂² distribution
  saturates against the fit's `var_max`). *The analytical handle on
  model-breakdown CNR.*
  Output: `output/analytical_floor.txt`.

- **`per_frame_in_diffusion_experiment.py [profile_model]`** — Runs the
  per-frame folded-normal and MAS corrections *inside* a full 12-frame
  diffusion experiment (using a dense `D`-free calibration), reports
  per-frame residual bias at CNR₀ ∈ {5, 8, 13, 30}. Cross-validates the
  per-profile-floor picture in the application setting.
  Wall time ≈ 45 min (Gaussian) to ≈ 80 min (Voigt) per shape.
  Output: `output/per_frame_in_diffusion_experiment.txt` (overwritten
  per run; one shape per invocation).

### Comparators

- **`moment_estimator_failure.py`** — Five-way method comparison
  (Gaussian only): four parametric-fit schemes plus the non-parametric
  `MomentModel` (discrete-variance / second-moment estimator).
  Demonstrates that the `MomentModel` is unstable in the low-CNR / low-D
  regime — its profile-ratio form passes through zero, producing wild
  and often negative `D`. *Negative result, kept as the comparator for
  the paper's "the moment estimator is not a viable substitute" leg.*
  Output: `output/method_comparison.png`.

## Legacy / deprecated

`scripts/legacy/` holds scripts that documented development decisions or
explored avenues that did not pan out. They run, but are not part of
the current canonical pipeline:

- **`legacy/validate_d_free.py`** — Validation of the *experimental 2-D*
  `var_bias(CNR, σ²)` calibration (`weighting_wall.calibration.calibrate_2d`).
  The 2-D calibration was not competitive at deep CNR; we use the 1-D
  `D=0` calibration (`calibrate_d_free`) instead. Kept as documentation
  of the failure.
- **`legacy/diagnose_calibration.py`** — One-off comparison of 1-D
  D-aware vs 2-D D-free calibration values at matched (CNR, σ²) points.
  Output of the same investigation.
- **`legacy/explore_regime.py`** — Early exploratory D × CNR₀ sweep
  used to find the bias-prone regime. The current canonical regime is
  locked at `D = 0.02 σ₀²/τ` (`L_D / FWHM ≈ 0.085`); this script's job
  is complete.

The corresponding output files live in `output/legacy/`.
