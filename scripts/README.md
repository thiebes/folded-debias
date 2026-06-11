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

### Real-data calibration and tooling

- **`build_calibration.py --tag <name> ...`** — Builds a geometry-specific
  joint `var_bias(CNR, σ²)` calibration for an arbitrary experiment (real or
  simulated): pass the pixel grid, σ range, CNR range, profile model, and MC
  budget. Saves `output/calibration_<tag>.npz`, loadable with
  `CalibrationGrid.load()`. *Used to build the Marc-geometry calibrations
  (`marc_{gaussian,voigt,gengauss}_filtered`) the realistic-transport
  validations consume.* Wall time ≈ 2.5–4 h for a 100×100-cell, 1000-runs/cell
  build. See the module docstring for example invocations.

- **`plot_calibration.py <calibration.npz>`** — Diagnostic heatmaps of a saved
  joint calibration: the `var_bias` and `rel_variance` surfaces over the
  (CNR, σ²) grid, with the Marc-data trajectory overlaid when applicable.
  Output: `output/<tag>_diagnostic.png`.

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

### Realistic-transport validation (Marc Meléndez's models)

These run the correction on Marc's reaction-diffusion diffusion maps — genuinely
non-Fickian transport with the model author's own `σ²(t)` as ground truth. They
need the Marc-geometry calibrations built by `build_calibration.py` and the
`.dat` data in `mc_experiments/marc_mn_trapping/data/` (gitignored; from Marc).

- **`marc_mn_trapping_plan_a.py`** — Per-frame variance recovery on the 0.51% Mn
  diffmap (Magdaleno et al. 2025). Test A: Gaussian fit on the as-is diffmap.
  Test B: Voigt fit on the PSF-convolved diffmap (gated on
  `calibration_marc_voigt_filtered.npz`). Compares corrected vs noiseless fit.
  Outputs: `output/marc_mn_trapping_plan_a.{png,txt}`.

- **`marc_diffmap_validation.py`** — The correction across **three estimators**
  (Gaussian, Voigt, generalized-Gaussian) on all four doping cases, each against
  its own noiseless-fit reference, plus the `MomentModel` as the non-truncated
  comparator. Separates the **correction residual** (vs noiseless fit) from the
  **shape gap** (noiseless fit vs true second moment). *Source for the poster
  Panel 6 realistic-transport result.*
  Outputs: `output/marc_diffmap_validation.{png,txt}`.

### Poster figures

- **`poster_figures.py`** — Regenerates the data-driven poster figures and the
  rough panel schematics in one place: Panel 1 (measurement schematic), Panel 2A
  (wall-and-fold schematic), Panel 4 (√CRLB/D precision floor), Panel 5
  (corrected bias, three shapes), and the Panel 5 floor inset. Okabe-Ito
  (Spain CVD) palette, ticks inside the frame. The fft-cnr panel (Panel 4 of the
  CNR-centric spine) is pulled separately from the sibling `fft-cnr-paper` repo.
  Outputs: `output/poster_panel*.png`.
  Requires the sibling `curved-text` package (the Panel 2A curve labels):
  `.venv/Scripts/python.exe -m pip install -e ../curved-text`.

### Parked / analytical exploration

- **`linearity_test.py`** — Tests the dimensional/CRLB scaling claim that the
  folded-normal bias `bias/σ²` is a function of `s = CNR·√(σ/dx)` alone, not CNR
  alone. The median bias tracks the analytical `λ(−s)/s` form in an intermediate
  band, but the mean is corrupted by broken-fit saturation outliers. *Parked:
  the closed-form derivation is paper-scope, not poster-blocking; result written
  up in DERIVATION.md §10.*
  Outputs: `output/linearity_test.{png,txt}`.

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

- **`legacy/validate_d_free.py`** — Validation of the *experimental joint*
  `var_bias(CNR, σ²)` calibration
  (`weighting_wall.calibration.calibrate_cnr_sigma2`). The joint calibration
  was not competitive at deep CNR; we use the single-σ_0 `D=0` calibration
  (`calibrate_d_free`) instead. Kept as documentation of the failure.
- **`legacy/diagnose_calibration.py`** — One-off comparison of 1-D
  D-aware vs 2-D D-free calibration values at matched (CNR, σ²) points.
  Output of the same investigation.
- **`legacy/explore_regime.py`** — Early exploratory D × CNR₀ sweep
  used to find the bias-prone regime. The current canonical regime is
  locked at `D = 0.02 σ₀²/τ` (`L_D / FWHM ≈ 0.085`); this script's job
  is complete.
- **`legacy/dense_calibration_and_test_robust.py`** — Robust multiplicative
  scheme (median-bias + MAD significance gate + multiplicative correction)
  added as a fifth scheme. Inert at the tested CNR₀: the gate zeroes the
  correction, so it reproduces `inverse_variance` rather than improving on
  `folded_normal`. The robust / closed-form line is parked behind the poster;
  kept as the record of the negative result.

The corresponding output files live in `output/legacy/`.

**Note:** the `legacy/` scripts set their import path with `../src`, which
resolves to `scripts/src` from `scripts/legacy/` and so does not find the
package — a leftover from when they were moved down a level. Run them with
`PYTHONPATH=src`, or fix the path to `../../src`
(`dense_calibration_and_test_robust.py` already uses the corrected form).
