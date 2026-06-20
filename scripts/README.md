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

- **`subdiffusive_alpha.py [profile_model] [--quick] [--floor F]`** — Clean
  single-population synthetic test that the correction de-biases the anomalous
  exponent `α`, not only `D`. On the `AnomalousExperiment` bed
  (`σ²(t) = σ₀² + 2 D t^α`, with `D` set per `α` to match the end-point
  diffusion length `L_D / FWHM₀ ≈ 0.15`) it sweeps true `α ∈ {0.6, 0.8, 1.0,
  1.2}` × CNR₀ and fits `MSD = 2 D t^α`, comparing the naive fit (raw, all
  frames) against folded-normal + the per-frame CNR floor cut. The bias is read
  from the mean trajectory; the per-realisation spread is the precision band.
  Because the simulated peak tracks decay only, the per-frame CNR is
  `CNR₀·exp(−t/τ)` for every `α` and `D`, so the exponents are compared at
  matched noise. The D-free calibration is `α`-independent (built at `D = 0`)
  and cached. *The clean synthetic version of the disordered-NC-solid `α`
  result; source for the poster's `α`-recovery point (Figure 6).* Finding: the
  naive exponent pins at the ballistic `α = 2.0` below the design floor
  (CNR₀ ≲ 20), while the correction recovers true `α` to within
  `|Δα| ≈ 0.02–0.10` for CNR₀ ≥ 20 across all four exponents.
  Wall time ≈ 10 min (60-point × 1000s-run calibration, then 4 × 7 × 2000
  evaluation); the cached calibration makes re-runs ≈ 3 min, and `--quick` is a
  small sanity budget. `--floor` overrides the per-frame CNR floor cut
  (default 1.0) for sensitivity studies, reusing the cached calibration.
  Outputs: `output/subdiffusive_alpha[_<model>].{png,txt}`,
  `output/calibration_subdiffusive[_<model>].npz`.

- **`multiexp_decay.py`** — Confirms the correction is not invalidated by
  multi-exponential decay. On fixed Fickian diffusion with three decay laws (one
  single-exponential, two bi-exponential, the `MultiExpExperiment` bed), it shows
  (a) the per-frame corrected residual collapses onto one curve vs *per-frame*
  CNR (across-law spread ≤ 0.011 σ₀² for CNR ≥ 1), so the correction keys on
  per-frame CNR not the decay law; (b) corrected `⟨D⟩/D_true` stays within
  0.9–1.1 across decay laws; (c) the *naive* `⟨D⟩/D_true` is strongly
  decay-specific (19× at CNR₀ = 8 for single-exponential vs ~1.3× bi-exponential),
  so any CNR₀-indexed threshold (the `CNR₀ ≳ 20` design rule, NC-solid
  `CNR₀ ≳ 50`) is single-exponential-specific — state thresholds in per-frame
  CNR. Reuses the cached `calibration_subdiffusive.npz`; wall time ≈ 6 min.
  Outputs: `output/multiexp_decay.{png,txt}`.

- **`coverage_check.py`** — Does the reported confidence interval on `D` cover
  `D_true` at the nominal rate (not just the mean)? `fit_D` now returns a
  covariance-based `D_se`; this sweeps CNR₀ and reports empirical coverage of the
  95% interval for the de-biased `folded_normal` (coverage is meaningless for the
  biased `inverse_variance`, shown as the failing contrast), a parametric
  bootstrap interval, and an analytical attribution
  (`corr_inflation = √(Var_full/Var_diag)`). **Finding:** the shared `σ̂²₀` is
  common-mode (it shifts the intercept, not the slope), so it is *not* the cause
  of under-coverage (inflation 0.91–0.93 < 1) — correcting the anticipated
  mechanism. The covariance CI instead under-covers erratically in the tight-CI
  regime (0.74–0.79) because the D-free calibration mis-estimates the absolute
  per-point variance at the application's D; the bootstrap covers ~0.94
  throughout. Recommendation: report a bootstrap/profile interval, not the
  covariance CI. Reuses `calibration_subdiffusive.npz`; wall time ≈ 5 min.
  Outputs: `output/coverage_check.{png,txt}`.

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

### Real-data validation (Enrique's stroboSCAT measurements)

- **`wse2_realdata_panel.py [n_scans]`** *(reads the pshare; data uncommitted)* —
  The correction on **real** bulk-WSe2 stroboSCAT diffusion data (Enrique Arévalo
  Rodríguez), the poster's real-data panel. Reduces the raw long-delay scans
  following Enrique's own pipeline (`P:\Lab\Enrique\python_enrique\MultiChannel\`):
  pump-on/off differential `ON/OFF−1`, **sign fixed per delay by a spot ROI**,
  width from a **line-cut band through the spot** (avoids the interference
  fringes / edge scatterers), and **time-since-pump = −delay** (the long
  `[−10, 0.65]` scans cover the full 0–10 ns window). The data reads as *white*
  noise (fft-cnr noise model: gain ≈ 0, constant floor — the bright probe
  pedestal), so a white-matched D-free `var_bias(CNR)` calibration at the data
  geometry is built and cached (`output/calibration_wse2_white.npz`). **Finding:**
  merging the 3 long scans at one power keeps per-frame CNR ≥ 2.6 (above the
  floor); the correction is ~0 at high-CNR frames and reduces the inflated
  late-frame widths, giving **D −7% (naive 1.10 → corrected 1.02, px²/ns)** — a
  measurable truncation bias on real data. A single scan (`n_scans=1`) reaches
  CNR ~0.8–0.9 (below the floor) where the correction over-subtracts
  catastrophically without a floor cut — a vivid illustration of the design rule.
  Widths are in px² (the pixel→µm scale is an open `X(um)` question with Enrique).
  Wall time ≈ 10 min on first run (calibration build, serial); instant after
  (cached). Outputs: `output/wse2_realdata_panel[_<n>scan].png`,
  `output/calibration_wse2_white.npz`.

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

- **`br_halide_validation.py`** *(preliminary)* — The correction on Marc's Br-I
  halide-mixing maps (kinetic MC of excitons on a halide-disorder landscape; data
  read from the pshare via `mc_experiments/halide_mixing/br_data.py`, kept
  uncommitted). Gaussian fit + folded-normal + floor cut; naive vs corrected vs
  the particle (carrier) truth, across the three Br fractions and a CNR₀ sweep,
  with a Gaussian D-free calibration at the Br-I geometry (cached). **Findings:**
  halide mixing suppresses transport (apparent α 0.97 → 0.90 and D 3–10× below the
  clean-layer D0 as the Br fraction rises) and the bounded fit recovers the
  carrier-truth α where the moment estimator reads it low (the 150 nm box clips
  the moment's tails). But the bed's 0.1 ns window keeps every frame's per-frame
  CNR well above the floor, so the correction is *inactive* (naive ≈ corrected) —
  a cross-material coverage + fit-vs-moment point, not a correction stress test.
  A longer-window Br-I run (requested from Marc, `docs/email_collaborators.md`
  Email 4) is needed to actually stress the correction.
  Outputs: `output/br_halide_validation.{png,txt}`,
  `output/calibration_br_halide.npz`.

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

- **`unbounded_fit_comparison.py`** — Head-to-head answering "is the correction
  even needed?": the folded-normal correction vs an *unbounded* width fit (the
  same shape-matched Gaussian fitted in the precision `b = 1/σ²`, so the fitted
  variance `1/b` can take any sign and the wall at zero is genuinely removed),
  with the bounded+OLS baseline and the `MomentModel` as references. Reports
  bias and precision at two levels: per-frame (isolated single profiles vs
  per-frame CNR) and D-level (a full diffusion experiment vs CNR₀). **Findings:**
  above per-frame CNR ≈ 2–3 the bounded and unbounded fits are identical and
  both unbiased (the wall never binds, the correction is moot); below ≈ 1.5
  removing the wall only *partly* debiases (per-frame CNR 0.5: mean σ̂² ≈ 2.4 →
  1.6 σ₀², still well above the true 1.0) and explodes the variance, because a
  nonlinear shape-matched fit intrinsically resists negative variances — a
  localized positive noise bump fits better than a global convex trend, so
  negatives stay rare even on pure noise (≈ 0.03 fraction). A variance-
  parameterised "unbounded" fit cannot reach negatives at all (σ²=0 is an
  uncrossable singularity). Only the linear `MomentModel` robustly un-folds, and
  it is catastrophic. *The poster/paper "removing the positivity constraint does
  not substitute for the correction" leg.*
  Outputs: `output/unbounded_fit_comparison.{png,txt}`.

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
