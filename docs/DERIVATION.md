# Folded-normal derivation of the MSD bias and its correction

This note derives, from a folded-normal model of the fitted width, why the
diffusion coefficient is overestimated and what the correct fix is. The model
is the one the boundary diagnostic selected: `scripts/diagnose_boundary.py`
found the fitted-variance distribution strictly interior and right-skewed with
no delta-mass at the lower bound, for all three profile models — **reflection,
not clipping**.

## 1. Model

At each frame *i* the analysis fits a profile and reports a width. A fitted
width cannot be negative. Reflection at that wall means the latent width
estimate behaves as the absolute value of a Gaussian: writing the latent
estimate as `X_i`, the reported width is `|X_i|` with

    X_i ~ N(σ_i, s_i²),

where `σ_i` is the true width at frame *i* and `s_i` is the per-frame
estimation noise (a decreasing function of the per-frame CNR). The fitted
**variance** is therefore the square of a Gaussian,

    σ̂²_i = |X_i|² = X_i².

Equivalently `σ̂²_i / s_i²` is a non-central χ²₁ with non-centrality `(σ_i/s_i)²`
— continuous, strictly positive, right-skewed, exactly the shape the diagnostic
saw.

## 2. Moments

For `X ~ N(σ, s²)`, using `E[X²] = σ² + s²` and `E[X⁴] = σ⁴ + 6σ²s² + 3s⁴`:

    E[σ̂²]   = σ² + s²
    Var[σ̂²] = E[X⁴] − E[X²]² = 4σ²s² + 2s⁴ = 2s²(2σ² + s²)

**The fitted variance carries an additive upward bias equal to the estimation
noise variance `s²`.** This is the Gaussian-width analogue of the Rician
identity `Â² = M² − σ²` (Gudbjartsson–Patz 1995) and of the Wardle–Kronberg
polarization estimator.

## 3. The per-frame truncation fraction

Define the latent width signal-to-noise `α_i = σ_i / s_i`. The truncation
fraction — the mass of the latent estimator below the wall — is

    p_i = P(X_i < 0) = Φ(−α_i),     so   α_i = −Φ⁻¹(p_i).

`p_i` is a clean, dimensionless readout of `α_i`, and on real data it follows
from the per-frame CNR (via `fft-cnr`). In terms of `p`:

- relative bias of σ̂²:      `s²/σ² = 1/α²              = 1 / Φ⁻¹(p)²`
- relative variance of σ̂²:  `Var[σ̂²]/σ⁴ = 2(2α²+1)/α⁴`, with `α = −Φ⁻¹(p)`

So the *shape* of the bias and variance is fixed by `p` alone. The *scale* is
not: `p` is dimensionless, the bias `s²` has units of variance. The absolute
scale must come from the per-frame CNR through a noise model (the localization
Cramér–Rao bound, `s² ∝ σ²/CNR²` to leading order) or by calibration against
matched simulations. Both `p` and the CNR scale are needed — which is why the
weighting takes the per-frame CNR as its input, not `p` alone.

## 4. Why the bias reaches D, and why a weight alone cannot remove it

The MSD point is `MSD_i = σ̂²_i − σ̂²_0`, so

    E[MSD_i] = (σ_i² − σ_0²) + (s_i² − s_0²) = MSD_true,i + (s_i² − s_0²).

Because CNR falls frame by frame, `s_i` grows with *i*; the bias `s_i² − s_0²`
is an increasing function of time. An increasing additive offset on `MSD(t)`
tilts the fitted slope upward — `D` is overestimated, the more so at low CNR.

A reweighting of the MSD-vs-t fit **cannot** remove this. Weights change which
points dominate the slope; they cannot subtract a mean offset that is present
in every point. Down-weighting late frames trades the bias for variance but
never eliminates it. The bias must be **subtracted**, not weighted away. This
is the central result: the standard inverse-variance scheme is structurally
unable to fix the bias, however the weights are chosen.

## 5. The derived method

1. **Bias-correct each frame.** Subtract the estimation-noise variance:
   `ṽ²_i = σ̂²_i − s_i²`, so `E[ṽ²_i] = σ_i²`. Build the MSD from the corrected
   variances: `MSD_i = ṽ²_i − ṽ²_0`.
2. **Weight by inverse variance.** `w_i = 1 / (Var[σ̂²_i] + Var[σ̂²_0])` with
   `Var[σ̂²] = 2s²(2σ² + s²)`. Subtracting a (calibrated, deterministic) bias
   does not change the variance, so the weight is the ordinary inverse-variance
   weight; the bias correction is the new ingredient.

In `weighting_wall` this is the `folded_normal` scheme. It differs from
`inverse_variance` *only* by step 1, so comparing the two isolates the effect
of the bias correction.

## 6. POC implementation

The per-frame noise variance `s_i²` is obtained by calibration: a Monte Carlo
CNR sweep gives `E[σ̂²] − σ²_true` as a function of CNR — the `var_bias(CNR)`
model in `TruncationModel`. The folded-normal model is what licenses this:
it tells us the bias is *additive*, *positive*, and a function of the per-frame
CNR, so a one-dimensional `var_bias(CNR)` calibration is the right object to
build. Calibration and evaluation use independent random seeds. On real data
the same calibration is done against simulations matched to the measurement
(Koay-style simulation-based debiasing).

## 7. Two alternatives, tested

`folded_normal` subtracts the *calibrated* `var_bias(CNR)`. When that
calibration is imperfect — it conflates frames of equal CNR but different true
width, and the low-CNR mean is heavy-tailed — the hard subtraction can
overshoot and push `D` below the truth in the deepest noise. Two routes were
tested against this.

**MAS-gated subtraction (`mas` scheme).** Following Plaszczynski's Modified
Asymptotic estimator, the subtraction is passed through a smooth gate:

    ṽ²_i = σ̂²_i − s_i² · (1 − exp(−σ̂²_i / s_i²)).

Writing `u = σ̂²/s²`, the corrected value is `s²·(u − 1 + e^−u)`, which is ≥ 0
for all `u ≥ 0` and increasing — the correction **cannot drive the variance
negative**. The gate is ≈ 1 when the frame is well above the noise (full `s²`
subtraction, matching `folded_normal`) and → 0 when it is noise dominated (the
frame is left uncorrected rather than overshot). On the narrow `CNR₀ ∈ [5,90]`
sweep (`scripts/moment_estimator_failure.py`, the earlier-named
`compare_methods.py`), `mas` gave the lowest bias of any
scheme — mean |⟨D⟩/D_true − 1| = 0.089 vs `folded_normal` 0.116 — and
removed the deep-regime overshoot.

**On the wider `CNR₀ ∈ [2,100]` sweep, however, the recommendation flips.**
At very low CNR₀ (≲ 4) the MAS gate switches the correction off where the
truncation bias is large, leaving most of the raw inflation in. On
`scripts/dense_calibration_and_test.py`'s 100-point sweep across all three
profile shapes, folded-normal's overall mean |⟨D⟩/D_true − 1| beats MAS:
Gaussian 0.195 vs 0.497, Voigt 0.145 vs 0.237, GenGauss 0.152 vs 0.172.
The medians remain nearly equal (the gap is in the deep-CNR tail), so for
typical measurements either scheme works. But **folded-normal degrades
gracefully** in deep noise (residuals erratic but bounded), while MAS
**fails ungracefully** (large positive residuals that grow with CNR
depth — the gate-off-leaves-bias problem). **Folded-normal is the
recommended scheme.** MAS remains in the codebase for paper-scope
comparison.

**Non-truncated moment estimator (`MomentModel`).** The alternative is to
remove the wall entirely: estimate the variance from the discrete second
moment of the profile, which is linear in the data, admits negative values,
and so needs no correction (cf. Vestergaard's covariance-based estimator,
unbiased *because* it allows negative values). It **fails here**: the moment
is a ratio, second moment over total intensity, and the denominator passes
through zero at low CNR, producing wild, often negative `D` (mean bias ~10³,
enormous spread). Removing the wall is sound in principle, but it needs a
*stable* non-truncated estimator — the covariance-based estimator achieves
this on raw displacement increments, which an intensity-profile MSD pipeline
does not have. An intensity-profile moment is not a viable substitute.

## 8. Calibration without knowledge of D

The calibration step measures `s²(CNR)` empirically by running synthetic
profile sequences and recording the empirical mean of `σ̂² − σ²_true` per
per-frame CNR. A naive implementation of that step (`calibrate(exp, ...)`)
takes the calibration experiment's `D` as an input — methodologically
awkward, since the goal is to estimate `D` from real data where `D` is
unknown, and meaningless on heterogeneous transport where no single `D`
describes the dynamics.

The fix is small. The folded-normal bias `s²(CNR)` is dominated by per-frame
CNR; the dependence on `σ²(t)` is weak (the second-order term in the
folded-normal expansion). In the low-`D` regime where the truncation bias
bites, `σ²(t)` barely changes from `σ_0²` across the measurement window, so
calibrating at `σ² = σ_0²` (i.e. `D = 0`) captures the bias the application
actually sees. `calibrate_d_free(cnr0_grid, ...)` is the thin wrapper that
runs the standard calibration sweep at `D = 0`; the lookup it returns is
keyed on per-frame CNR alone and is reused unchanged across any application
`D`.

Validated by Monte Carlo across `D ∈ [0.005, 0.1]` on Gaussian, Voigt, and
generalized-Gaussian profiles (`scripts/validate_d_zero.py`). In the
poster-relevant regime `D ≥ 0.02` (`L_D / FWHM ≥ 4%`), the corrected mean
|bias − 1| is 0.01–0.10 per shape with the same calibration. At `D < 0.02`
the residual bias grows (`D ≈ 0.005` has `L_D ≈ 1.5% FWHM`, near the
information-theoretic floor for *any* unbiased estimator at these CNRs); the
calibration is not at fault.

An experimental joint `var_bias(CNR, σ²)` calibration
(`weighting_wall.calibration.calibrate_cnr_sigma2`) was prototyped for
cases of large `σ²` spread (high-`D`, heterogeneous transport). It is not currently
competitive: the per-cell Monte-Carlo noise at low CNR dominates the lookup,
and the additional sample budget needed to fix that (≳ 5000 runs/cell plus
post-hoc smoothing) is not yet justified. See the module docstring for
status.

## 9. Where the correction works: the per-profile floor

Because the slope fit is a deterministic linear functional of the corrected
per-frame variances (the weights are deterministic in CNR), the bias in
`D̂` is fully determined by the per-frame residuals `E[σ̃²(t)] − σ²_true(t)`.
The cleanest characterisation is therefore an isolated single-profile
sweep: generate many noisy profiles at a chosen per-profile CNR (no
diffusion, no decay, no frame indexing), apply the calibrated
`var_bias(CNR)` subtraction, and measure the residual mean with its MC
standard error.

`scripts/per_profile_floor.py` runs this sweep with 50,000 isolated fits
per CNR cell and the dense calibration. For Gaussian profiles at
`σ²_true = σ_0²`:

| per-profile CNR | folded-normal residual (σ_0²) | reading |
|---|---|---|
| ≳ 1 | \|residual\| ≲ 0.01 | correction excellent (≲ 1% of σ²_true) |
| 0.7 – 1.0 | up to ≈ 0.05 | useful; a few % of σ²_true |
| 0.5 – 0.7 | 0.1 – 0.3 | unreliable per realisation; mean still toward truth |
| 0.3 – 0.5 | 0.3 – 0.8 | failing; residuals comparable to σ²_true |
| < 0.3 | ≳ σ²_true | broken |

Voigt and generalized-Gaussian sweeps in progress (launched 26 May).
Expected to show the same floor *location* with smaller residual
*magnitudes* — at very low CNR, Gaussian profiles inflate the most (E[raw]
≈ 6.7 σ_0² at per-profile CNR 0.25), while Voigt and GenGauss inflate
only ≈ 1.3× because their heavier tails resist the truncation fold.

### Analytical handle on where the model breaks

The calibrated `var_bias(CNR)` follows the localization-CRLB power-law
`var_bias(CNR) ≈ A / CNR²` σ_0² at high CNR. Fitting that form to the
calibration on `CNR > 8` (`scripts/analytical_floor.py`) gives `A ≈ 0.37`
σ_0² for Gaussian (0.42 for Voigt, 0.37 for GenGauss). The power-law
holds from CNR = 0.2 up to 100. Below CNR ≈ 0.1, the empirical
`var_bias(CNR)` saturates because the σ̂² distribution is bounded above
by the fit's `var_max ≈ 144 σ_0²` — values that would land beyond pile
up at the ceiling instead. So `E[σ̂²]` stops growing as `1/CNR²` and
approaches a plateau.

This saturation is *not* a failure of the correction in the
mean-unbiased sense — the calibration captures the empirical mean
whether or not the distribution is folded-normal, and subtracting the
calibrated mean still recovers `σ²_true` in expectation. **There is no
fundamental analytical floor of the correction**; the practical floor is
set by calibration MC budget. The gap between the per-profile **working
floor** (CNR ≈ 0.5–1.0) and the **model-breakdown CNR** (≈ 0.1) is what
denser calibration could close, at exponential cost.

### Design recommendation that follows

A measurement is bias-correctable end-to-end if **every frame's
per-profile CNR exceeds the working floor**. Equivalently, the deepest
frame's CNR — `CNR₀ · exp(−t_max/τ)` — must satisfy

    CNR₀ · exp(−t_max/τ)  ≳  1.

For the typical `t_max = 3 τ` this means `CNR₀ ≳ exp(3) ≈ 20`. If
experimental constraints prevent that, the late frames must be discarded
or replaced with denser early-time sampling so the slope fit is
dominated by frames inside the working regime.

## 10. Robust calibration and multiplicative correction

(In development.) Two failure modes of the basic Section-8 calibration and
Section-5 subtractive scheme, surfaced once the method was applied beyond
the headline `L_D / FWHM ≤ 0.20` regime:

1. **Calibration-vs-application width mismatch.** The 1-D D-free
   calibration measures `s²(CNR)` at `σ²_true = σ_0²` (the calibration
   condition at `D = 0`). When the application's `σ²(t)` varies
   substantially across the measurement (e.g., reaction-diffusion data
   with large variance growth, or any moderate-to-high `D`), the
   subtracted bias is the wrong absolute value for the data's actual
   width.
2. **Heavy-tail contamination of the calibrated mean.** At low CNR, a
   fraction of fits peg at the fitter's upper bound `var_max ≈ window²`
   (Section 9's saturation regime). The sample mean of `σ̂²` is dragged
   upward by this tail; the contribution scales with `var_max`, not with
   `σ_0²`, and so breaks any clean functional form of the bias.

The two together produce over-correction at moderate-to-low CNR when the
calibration's `σ_0` does not match the data's `σ²(t)`. Both are addressed
by replacing the calibration summary statistic with a robust one, and
applying the correction multiplicatively rather than additively.

### 10.1 Multiplicative form

A CRLB-based scaling argument gives, to leading order at fixed pixel-grid
geometry,

    bias(CNR, σ²_true) = σ²_true × g(CNR),

with `g` dimensionless. Linearity in `σ²_true` means
`E[σ̂²] = σ²_true × (1 + g(CNR))`, which inverts in closed form:

    σ²_true ≈ σ̂² / (1 + g(CNR)),     g(CNR) = bias_cal(CNR) / σ_0².

This is the **multiplicative correction**. It collapses to the
subtractive form `σ̂² − σ_0² × g(CNR)` when `σ²_true ≈ σ_0²` (the
calibration's working condition), so headline-regime results carry over
unchanged. It scales correctly when `σ²_true ≠ σ_0²`, and it stays
positive — `σ̂² / (1 + g) > 0` for any `g ≥ 0` — which removes the
subtractive form's negative-variance pathology below the per-profile
floor.

The linearity assumption is exact in the asymptotic CRLB regime; it
weakens once a non-negligible fraction of fits enters the broken-fit
saturation regime (Section 9's analytical-floor discussion), and there is
a residual `σ/dx` dependence at finer pixel resolutions that we do not
yet capture in the 1-D lookup. Both effects are second-order in the
headline regime.

### 10.2 Robust calibration statistics

For each (per-frame CNR) calibration cell with `N` independent noisy
realizations producing fitted variances `σ̂²_1, …, σ̂²_N`, replace
sample mean and sample std with:

    m   = median(σ̂²_1, …, σ̂²_N),                       (location)
    MAD = median(|σ̂²_1 − m|, …, |σ̂²_N − m|).            (scale)

The MAD is the median absolute deviation from the median — a single
median operation, no centring around the mean. It ignores the upper tail
that the broken-fit pile-up contributes.

Under the assumption that the **bulk** of the σ̂² distribution
(excluding the broken-fit tail) is approximately Normal — which holds in
the analytical CRLB regime that the actual bias inhabits — two standard
statistical conversions apply:

    σ_bulk ≈ MAD / Φ⁻¹(0.75) ≈ MAD × 1.4826,             (1)

    SE_median ≈ √(π / 2) × σ_bulk / √N ≈ 1.253 × σ_bulk / √N.   (2)

The factor `Φ⁻¹(0.75) ≈ 0.6745` in (1) is the 75th-percentile point of
the standard Normal: for `X ~ N(0, σ²)`, `P(|X| ≤ 0.6745 σ) = 0.5`, so
the median of `|X|` is `0.6745 σ`, giving the inversion above. The
factor `√(π/2) ≈ 1.2533` in (2) is the asymptotic ratio of the
sample-median standard error to the sample-mean standard error for a
Normal sample (sample mean has variance `σ²/N`; sample median has
asymptotic variance `π σ² / (2N)`).

Combining,

    SE_median ≈ 1.4826 × 1.253 × MAD / √N ≈ 1.858 × MAD / √N.   (3)

(3) gives the standard error of the calibrated median bias from each
cell's MAD and its sample count. Unlike the sample-std-based SE, it is
not inflated by the broken-fit tail — `var_max`-pegged outliers do not
move the MAD because they are far from the median; the median itself
sits in the bulk.

### 10.3 Significance gate

A calibrated `median_bias(CNR)` is meaningful only where it stands above
its own MC noise. Define a two-sigma gate:

    apply correction iff   |median_bias(CNR)| > 2 × SE_median(CNR).

Cells that fail the gate contribute `g(CNR) = 0` to the lookup. The
correction at any per-frame CNR interpolating into a failed cell passes
the data through unchanged. This prevents the multiplicative correction
from amplifying calibration noise at moderate-to-high CNR where the true
bias is essentially zero but our finite-`N` median bias estimate
fluctuates around it.

### 10.4 Algorithm summary

Calibration, once per profile model and geometry:

1. Run a `D = 0` MC sweep across a CNR_0 grid with `N` realizations per
   cell. Each cell yields `N` independent samples of σ̂² at a per-frame
   CNR.
2. For each cell, compute `m`, `MAD`, and `SE_median` from (3); record
   `(CNR, median_bias = m − σ_0², SE_median, rel_variance)` where
   `rel_variance = Var(MSD)` is the empirical variance of the per-frame
   MSD used for inverse-variance weighting.
3. Concatenate all cells into a 1-D lookup sorted by per-frame CNR.

Application, per noisy realization:

1. For each frame, interpolate `(median_bias, SE_median)` at the frame's
   CNR.
2. Apply the gate (Section 10.3). Set `g = median_bias / σ_0²` where the
   gate passes, otherwise `g = 0`.
3. Replace each σ̂² with σ̂² / (1 + g) (Section 10.1).
4. Form MSD = corrected σ̂²(t) − corrected σ̂²(0) and fit with
   inverse-variance weighting from the calibration's `rel_variance`.

Implemented in `scripts/dense_calibration_and_test_robust.py` (headline-
regime validation, three profile shapes) and
`scripts/marc_mn_trapping_plan_a.py` (Marc reaction-diffusion
application).

## 11. Open points

- Subdiffusive transport (`MSD ∝ t^α`, α < 1) — confirm the correction
  de-biases α as well as D.
- Voigt and generalized-Gaussian models showed a minor (~5%) clipping pile-up
  at CNR ≈ 1; a small censored term may be needed there, on top of the
  folded-normal correction, for the deepest-noise frames.
- Validation on heterogeneous-transport Monte Carlo from Marc Meléndez
  (effective `D` from a population with site-dependent mobility) — the test
  that motivated the D-free calibration in the first place.
- Validation on trapped-exciton-background data from Marc (pulsed-laser
  accumulated background; structurally a non-zero `B(x)` added to each
  frame).
