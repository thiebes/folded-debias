# Validation design: truth definition and the two workflows

This note fixes what "truth" means when the pipeline's diffusion coefficient and
anomalous exponent are validated, and how that truth is obtained in each
setting. It applies to every test bed in the repository (the analytic profiles,
the Saha reaction-diffusion model, the Mn-doped and nanocrystal-solid maps) and
to real measurements.

## Truth is the physical second moment

The quantity the pipeline must recover is the spatial second central moment of
the profile at each time, σ²(t), and the anomalous exponent α read from its
growth, with MSD(t) = σ²(t) − σ²(0) proportional to t^α. The second moment is
the model-free physical width. Parametric width fits (Gaussian, Voigt,
generalized-Gaussian) are proxies for it. For a single-shape profile a matched
fit equals the second moment. For a multi-population profile, two diffusing
species of different width, no single-shape fit equals it, and the fit and the
second moment diverge. Validation truth is the second moment, never a fit.

## Two workflows, two sources of truth

The repository supports two workflows that differ only in where the truth comes
from.

**Validation (simulation).** The profiles are generated from a forward model
with known inputs, so the truth is known a priori. It is the input parameter
(for example a set σ²(t) = σ0² + 2 D t) or a quantity derived from the inputs
(the second moment of the noise-free profile the inputs produce). For the Saha
model the truth is the second moment of the deterministic noise-free emission
map, fixed by De, Df, ρ, ν and the initial profile. The truth is not estimated
in this workflow; it is possessed. The purpose is to add noise, run the
estimator and the correction, and confirm recovery against that known second
moment.

**Experimental.** Real data give a noisy profile and no truth. The physical
second moment must be recovered, either by fitting the forward simulation to the
data and reading the truth from the inferred parameters, or, if one is derived,
through an analytical relationship. The calibration built in the validation
workflow is the bridge: proven against known truth there, then applied here to
recover unknown truth.

## The validation recipe

1. Reference: the input-derived second moment of the noise-free profile, and the
   α from its slope.
2. Estimator: a precise parametric fit chosen to match the observed profile
   shape. Gaussian for a Gaussian profile, Voigt (mixed Gaussian-Lorentzian) for
   a profile carrying an instrument point-spread function, generalized-Gaussian
   for a near-Gaussian profile with non-zero excess kurtosis. Matching the shape
   buys precision at low contrast-to-noise ratio (CNR), where the second-moment
   estimator itself is too noisy to use directly.
3. Calibration: anchored to the second moment. The calibrated per-frame bias is
   E[fitted variance under noise] − (true second moment), built on profiles of
   the actual shape. This removes the noise-induced (truncation) bias and the
   fit-versus-second-moment shape gap together, so the precise fit recovers the
   physical second moment rather than its own noise-free value.
4. Success: the corrected estimate recovers the known second-moment α within the
   calibration's reliable CNR band (see the per-profile floor).

## The pitfall this corrects

Do not anchor the validation reference to the estimator's own noise-free fit.
For multi-population profiles the noise-free fit differs from the second moment
by an amount that no choice of single-shape estimator removes. On the Saha
emission, fitting with a generalized-Gaussian at the profile's own best-fit
exponent (≈ 1.93) gives the same α as a Gaussian fit, to within 0.001, and both
exceed the second-moment α by up to ≈ 0.09; the Mn-doped maps show the same
effect, with a noise-free Gaussian-fit width of 0.089 µm² against a true second
moment of 0.122 µm². This gap is a property of the estimator's relationship to a
multi-component profile, not of the truncation correction. Anchoring to the fit
measures fit-to-fit recovery and mislabels the residual as an estimator shape
bias; it is instead the reason the truth must be the second moment and the
reason the calibration is anchored to the second moment.

## Computing the second moment

- Define it on the underlying diffusive profile, before any instrument
  point-spread function. A Lorentzian point-spread function has a divergent
  second moment, so the convolved profile's second moment is not usable. The
  Voigt fit recovers the diffusive (Gaussian-component) width, and the truth is
  the diffusive second moment.
- The second moment is sensitive to the tails and to the integration window.
  Compute it on the noise-free profile as the simulation produces it, on a
  window wide enough to capture the tails.
- The second-moment estimator is unbiased only when negative values are
  admitted. Clipping negatives or leaving a positive background reintroduces an
  upward floor bias.

## Simplest instance

The analytic Gaussian test bed is the simplest case of this design. The truth
σ²(t) = σ0² + 2 D t is the input, a Gaussian's second moment equals its σ², and
a Gaussian fit matches the shape, so the fit reference and the second-moment
reference coincide. The distinction becomes load-bearing only for
multi-population or instrument-broadened profiles.

## Where this lives in the code

- Estimators: `models.py` (`GaussianModel`, `VoigtModel`,
  `GeneralizedGaussianModel`, and the non-truncated `MomentModel`, which returns
  the second moment directly).
- Calibration: `calibration.py` (the joint (CNR, σ²) lookup; anchoring it to the
  second moment is the construction to use for non-Gaussian beds).
- Saha reaction-diffusion bed and its validation harness:
  `mc_experiments/saha_reaction_diffusion/`.
