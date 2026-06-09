# Validation design: estimator selection, truth, and the two workflows

This note fixes how the pipeline's diffusion coefficient and anomalous exponent
are validated: what "truth" means, how it is obtained, and how to choose the
width estimator for a given experiment. It applies to every test bed in the
repository (the analytic profiles, the Saha reaction-diffusion model, the
Mn-doped and nanocrystal-solid maps) and to real measurements.

The method is a selection framework, not a single recipe. The scientist chooses
the estimator that matches their experiment's profile; the truncation correction
applies to the parametric-fit choices; and the physical second moment is the
common yardstick that validates every path.

## Truth is the physical second moment

The quantity to recover is the spatial second central moment of the profile at
each time, σ²(t), and the anomalous exponent α read from its growth, with
MSD(t) = σ²(t) − σ²(0) proportional to t^α. The second moment is the model-free
physical width. Parametric width fits (Gaussian, Voigt, generalized-Gaussian)
are proxies for it. For a profile whose shape matches the estimator, the matched
fit equals the second moment. For a profile outside the estimator's family, most
importantly a mixture of populations of different width, the fit and the second
moment differ. Validation truth is the second moment, on every path.

## Two workflows, two sources of truth

The repository supports two workflows that differ only in where the truth comes
from.

**Validation (simulation).** The profiles are generated from a forward model
with known inputs, so the truth is known a priori: the input parameter (for
example a set σ²(t) = σ0² + 2 D t) or a quantity derived from the inputs (the
second moment of the noise-free profile the inputs produce). For the Saha model
the truth is the second moment of the deterministic noise-free emission map,
fixed by De, Df, ρ, ν and the initial profile. The truth is not estimated here;
it is possessed. The purpose is to add noise, run the chosen estimator and any
correction, and confirm recovery against that known second moment.

**Experimental.** Real data give a noisy profile and no truth. The physical
second moment must be recovered: by fitting the forward simulation to the data
and reading the truth from the inferred parameters, or, if one is derived,
through an analytical relationship. The calibration built in the validation
workflow is the bridge: proven against known truth there, then applied here.

## Selecting the estimator

Choose by the profile's shape and population structure:

| Experiment / profile | Estimator | Note |
| ------------------------------------------------ | ----------------------------------- | ------------------------------------------ |
| Single population, Gaussian | Gaussian | matched: noise-free fit equals second moment |
| Single population with instrument point-spread | Voigt (mixed Gaussian-Lorentzian), fixed γ | recovers the diffusive Gaussian width |
| Single population, anomalous / non-Gaussian shape | generalized-Gaussian, matched exponent | matched single shape |
| Mixture of populations of different width (free charges + excitons; free + trapped) | second moment (effective transport), or forward-model fit (underlying De, Df) | no single shape matches |

The matched-shape rows share one property: with no noise the fit equals the
second moment, so there is no shape gap and the only error to remove is the
noise bias. The multi-population row has no matched single shape; the fit would
differ from the second moment even with no noise, which is why it routes to the
second moment or the forward fit rather than to a single-shape fit.

## The truncation correction is the tool for the parametric-fit paths

On the parametric-fit paths the bounded width fit is precise but biased high at
low CNR: the width cannot go negative, so its sampling distribution is
left-truncated and right-skewed. The folded-normal correction calibrates that
additive noise bias against per-frame CNR and subtracts it (see
`docs/DERIVATION.md`). It removes the noise bias only; it is not a shape
correction.

The moment estimator is the alternative for the multi-population path. It admits
negative values, so it does not truncate and is approximately unbiased,
returning the second moment directly. Its cost is variance: it is noisy at low
CNR and unstable below CNR ≈ 3, so it needs adequate CNR or averaging. It does
not use the folded-normal correction.

The moment route is not clean when the sample is observed through an instrument
point-spread function. A Lorentzian point-spread function has a divergent second
moment, so the second moment of the observed (convolved) profile is not usable.
Deconvolve the point-spread function first, or, more robustly, fit the forward
model with the point-spread function and both populations included. For
realistic multi-population data observed through a point-spread function (the
perovskite transport-microscopy case), forward-model fitting is the clean path,
not the bare moment.

For the underlying two-population parameters (De, Df) rather than an effective
width, fit the forward model to the data; a per-frame width reduction used as
input to that fit is debiased by the folded-normal correction on its fit path.

## Keep the noise bias and the shape gap separate

Do not fold the shape gap into the truncation calibration, and do not anchor the
validation reference to the estimator's own noise-free fit on a multi-population
profile. The truncation bias is a noise effect and depends on CNR. The shape gap
is CNR-independent: it is present with no noise at all, and no choice of
single-shape estimator removes it. On the Saha emission, fitting with a
generalized-Gaussian at the profile's own best-fit exponent (≈ 1.93) gives the
same α as a Gaussian fit, to within 0.001, and both exceed the second-moment α
by up to ≈ 0.09; the Mn-doped maps show the same effect, with a noise-free
Gaussian-fit width of 0.089 µm² against a true second moment of 0.122 µm². The
correct response is selection, not patching: route a multi-population sample to
the moment estimator or the forward fit.

## Computing the second moment

- Define it on the underlying diffusive profile, before any instrument
  point-spread function. A Lorentzian point-spread function has a divergent
  second moment, so the convolved profile's second moment is not usable; the
  Voigt fit recovers the diffusive (Gaussian-component) width, and the truth is
  the diffusive second moment.
- The second moment is sensitive to the tails and to the integration window.
  Compute it on the noise-free profile as the simulation produces it, on a
  window wide enough to capture the tails.
- The second-moment estimator is unbiased only when negative values are
  admitted. Clipping negatives or leaving a positive background reintroduces an
  upward floor bias.

## The validation recipe (any path)

1. Reference: the input-derived second moment of the noise-free profile, and the
   α from its slope.
2. Estimator: selected per the table above.
3. Correction: folded-normal on the parametric-fit paths; none on the moment
   path.
4. Success: the corrected estimate recovers the known second-moment α within the
   path's reliable CNR band (the per-profile floor for the fit paths; the
   variance/stability floor near CNR ≈ 3 for the moment path).

## Where this lives in the code

- Estimators: `models.py` (`GaussianModel`, `VoigtModel`,
  `GeneralizedGaussianModel`, and the non-truncated `MomentModel`, which returns
  the second moment directly).
- Truncation correction and calibration: `weights.py`, `truncation.py`,
  `calibration.py` (the per-frame CNR-to-bias lookup).
- The MSD-to-D and per-frame fits: `fit.py`; the validation sweep: `validate.py`.
- Saha reaction-diffusion bed and its harness:
  `mc_experiments/saha_reaction_diffusion/`.
