# Validation design: recovering the carrier distribution, truth, and the two workflows

This note fixes the goal of the pipeline, what "truth" means, how truth is
obtained, and how to choose the width estimator. It applies to every test bed in
the repository (the analytic profiles, the Saha reaction-diffusion model, the
Mn-doped and nanocrystal-solid maps) and to real measurements.

## The goal

In the experimental frame the deliverable is the **carrier distribution** and
its transport: where the carriers are at each time, and therefore the
mean-squared displacement and the diffusion coefficient (or, for two coexisting
species, De and Df). The recorded profile is not the carrier distribution.
Recovering the carrier distribution accurately, with the measurement-noise bias
removed, is the objective; everything the pipeline computes serves that.

## Three distortions between the measurement and the carrier distribution

The recorded profile differs from the carrier distribution in three independent
ways. Recovering the carrier distribution means inverting all three, and they
need different tools:

1. **Measurement noise**, and the truncation bias it induces in any bounded
   width estimator at low CNR. A fitted width cannot go negative, so its
   sampling distribution is left-truncated and biased high. This is what the
   folded-normal correction in this repository removes, and it is the only one
   of the three the correction addresses.
2. **The instrument point-spread function.** The recorded profile is the
   emission convolved with the instrument response. Removed by deconvolution, by
   a Voigt (mixed Gaussian-Lorentzian) fit with the known Lorentzian width, or
   inside a forward model.
3. **The emission-to-carrier mapping.** The emission is a nonlinear function of
   the carrier density n and the Saha free-charge fraction x, namely
   E = qe·ν·(1−x)·n + qf·ρ·x²·n². The emission profile is therefore a distorted
   view of the carrier distribution, with a sharper peak and a
   population-dependent shape. This is inverted only with the forward model; no
   width fit recovers it.

So the pipeline is two stages. A **width-extraction** stage produces an
accurate, noise-debiased measurement of the recorded profile's width over time
(distortion 1, and distortion 2 if the estimator deconvolves the
point-spread function). A **forward-model** stage turns that measurement into
the carrier distribution and its transport (distortions 2 and 3). The
folded-normal correction lives in the first stage.

A useful simplification: when emission is linear in carrier density (a single
population emitting in proportion to n, for example excitonic emission), the
emission profile is proportional to the carrier profile, the emission width
equals the carrier width, and distortion 3 vanishes. There the first stage alone
-- a matched width fit with the noise correction -- recovers the carrier
transport. The forward-model stage is needed only when emission is nonlinear in
n (multiple populations, bimolecular emission), which is the perovskite case.

## Truth and the two workflows

Truth is the **carrier distribution**: its second central moment at each time,
and the anomalous exponent α or coefficient D from the moment's growth. It is
not the recorded profile's moment.

**Validation (simulation).** Truth is known a priori. The carrier distribution
is generated from the forward model with known inputs, so its second moment is
an exact, computed quantity, derived from the inputs. The recorded emission
profile's moment is an intermediate the pipeline must map back through; it equals
the carrier moment only when emission is linear in n. Validate the full pipeline
-- noise debias, point-spread function, emission inversion -- against the
carrier-distribution truth.

**Experimental.** Truth is not available. The carrier distribution is recovered
by fitting the forward model (carrier transport, the emission expression, the
point-spread function, and a noise model) to the data, or by an analytical
relationship if one is derived. The sample second moment of the recorded profile
is an estimator here, not truth, and a distorted one when emission is nonlinear;
it cannot adjudicate.

## The width-extraction stage: estimator selection and the noise correction

Choose the estimator by the recorded profile's shape and population structure:

| Recorded profile | Estimator | Note |
| ------------------------------------------------ | ----------------------------------- | ------------------------------------------ |
| Single population, Gaussian | Gaussian | matched: noise-free fit equals the profile's second moment |
| Single population with instrument point-spread | Voigt, fixed γ | recovers the diffusive Gaussian width |
| Single population, anomalous / non-Gaussian shape | generalized-Gaussian, matched exponent | matched single shape |
| Mixture of populations of different width (free charges + excitons; free + trapped) | second moment, or forward-model fit | no single shape matches the recorded profile |

On the parametric-fit choices the bounded width fit is precise but biased high at
low CNR; the folded-normal correction calibrates that additive noise bias against
per-frame CNR and subtracts it (see `docs/DERIVATION.md`). It removes the noise
bias only; it is not a shape correction and not an emission inversion.

The moment estimator is the alternative for the multi-population recorded
profile. It admits negative values, so it does not truncate and is approximately
unbiased, returning the recorded profile's second moment directly. Its cost is
variance: it is noisy at low CNR and unstable below CNR ≈ 3, so it needs adequate
CNR or averaging, and it takes no folded-normal correction. It is not clean when
the profile carries a Lorentzian point-spread function, which has a divergent
second moment; deconvolve first, or use the forward fit.

## Two gaps, kept distinct

- The **shape gap** is within the width-extraction stage: a single-shape fit of
  the recorded profile differs from that profile's own second moment when the
  profile is multi-population. It is a selection issue (use a matched estimator
  or the moment estimator), not something the noise correction removes. On the
  Saha emission, a generalized-Gaussian fit at the profile's best-fit exponent
  (≈ 1.93) gives the same α as a Gaussian fit to within 0.001, and both exceed
  the recorded profile's second-moment α by up to ≈ 0.09; the Mn maps show the
  same (Gaussian-fit width 0.089 µm² against a recorded second moment of
  0.122 µm²).
- The **emission-to-carrier gap** is the forward-model stage: the recorded
  profile's second moment differs from the carrier distribution's second moment
  when emission is nonlinear in n. It is inverted by the model, not by any width
  estimator.

Conflating the two, or folding either into the noise calibration, is the error to
avoid. The folded-normal correction's scope is distortion 1 only.

## Computing the second moment

- Distinguish the carrier moment (the truth) from the recorded-profile moment
  (the width-extraction target). They coincide only for linear emission.
- Take it on the underlying profile, before any instrument point-spread
  function. A Lorentzian point-spread function has a divergent second moment, so
  the moment of the convolved profile is not usable.
- The second moment is sensitive to the tails and the integration window;
  compute it on the noise-free profile as the simulation produces it, on a wide
  enough window.
- The moment estimator is unbiased only when negative values are admitted;
  clipping negatives or leaving a positive background reintroduces a floor bias.

## Validation recipe

1. Reference: the carrier distribution's second moment from the known inputs, and
   its α.
2. Width extraction: the matched estimator on the recorded profile, with the
   folded-normal correction on the parametric paths (or the moment estimator).
3. Carrier recovery: forward-model inversion of the point-spread function and the
   emission mapping (trivial when emission is linear in n).
4. Success: the recovered carrier-distribution α matches the known input within
   the reliable CNR band (the per-profile floor for the fit paths; the
   variance/stability floor near CNR ≈ 3 for the moment path).

## Where this lives in the code

- Estimators (width extraction): `models.py` (`GaussianModel`, `VoigtModel`,
  `GeneralizedGaussianModel`, and the non-truncated `MomentModel`).
- Noise correction and its calibration: `weights.py`, `truncation.py`,
  `calibration.py`.
- Per-frame and MSD-to-D fits, and the validation sweep: `fit.py`, `validate.py`.
- Forward models (the emission-to-carrier mapping, for both simulation and the
  experimental inversion): `simulate.py` and the per-bed models under
  `mc_experiments/`, including `mc_experiments/saha_reaction_diffusion/`.
