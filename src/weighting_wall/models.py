"""Profile models: how each frame's spatial profile is shaped and fitted.

The truncation-at-zero bias is not specific to a Gaussian fit -- it afflicts
*any* non-negative width parameter. This module makes the profile model
pluggable so the boundary diagnostic, the Monte Carlo, and the weighting can be
run for each of:

* :class:`GaussianModel` -- the baseline; a pure Gaussian profile.
* :class:`VoigtModel` -- a diffusing Gaussian convolved with a *fixed*
  Lorentzian instrument PSF. This is the model Prins' group uses (Solari et
  al. 2026 fit Voigt profiles). The diffusive width is the Gaussian component
  sigma_G; the Lorentzian HWHM gamma is a fixed instrument parameter.
* :class:`GeneralizedGaussianModel` -- a profile with a tunable shape exponent
  p (p = 2 Gaussian, p < 2 heavy-tailed, p > 2 flat-topped), matching the
  generalized-Gaussian family fitted by the ``fft-cnr`` package. This is the
  handle for non-Gaussian transport (e.g. subdiffusive nanocrystal solids).

Every model exposes the same two-method interface used by ``montecarlo``:

* ``noise_free_series(exp)`` -- the clean profile stack for an Experiment.
* ``fit_variance(x, y)``     -- fit one noisy profile, return its diffusive
  variance as a :class:`~weighting_wall.fit.FrameFit`.

For every model the *diffusive variance* (the quantity that grows as
sigma^2(t) = sigma_0^2 + 2 D t and enters MSD(t) = variance(t) - variance(0))
is the same physical quantity; only the profile shape and the estimator differ.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gamma as _gamma
from scipy.special import voigt_profile as _voigt

from .fit import FrameFit, fit_variance as _fit_gaussian_variance
from .simulate import Experiment, gaussian, noise_free_series as _gaussian_series


# --------------------------------------------------------------------------
# Gaussian -- the baseline model (delegates to the existing implementation)
# --------------------------------------------------------------------------
class GaussianModel:
    """Pure Gaussian profile; the model the original POC was built on."""

    name = "gaussian"

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        return _gaussian_series(exp)

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        return _fit_gaussian_variance(x, y)


# --------------------------------------------------------------------------
# Voigt -- diffusing Gaussian convolved with a fixed Lorentzian PSF
# --------------------------------------------------------------------------
def _voigt_peaknorm(x: np.ndarray, amp: float, mean: float,
                     sigma_g: float, gamma: float) -> np.ndarray:
    """Voigt profile scaled so that ``amp`` is the peak height."""
    prof = _voigt(x - mean, sigma_g, gamma)
    peak = _voigt(np.array([0.0]), sigma_g, gamma)[0]
    return amp * prof / peak


@dataclass
class VoigtModel:
    """Diffusing Gaussian (variance sigma_G^2) convolved with a fixed
    Lorentzian PSF of half-width-at-half-maximum ``gamma``.

    The Voigt fit deconvolves the fixed instrument Lorentzian from the
    diffusing Gaussian, so the recovered, truncation-prone width is the
    Gaussian component sigma_G; MSD(t) = sigma_G^2(t) - sigma_G^2(0).
    """

    gamma: float = 0.3          # Lorentzian HWHM, in sigma_0 units (fixed PSF)
    fit_gamma: bool = False     # if True, gamma is a free fit parameter too
    name: str = "voigt"

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        x = exp.x
        return np.array([
            _voigt_peaknorm(x, a, exp.mean, np.sqrt(v), self.gamma)
            for a, v in zip(exp.amplitude, exp.true_variance)
        ])

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        span = float(x[-1] - x[0])
        sg_min = span / x.size           # ~ one pixel
        sg_max = 0.45 * span
        sg0 = max(min(span / 8.0, sg_max), 2 * sg_min)
        amp0 = float(np.max(np.abs(y))) or 1.0
        mean0 = float(x[int(np.argmax(y))])

        if self.fit_gamma:
            def model(xx, amp, mean, sg, gam):
                return _voigt_peaknorm(xx, amp, mean, sg, gam)
            p0 = [amp0, mean0, sg0, self.gamma]
            lo = [0.0, x[0], sg_min, sg_min]
            hi = [2 * amp0, x[-1], sg_max, sg_max]
        else:
            def model(xx, amp, mean, sg):
                return _voigt_peaknorm(xx, amp, mean, sg, self.gamma)
            p0 = [amp0, mean0, sg0]
            lo = [0.0, x[0], sg_min]
            hi = [2 * amp0, x[-1], sg_max]

        try:
            popt, _ = curve_fit(model, x, y, p0=p0, bounds=(lo, hi), maxfev=10000)
            sigma_g = float(popt[2])
            amp = float(popt[0])
            ok = True
        except (RuntimeError, ValueError):
            sigma_g, amp, ok = sg0, amp0, False

        at_wall = sigma_g <= 2 * sg_min
        return FrameFit(variance=sigma_g ** 2, amplitude=amp,
                        at_wall=bool(at_wall), ok=ok)


# --------------------------------------------------------------------------
# Generalized Gaussian -- tunable shape exponent (matches fft-cnr's family)
# --------------------------------------------------------------------------
def _genstd_var_factor(p: float) -> float:
    """Variance of exp(-0.5 |z|^p) as a density, in units of the scale s^2.

    For a profile A * exp(-0.5 |(x-c)/s|^p), the second central moment is
    s^2 * 2^(2/p) * Gamma(3/p) / Gamma(1/p).  Equals 1 at p = 2 (Gaussian).
    """
    return float(2.0 ** (2.0 / p) * _gamma(3.0 / p) / _gamma(1.0 / p))


def _gen_gaussian(x: np.ndarray, amp: float, mean: float,
                  scale: float, shape: float) -> np.ndarray:
    """Generalized-Gaussian profile, peak-normalised to ``amp``."""
    z = np.abs((x - mean) / scale)
    return amp * np.exp(-0.5 * z ** shape)


@dataclass
class GeneralizedGaussianModel:
    """Profile ``A exp(-0.5 |(x-c)/s|^p)`` with shape exponent ``p``.

    p = 2 recovers the Gaussian; p < 2 is heavy-tailed (sharp peak, fat
    tails); p > 2 is flat-topped.  This is the profile family fitted by
    ``fft-cnr`` -- the handle for non-Gaussian transport.  The diffusive
    variance is s^2 * factor(p); the fit recovers the scale s with p held
    fixed at the known transport shape.
    """

    shape: float = 2.0          # p
    name: str = "generalized_gaussian"

    @property
    def var_factor(self) -> float:
        return _genstd_var_factor(self.shape)

    def _scale_for_variance(self, variance: np.ndarray) -> np.ndarray:
        return np.sqrt(np.asarray(variance, float) / self.var_factor)

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        x = exp.x
        scales = self._scale_for_variance(exp.true_variance)
        return np.array([
            _gen_gaussian(x, a, exp.mean, s, self.shape)
            for a, s in zip(exp.amplitude, scales)
        ])

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        span = float(x[-1] - x[0])
        s_min = span / x.size
        s_max = 0.45 * span
        s0 = max(min(span / 8.0, s_max), 2 * s_min)
        amp0 = float(np.max(np.abs(y))) or 1.0
        mean0 = float(x[int(np.argmax(y))])

        def model(xx, amp, mean, scale):
            return _gen_gaussian(xx, amp, mean, scale, self.shape)

        try:
            popt, _ = curve_fit(
                model, x, y, p0=[amp0, mean0, s0],
                bounds=([0.0, x[0], s_min], [2 * amp0, x[-1], s_max]),
                maxfev=10000,
            )
            scale = float(popt[2])
            amp = float(popt[0])
            ok = True
        except (RuntimeError, ValueError):
            scale, amp, ok = s0, amp0, False

        at_wall = scale <= 2 * s_min
        variance = scale ** 2 * self.var_factor
        return FrameFit(variance=variance, amplitude=amp,
                        at_wall=bool(at_wall), ok=ok)


# --------------------------------------------------------------------------
# Moment estimator -- the non-truncated alternative ("remove the wall")
# --------------------------------------------------------------------------
@dataclass
class MomentModel:
    """Non-truncated, moment-based width estimator.

    Instead of a bounded curve fit, the variance is the discrete second
    central moment of the (background-subtracted) profile. That estimator is
    linear in the data: it can return a *negative* variance and does not fold
    at zero. Its sampling distribution is therefore approximately symmetric
    and, to leading order, unbiased -- so no bias correction is needed (cf.
    Vestergaard's covariance-based estimator: unbiased *because* it admits
    negative values). The cost is higher variance, since the x^2 weighting
    amplifies tail noise.

    The estimator assumes the profile is centred near x = 0, which the
    simulation guarantees; a production implementation would track the
    centroid. ``noise_free_series`` reuses the Gaussian profile -- the moment
    estimator is the *estimator* under test, applied to ordinary Gaussian
    data.
    """

    name: str = "moment"
    edge_frac: float = 0.1       # fraction of pixels each side used for baseline

    def noise_free_series(self, exp: Experiment) -> np.ndarray:
        return _gaussian_series(exp)

    def fit_variance(self, x: np.ndarray, y: np.ndarray) -> FrameFit:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        n = x.size
        edge = max(1, int(self.edge_frac * n))
        baseline = 0.5 * (y[:edge].mean() + y[-edge:].mean())
        ys = y - baseline                       # may be negative (noise)

        total = float(ys.sum())
        amp = float(np.nanmax(y))
        if not np.isfinite(total) or total <= 0.0:
            # Negative/zero total signal: a noise-dominated frame, unusable.
            return FrameFit(variance=np.nan, amplitude=amp,
                            at_wall=False, ok=False)

        mean = float(np.sum(ys * x) / total)
        variance = float(np.sum(ys * (x - mean) ** 2) / total)
        # No wall: the moment estimator is not bounded below.
        return FrameFit(variance=variance, amplitude=amp,
                        at_wall=False, ok=True)


# Convenience registry for scripts that select a model by name.
MODELS = {
    "gaussian": GaussianModel,
    "voigt": VoigtModel,
    "generalized_gaussian": GeneralizedGaussianModel,
    "moment": MomentModel,
}
