"""Generation of synthetic 1-D profile sequences for Monte Carlo validation.

A population of excited states is modelled as a 1-D Gaussian that broadens by
Fickian diffusion and decays by a single exponential lifetime.  White noise is
added at the level of the raw spatial profile -- the only level at which it can
be added correctly, because the fitting hierarchy reshapes the error
distribution downstream (Thiebes & Grumstrup, JCP 2024).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def gaussian(x: np.ndarray, amplitude: float, mean: float, sigma: float) -> np.ndarray:
    """A 1-D Gaussian profile."""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


@dataclass(frozen=True)
class Experiment:
    """Parameters of a single simulated diffusion measurement.

    Lengths are in arbitrary spatial units; the initial spot width ``sigma0``
    sets the scale.  The nanocrystal-solid regime targeted here is
    ``L_D / FWHM_0`` of order 0.05--0.2 (e.g. a 30--80 nm diffusion length
    against a ~400 nm diffraction-limited spot).
    """

    sigma0: float = 1.0          # initial Gaussian width (sets the length scale)
    D: float = 0.02              # diffusion coefficient (variance units / time)
    tau: float = 1.0             # exponential lifetime
    amplitude0: float = 1.0      # initial peak amplitude
    mean: float = 0.0            # profile centre
    half_window: float = 6.0     # spatial axis spans [-half_window, +half_window] * sigma0
    n_pixels: int = 241          # spatial samples
    n_frames: int = 12           # number of time points
    t_max: float = 3.0           # last time point (in units of tau)

    @property
    def x(self) -> np.ndarray:
        return np.linspace(-self.half_window * self.sigma0,
                            self.half_window * self.sigma0, self.n_pixels)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0.0, self.t_max * self.tau, self.n_frames)

    @property
    def true_variance(self) -> np.ndarray:
        """Ground-truth profile variance at each frame: sigma0^2 + 2 D t."""
        return self.sigma0 ** 2 + 2.0 * self.D * self.times

    @property
    def true_msd(self) -> np.ndarray:
        """Ground-truth mean-squared displacement: 2 D t."""
        return 2.0 * self.D * self.times

    @property
    def amplitude(self) -> np.ndarray:
        """Peak amplitude at each frame (exponential decay only).

        Diffusive spreading also lowers the peak, but for CNR bookkeeping we
        track the decay term explicitly and let the noise floor be constant.
        """
        return self.amplitude0 * np.exp(-self.times / self.tau)


def noise_sigma_for_cnr0(exp: Experiment, cnr0: float) -> float:
    """Noise standard deviation that yields the requested initial CNR.

    CNR_0 is defined as the initial peak amplitude divided by the noise RMS.
    """
    return exp.amplitude0 / cnr0


def cnr_per_frame(exp: Experiment, noise_sigma: float) -> np.ndarray:
    """True per-frame contrast-to-noise ratio.

    Uses the *current* peak height of the broadened, decayed profile so that
    both decay and diffusive spreading lower the contrast -- this is the
    quantity the per-frame weighting must respond to.
    """
    peak = exp.amplitude * (exp.sigma0 / np.sqrt(exp.true_variance))
    return peak / noise_sigma


def noise_free_series(exp: Experiment) -> np.ndarray:
    """Stack of noise-free profiles, shape (n_frames, n_pixels)."""
    x = exp.x
    return np.array([
        gaussian(x, a, exp.mean, np.sqrt(v))
        for a, v in zip(exp.amplitude, exp.true_variance)
    ])


def noisy_series(exp: Experiment, noise_sigma: float,
                 rng: np.random.Generator) -> np.ndarray:
    """One noisy realisation of the full profile sequence."""
    clean = noise_free_series(exp)
    return clean + rng.normal(0.0, noise_sigma, size=clean.shape)
