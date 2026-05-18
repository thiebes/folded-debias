"""Monte Carlo orchestration: many noisy realisations of one experiment.

Because true D is known here, the ensemble exposes the bias that real
measurements cannot.  The output is a stack of per-frame fitted variances plus
the MSD curves derived from them, ready for the truncation and weighting
analysis.

The profile model (Gaussian / Voigt / generalized-Gaussian) is pluggable --
see :mod:`weighting_wall.models`.  The default is the Gaussian model, so
existing callers are unaffected.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import GaussianModel
from .simulate import Experiment, noise_sigma_for_cnr0


@dataclass
class Ensemble:
    """Results of an ``n_runs`` Monte Carlo sweep of one experiment."""

    exp: Experiment
    cnr0: float
    noise_sigma: float
    cnr_frame: np.ndarray        # per-frame CNR (peak / noise RMS), shape (n_frames,)
    variance: np.ndarray         # fitted variances, shape (n_runs, n_frames)
    msd: np.ndarray              # MSD curves, shape (n_runs, n_frames)
    at_wall: np.ndarray          # width-pinned flag, shape (n_runs, n_frames)
    ok: np.ndarray               # fit-converged flag, shape (n_runs, n_frames)
    model_name: str = "gaussian"

    @property
    def true_D(self) -> float:
        return self.exp.D

    @property
    def n_runs(self) -> int:
        return self.variance.shape[0]


def _msd_from_variance(variance: np.ndarray) -> np.ndarray:
    """MSD(t) = sigma^2(t) - sigma^2(0), using the fitted t=0 frame.

    A real measurement never knows the true initial width; it subtracts the
    fitted one.  Doing the same here keeps the simulated bias honest.
    """
    return variance - variance[:, [0]]


def run_ensemble(exp: Experiment, cnr0: float, n_runs: int = 600,
                 seed: int = 0, model=None) -> Ensemble:
    """Run ``n_runs`` independent noisy realisations of ``exp`` at ``cnr0``.

    ``model`` is a profile model from :mod:`weighting_wall.models`
    (default :class:`~weighting_wall.models.GaussianModel`).  The per-frame CNR
    is taken from the actual noise-free profile peak, so it is correct for any
    profile shape.
    """
    if model is None:
        model = GaussianModel()
    noise_sigma = noise_sigma_for_cnr0(exp, cnr0)
    x = exp.x

    clean = model.noise_free_series(exp)            # (n_frames, n_pixels)
    cnr_frame = clean.max(axis=1) / noise_sigma     # peak / noise RMS

    rng = np.random.default_rng(seed)
    variance = np.full((n_runs, exp.n_frames), np.nan)
    at_wall = np.zeros((n_runs, exp.n_frames), dtype=bool)
    ok = np.zeros((n_runs, exp.n_frames), dtype=bool)

    for r in range(n_runs):
        noisy = clean + rng.normal(0.0, noise_sigma, size=clean.shape)
        for f in range(exp.n_frames):
            ff = model.fit_variance(x, noisy[f])
            variance[r, f] = ff.variance
            at_wall[r, f] = ff.at_wall
            ok[r, f] = ff.ok

    msd = _msd_from_variance(variance)
    return Ensemble(exp=exp, cnr0=cnr0, noise_sigma=noise_sigma,
                    cnr_frame=cnr_frame, variance=variance, msd=msd,
                    at_wall=at_wall, ok=ok, model_name=model.name)
