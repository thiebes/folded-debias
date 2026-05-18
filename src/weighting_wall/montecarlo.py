"""Monte Carlo orchestration: many noisy realisations of one experiment.

Because true D is known here, the ensemble exposes the bias that real
measurements cannot.  The output is a stack of per-frame fitted variances plus
the MSD curves derived from them, ready for the truncation and weighting
analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import FrameFit, fit_series_variance
from .simulate import Experiment, cnr_per_frame, noise_sigma_for_cnr0, noisy_series


@dataclass
class Ensemble:
    """Results of an ``n_runs`` Monte Carlo sweep of one experiment."""

    exp: Experiment
    cnr0: float
    noise_sigma: float
    cnr_frame: np.ndarray        # true per-frame CNR, shape (n_frames,)
    variance: np.ndarray         # fitted variances, shape (n_runs, n_frames)
    msd: np.ndarray              # MSD curves, shape (n_runs, n_frames)
    at_wall: np.ndarray          # width-pinned flag, shape (n_runs, n_frames)
    ok: np.ndarray               # fit-converged flag, shape (n_runs, n_frames)

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
                 seed: int = 0) -> Ensemble:
    """Run ``n_runs`` independent noisy realisations of ``exp`` at ``cnr0``."""
    noise_sigma = noise_sigma_for_cnr0(exp, cnr0)
    cnr_frame = cnr_per_frame(exp, noise_sigma)
    rng = np.random.default_rng(seed)
    x = exp.x

    variance = np.full((n_runs, exp.n_frames), np.nan)
    at_wall = np.zeros((n_runs, exp.n_frames), dtype=bool)
    ok = np.zeros((n_runs, exp.n_frames), dtype=bool)

    for r in range(n_runs):
        profiles = noisy_series(exp, noise_sigma, rng)
        fits: list[FrameFit] = fit_series_variance(x, profiles)
        for f, ff in enumerate(fits):
            variance[r, f] = ff.variance
            at_wall[r, f] = ff.at_wall
            ok[r, f] = ff.ok

    msd = _msd_from_variance(variance)
    return Ensemble(exp=exp, cnr0=cnr0, noise_sigma=noise_sigma,
                    cnr_frame=cnr_frame, variance=variance, msd=msd,
                    at_wall=at_wall, ok=ok)
