"""End-to-end validation: does CNR-aware weighting reduce the bias in D?

The validation has two stages, deliberately on independent random seeds:

1. **Calibrate** -- run Monte Carlo ensembles across a CNR sweep and build a
   :class:`~weighting_wall.truncation.TruncationModel`, p(CNR) and v(CNR).
2. **Evaluate** -- run fresh ensembles, fit D under each weighting scheme using
   only the per-frame CNR plus the calibrated model, and report the bias
   ``<D_est> / D_true`` as a function of CNR_0.

A scheme that works drives the bias curve toward 1 at low CNR, where the
standard inverse-variance scheme still leaves D inflated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import fit_D
from .montecarlo import run_ensemble
from .simulate import Experiment
from .truncation import TruncationModel, frame_stats
from .weights import SCHEMES, make_weights


def sweep_stats(exp: Experiment, cnr0_grid: np.ndarray, n_runs: int = 400,
                seed: int = 1, model=None) -> list:
    """Per-frame statistics from a Monte Carlo CNR sweep.

    Returns one :class:`~weighting_wall.truncation.FrameStats` per CNR_0.  This
    is the raw material both for the truncation model and for the diagnostic
    panel of the POC figure, so callers can reuse it instead of resampling.
    ``model`` selects the profile model (see :mod:`weighting_wall.models`).
    """
    stats = []
    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i,
                           model=model)
        stats.append(frame_stats(ens))
    return stats


def calibrate(exp: Experiment, cnr0_grid: np.ndarray, n_runs: int = 400,
              seed: int = 1, model=None) -> TruncationModel:
    """Build a truncation model from a Monte Carlo CNR sweep."""
    return TruncationModel.from_stats(
        sweep_stats(exp, cnr0_grid, n_runs=n_runs, seed=seed, model=model))


@dataclass
class SchemeResult:
    """Bias curve for one weighting scheme across the CNR_0 sweep."""

    scheme: str
    cnr0: np.ndarray
    mean_bias: np.ndarray     # <D_est> / D_true
    median_bias: np.ndarray   # median(D_est) / D_true
    spread: np.ndarray        # std(D_est / D_true)


def evaluate(exp: Experiment, cnr0_grid: np.ndarray, model: TruncationModel,
             n_runs: int = 400, gamma: float = 2.0,
             seed: int = 20000, profile_model=None) -> dict[str, SchemeResult]:
    """Fit D under every scheme and return the bias curves.

    ``model`` is the calibrated :class:`TruncationModel`; ``profile_model`` is
    the profile model (Gaussian / Voigt / generalized-Gaussian) and must match
    the one used for calibration.
    """
    times = exp.times
    cnr0_grid = np.asarray(cnr0_grid, float)
    n = cnr0_grid.size
    out = {s: {"mean": np.full(n, np.nan),
               "median": np.full(n, np.nan),
               "spread": np.full(n, np.nan)} for s in SCHEMES}

    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i,
                           model=profile_model)
        for scheme in SCHEMES:
            w = make_weights(scheme, ens.cnr_frame, model, gamma=gamma)
            ratios = []
            for r in range(ens.n_runs):
                d = fit_D(times, ens.msd[r], w)
                if d.ok and np.isfinite(d.D):
                    ratios.append(d.D / exp.D)
            ratios = np.asarray(ratios, float)
            if ratios.size:
                out[scheme]["mean"][i] = ratios.mean()
                out[scheme]["median"][i] = np.median(ratios)
                out[scheme]["spread"][i] = ratios.std()

    return {s: SchemeResult(scheme=s, cnr0=cnr0_grid,
                            mean_bias=out[s]["mean"],
                            median_bias=out[s]["median"],
                            spread=out[s]["spread"])
            for s in SCHEMES}
