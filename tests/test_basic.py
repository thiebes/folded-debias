"""Sanity tests for the weighting-wall POC.

These are smoke tests, not a validation suite -- they keep the moving parts
wired together while the analytical weight form is still being derived.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighting_wall import (Experiment, calibrate, evaluate, frame_stats,
                            run_ensemble, truncation_factor)
from weighting_wall.fit import fit_D, fit_variance
from weighting_wall.simulate import noise_free_series


def test_noise_free_fit_recovers_variance():
    """A clean profile must fit back to its known variance."""
    exp = Experiment(D=0.0)
    profiles = noise_free_series(exp)
    ff = fit_variance(exp.x, profiles[0])
    assert ff.ok
    assert ff.variance == pytest.approx(exp.sigma0 ** 2, rel=1e-3)


def test_noise_free_fit_D_is_exact():
    """With no noise the linear MSD fit returns D exactly."""
    exp = Experiment(D=0.03)
    profiles = noise_free_series(exp)
    var = np.array([fit_variance(exp.x, row).variance for row in profiles])
    msd = var - var[0]
    d = fit_D(exp.times, msd, np.ones_like(msd))
    assert d.ok
    assert d.D == pytest.approx(exp.D, rel=1e-3)


def test_truncation_factor_limits():
    """p=0 -> factor 1 (clean); p=0.5 -> factor 0 (pure noise)."""
    assert truncation_factor(0.0) == pytest.approx(1.0)
    assert truncation_factor(0.5) == pytest.approx(0.0)
    assert 0.0 < truncation_factor(0.25) < 1.0


def test_truncation_fraction_grows_at_low_cnr():
    """The wall bites harder at low CNR: p is larger when CNR_0 is small."""
    exp = Experiment(D=0.02)
    lo = frame_stats(run_ensemble(exp, cnr0=2.0, n_runs=300, seed=0))
    hi = frame_stats(run_ensemble(exp, cnr0=40.0, n_runs=300, seed=0))
    assert np.nanmean(lo.trunc_fraction) > np.nanmean(hi.trunc_fraction)


def test_low_cnr_inflates_D():
    """The known result: at low CNR the mean fitted D is biased high."""
    exp = Experiment(D=0.02)
    ens = run_ensemble(exp, cnr0=3.0, n_runs=400, seed=0)
    slopes = []
    for r in range(ens.n_runs):
        d = fit_D(exp.times, ens.msd[r], np.ones(exp.n_frames))
        if d.ok:
            slopes.append(d.D)
    assert np.mean(slopes) > exp.D  # positive (high) bias


def test_evaluate_runs_and_returns_all_schemes():
    """The end-to-end harness produces a bias curve for every scheme."""
    exp = Experiment(D=0.02)
    grid = np.array([3.0, 10.0, 30.0])
    model = calibrate(exp, grid, n_runs=150, seed=1)
    results = evaluate(exp, grid, model, n_runs=150, seed=5000)
    assert set(results) == {"ols", "inverse_variance", "cnr_aware"}
    for r in results.values():
        assert r.mean_bias.shape == grid.shape
        assert np.all(np.isfinite(r.mean_bias))
