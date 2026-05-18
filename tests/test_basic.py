"""Sanity tests for the weighting-wall POC.

These are smoke tests, not a validation suite -- they keep the moving parts
wired together.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighting_wall import (Experiment, GaussianModel, GeneralizedGaussianModel,
                            VoigtModel, calibrate, evaluate, frame_stats,
                            run_ensemble)
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


def test_var_bias_is_positive_and_grows_at_low_cnr():
    """The folded-normal prediction: sigma_hat^2 is biased high, more so at
    low CNR (DERIVATION.md s.2)."""
    exp = Experiment(D=0.02)
    lo = frame_stats(run_ensemble(exp, cnr0=4.0, n_runs=300, seed=0))
    hi = frame_stats(run_ensemble(exp, cnr0=60.0, n_runs=300, seed=0))
    assert np.nanmean(lo.var_bias) > 0.0
    assert np.nanmean(lo.var_bias) > np.nanmean(hi.var_bias)


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


@pytest.mark.parametrize("model", [
    GaussianModel(),
    VoigtModel(gamma=0.3),
    GeneralizedGaussianModel(shape=1.5),
    GeneralizedGaussianModel(shape=3.0),
])
def test_models_recover_noise_free_variance(model):
    """Every profile model must fit its own clean profiles back to truth."""
    exp = Experiment(D=0.02, n_frames=6, t_max=2.0)
    clean = model.noise_free_series(exp)
    for f in range(exp.n_frames):
        ff = model.fit_variance(exp.x, clean[f])
        assert ff.ok
        assert ff.variance == pytest.approx(exp.true_variance[f], rel=2e-3)


@pytest.mark.parametrize("model", [
    VoigtModel(gamma=0.3),
    GeneralizedGaussianModel(shape=1.5),
])
def test_models_run_through_montecarlo(model):
    """Voigt and generalized-Gaussian models run through the MC pipeline."""
    exp = Experiment(D=0.02, n_frames=8, t_max=2.0)
    ens = run_ensemble(exp, cnr0=20.0, n_runs=40, seed=0, model=model)
    assert ens.model_name == model.name
    assert ens.variance.shape == (40, exp.n_frames)
    assert np.isfinite(ens.cnr_frame).all()


def test_generalized_gaussian_variance_factor():
    """The p=2 generalized Gaussian must reduce to the plain Gaussian."""
    g2 = GeneralizedGaussianModel(shape=2.0)
    assert g2.var_factor == pytest.approx(1.0, rel=1e-6)


def test_evaluate_runs_and_returns_all_schemes():
    """The end-to-end harness produces a bias curve for every scheme."""
    exp = Experiment(D=0.02)
    grid = np.array([3.0, 10.0, 30.0])
    model = calibrate(exp, grid, n_runs=150, seed=1)
    results = evaluate(exp, grid, model, n_runs=150, seed=5000)
    assert set(results) == {"ols", "inverse_variance", "folded_normal"}
    for r in results.values():
        assert r.mean_bias.shape == grid.shape
        assert np.all(np.isfinite(r.mean_bias))


def test_folded_normal_beats_inverse_variance_at_low_cnr():
    """The derived scheme should leave less D-bias than inverse-variance in
    the low-CNR regime where the truncation bias is large."""
    exp = Experiment(D=0.02)
    grid = np.array([6.0, 10.0, 18.0, 32.0])
    model = calibrate(exp, grid, n_runs=300, seed=1)
    results = evaluate(exp, grid, model, n_runs=300, seed=5000)
    iv = np.nanmean(np.abs(results["inverse_variance"].mean_bias - 1.0))
    fn = np.nanmean(np.abs(results["folded_normal"].mean_bias - 1.0))
    assert fn < iv
