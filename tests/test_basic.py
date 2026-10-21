"""Sanity tests for the folded-debias POC.

These are smoke tests, not a validation suite -- they keep the moving parts
wired together.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from folded_debias import (AnomalousExperiment, Experiment, GaussianModel,
                            GeneralizedGaussianModel, MomentModel,
                            MultiExpExperiment, UnboundedGaussianModel,
                            VoigtModel, calibrate, evaluate,
                            frame_stats, model_from_spec, run_ensemble,
                            spec_tag)
from folded_debias.fit import fit_D, fit_variance
from folded_debias.simulate import noise_free_series


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


def test_fit_D_covariance_se():
    """fit_D reports the weighted-LS covariance SE of D: it matches the closed
    form Var(slope)=Sw/(Sw*Swtt-Swt^2) and shrinks as 1/sqrt(precision)."""
    t = np.array([0.0, 1.0, 2.0, 3.0])
    msd = np.array([0.0, 0.04, 0.08, 0.12])   # slope 0.04 -> D = 0.02
    w = np.array([0.0, 1.0, 1.0, 1.0])        # t=0 carries no slope weight
    d = fit_D(t, msd, w)
    assert d.ok and np.isfinite(d.D_se) and d.D_se > 0
    good = w > 0
    sw = w[good].sum(); swt = (w[good] * t[good]).sum()
    swtt = (w[good] * t[good] ** 2).sum()
    assert d.D_se == pytest.approx(np.sqrt(sw / (sw * swtt - swt ** 2)) / 2,
                                   rel=1e-9)
    # 10x more precise data -> SE shrinks by sqrt(10).
    assert fit_D(t, msd, 10.0 * w).D_se == pytest.approx(d.D_se / np.sqrt(10),
                                                         rel=1e-6)


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
    MomentModel(),
])
def test_models_recover_noise_free_variance(model):
    """Every estimator must recover the known variance from clean profiles."""
    exp = Experiment(D=0.02, n_frames=6, t_max=2.0)
    clean = model.noise_free_series(exp)
    for f in range(exp.n_frames):
        ff = model.fit_variance(exp.x, clean[f])
        assert ff.ok
        assert ff.variance == pytest.approx(exp.true_variance[f], rel=5e-3)


@pytest.mark.parametrize("model", [
    VoigtModel(gamma=0.3),
    GeneralizedGaussianModel(shape=1.5),
    MomentModel(),
])
def test_models_run_through_montecarlo(model):
    """Voigt, generalized-Gaussian and moment models run through the pipeline."""
    exp = Experiment(D=0.02, n_frames=8, t_max=2.0)
    ens = run_ensemble(exp, cnr0=20.0, n_runs=40, seed=0, model=model)
    assert ens.model_name == model.name
    assert ens.variance.shape == (40, exp.n_frames)
    assert np.isfinite(ens.cnr_frame).all()


def test_moment_estimator_can_go_negative():
    """The non-truncated moment estimator must admit negative variances at
    low CNR -- that is what makes it un-folded and (to leading order)
    unbiased."""
    exp = Experiment(D=0.02, n_frames=8, t_max=2.0)
    ens = run_ensemble(exp, cnr0=4.0, n_runs=300, seed=0, model=MomentModel())
    finite = ens.variance[np.isfinite(ens.variance)]
    assert np.any(finite < 0.0)


def test_unbounded_matches_bounded_at_high_cnr():
    """With the wall removed, the fit must be identical to the bounded one when
    the wall does not bind -- i.e. at healthy CNR, where the truncation bias and
    its correction are moot."""
    exp = Experiment(D=0.0)
    clean = noise_free_series(exp)
    rng = np.random.default_rng(0)
    noisy = clean[0] + rng.normal(0.0, exp.amplitude0 / 30.0, exp.x.size)
    bnd = fit_variance(exp.x, noisy)
    unb = UnboundedGaussianModel().fit_variance(exp.x, noisy)
    assert unb.variance == pytest.approx(bnd.variance, rel=1e-3)
    assert not unb.at_wall   # the unbounded fit has no wall to pin against


def test_unbounded_recovers_noise_free_variance():
    """The unbounded fit must still recover the known variance from clean
    profiles -- removing the wall must not break the high-CNR estimator."""
    exp = Experiment(D=0.02, n_frames=6, t_max=2.0)
    model = UnboundedGaussianModel()
    clean = model.noise_free_series(exp)
    for f in range(exp.n_frames):
        ff = model.fit_variance(exp.x, clean[f])
        assert ff.ok
        assert ff.variance == pytest.approx(exp.true_variance[f], rel=5e-3)


def test_unbounded_admits_negative_variance_in_deep_noise():
    """The precision-fitted unbounded estimator must be *able* to return a
    negative variance on peakless data -- the property a variance-parameterised
    fit cannot have (sigma^2 = 0 is an uncrossable singularity). Negatives are
    rare even here (a nonlinear fit favours a positive noise bump), so this
    needs enough pure-noise trials to be a reliable smoke test, not a one-shot.
    """
    model = UnboundedGaussianModel()
    x = Experiment().x
    rng = np.random.default_rng(3)
    saw_negative = False
    for _ in range(400):
        ff = model.fit_variance(x, rng.normal(0.0, 1.0, x.size))
        if np.isfinite(ff.variance) and ff.variance < 0.0:
            saw_negative = True
            break
    assert saw_negative


def test_anomalous_reduces_to_fickian_at_alpha_one():
    """alpha = 1 must reproduce the Fickian Experiment exactly -- the
    consistency check the subdiffusive validation leans on."""
    fickian = Experiment(D=0.03)
    anomalous = AnomalousExperiment(D=0.03, alpha=1.0)
    assert np.allclose(anomalous.true_variance, fickian.true_variance)
    assert np.allclose(anomalous.true_msd, fickian.true_msd)


def test_anomalous_curvature_sign_matches_exponent():
    """MSD(t) = 2 D t^alpha is concave for alpha < 1 and convex for alpha > 1.
    That curvature is the disorder signature the correction must preserve."""
    t = AnomalousExperiment(alpha=0.6).times
    sub = AnomalousExperiment(D=0.02, alpha=0.6).true_msd
    sup = AnomalousExperiment(D=0.02, alpha=1.4).true_msd
    assert np.all(np.diff(sub[1:], 2) < 0)   # concave (drop the t=0 origin)
    assert np.all(np.diff(sup, 2) > 0)       # convex


def test_anomalous_noise_free_fit_recovers_imposed_variance():
    """The bed must actually produce profiles whose fitted width follows the
    imposed sigma^2(t) = sigma0^2 + 2 D t^alpha -- the premise of scoring the
    recovered alpha against truth."""
    exp = AnomalousExperiment(D=0.05, alpha=0.7, n_frames=8, t_max=3.0)
    clean = noise_free_series(exp)
    var = np.array([fit_variance(exp.x, row).variance for row in clean])
    assert np.allclose(var, exp.true_variance, rtol=5e-3)


def test_multiexp_single_component_reduces_to_single_exponential():
    """One component must reproduce the inherited single-exponential decay."""
    single = Experiment(tau=2.0)
    multi = MultiExpExperiment(tau=2.0, fractions=(1.0,), taus=(2.0,))
    assert np.allclose(multi.amplitude, single.amplitude)


def test_multiexp_normalised_to_initial_amplitude():
    """The summed decay is normalised so amplitude(0) = amplitude0, keeping the
    meaning of CNR_0 fixed across decay laws."""
    multi = MultiExpExperiment(amplitude0=1.0, fractions=(0.5, 0.5),
                               taus=(0.3, 3.0))
    assert multi.amplitude[0] == pytest.approx(1.0)
    assert np.all(np.diff(multi.amplitude) < 0)   # monotone decay


def test_multiexp_rejects_mismatched_components():
    """fractions and taus must pair up; a mismatch is a silent-bug risk."""
    with pytest.raises(ValueError):
        MultiExpExperiment(fractions=(0.5, 0.5), taus=(1.0,))


def test_generalized_gaussian_variance_factor():
    """The p=2 generalized Gaussian must reduce to the plain Gaussian."""
    g2 = GeneralizedGaussianModel(shape=2.0)
    assert g2.var_factor == pytest.approx(1.0, rel=1e-6)


def test_model_from_spec_sets_shape():
    """A shape in the spec must reach the model; the bare name is p = 2.

    The bare ``generalized_gaussian`` is a Gaussian profile, so a validation
    that means to test a non-Gaussian shape must carry the exponent through
    the spec string that worker processes rebuild the model from.
    """
    assert model_from_spec("generalized_gaussian").shape == 2.0
    heavy = model_from_spec("generalized_gaussian:1.5")
    assert isinstance(heavy, GeneralizedGaussianModel)
    assert heavy.shape == 1.5
    assert isinstance(model_from_spec("voigt"), VoigtModel)
    assert spec_tag("generalized_gaussian:4") == "generalized_gaussian_p4"
    assert spec_tag("gaussian") == "gaussian"
    with pytest.raises(ValueError):
        model_from_spec("voigt:1.5")
    with pytest.raises(ValueError):
        model_from_spec("lorentzian")


def test_evaluate_runs_and_returns_all_schemes():
    """The end-to-end harness produces a bias curve for every scheme."""
    exp = Experiment(D=0.02)
    grid = np.array([3.0, 10.0, 30.0])
    model = calibrate(exp, grid, n_runs=150, seed=1)
    results = evaluate(exp, grid, model, n_runs=150, seed=5000)
    assert set(results) == {"ols", "inverse_variance", "folded_normal", "mas"}
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


def test_interp_se_matches_hand_calculation():
    """The 1-D lookup SE uses the same interpolation weights as the lookup."""
    from folded_debias.truncation import _interp_se
    xp = np.array([1.0, 2.0, 4.0])
    sp = np.array([0.3, 0.4, 0.1])
    # x = 1.5 sits halfway between points 0 and 1.
    assert _interp_se(1.5, xp, sp) == pytest.approx(np.sqrt(0.25 * 0.09 + 0.25 * 0.16))
    # Outside the grid the edge point's SE applies (np.interp clamps).
    assert _interp_se(10.0, xp, sp) == pytest.approx(0.1)


def test_calibration_se_fields_populated_and_absent():
    """frame_stats -> from_stats carries var_bias SE; a bare model returns NaN."""
    from folded_debias import TruncationModel, calibrate_d_free
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    assert trunc.varbias_se_grid is not None
    assert trunc.varbias_se_grid.shape == trunc.varbias_grid.shape
    assert np.all(trunc.varbias_se_grid[np.isfinite(trunc.varbias_se_grid)] >= 0)
    bare = TruncationModel(cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
                           relvar_grid=trunc.relvar_grid,
                           varbias_grid=trunc.varbias_grid)
    assert np.isnan(bare.var_bias_se(3.0))


def test_frame_variance_weights_do_not_spike_at_calibration_cnr0():
    """The 1-D weights use per-frame Var(sigma_hat^2), not the MSD-point lookup.

    The MSD-point lookup has a 0 at every calibration CNR_0 (MSD(0) = 0 by
    construction), so a frame next to one drew a huge weight. The per-frame
    grid has no zeros, and the weights it gives grow smoothly with CNR.
    """
    from dataclasses import replace
    from folded_debias import calibrate_d_free, msd_and_weights
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    assert trunc.framevar_grid is not None
    assert trunc.framevar_grid.shape == trunc.cnr_grid.shape
    assert np.all(trunc.framevar_grid > 0)
    assert np.any(trunc.relvar_grid == 0)
    # Frames of a CNR_0 = 20 measurement; one sits next to the CNR_0 = 5 point.
    cnr = 20.0 * np.exp(-Experiment().times)
    k = np.argmin(np.abs(cnr - 5.0))
    cnr[k] = 5.0 * 1.001
    variance_row = np.ones_like(cnr)
    _, w_new = msd_and_weights("folded_normal", variance_row, cnr, trunc)
    _, w_old = msd_and_weights("folded_normal", variance_row, cnr,
                               replace(trunc, framevar_grid=None))
    neighbours = w_new[[k - 1, k + 1]]
    assert neighbours.min() < w_new[k] < neighbours.max()
    assert w_old[k] > 3 * w_old[[k - 1, k + 1]].max()


def test_model_without_frame_variance_keeps_msd_point_weights():
    """A model saved before framevar_grid existed weights by rel_variance."""
    from folded_debias import TruncationModel, calibrate_d_free, msd_and_weights
    trunc = calibrate_d_free(np.array([5.0, 20.0]), n_runs=60, seed=3)
    bare = TruncationModel(cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
                           relvar_grid=trunc.relvar_grid,
                           varbias_grid=trunc.varbias_grid)
    assert bare.frame_variance(3.0) is None
    cnr = 12.0 * np.exp(-Experiment().times)
    _, w = msd_and_weights("inverse_variance", np.ones_like(cnr), cnr, bare)
    expected = 1.0 / bare.rel_variance(cnr)
    assert w[0] == 0.0
    assert np.allclose(w[1:], expected[1:])


def test_calibration_D_se_predicts_draw_to_draw_scatter():
    """The propagated calibration SE matches the scatter of corrected D across
    independent calibration draws applied to one fixed evaluation set.

    The calibration error is common to every run evaluated with that
    calibration, so averaging runs does not remove it; the spread of the
    run-averaged D across draws isolates it.
    """
    from folded_debias import calibrate_d_free, calibration_D_se, msd_and_weights
    exp = Experiment(D=0.02)
    ens = run_ensemble(exp, 13.0, n_runs=60, seed=777)
    means, predicted = [], []
    for k in range(8):
        trunc = calibrate_d_free(np.array([6.0, 13.0, 30.0]), n_runs=80, seed=100 + k)
        ds, ses = [], []
        for r in range(ens.n_runs):
            msd, w = msd_and_weights("folded_normal", ens.variance[r],
                                     ens.cnr_frame, trunc)
            d = fit_D(exp.times, msd, w)
            if d.ok:
                ds.append(d.D)
                ses.append(calibration_D_se(exp.times, msd, w, ens.cnr_frame, trunc))
        means.append(np.mean(ds))
        predicted.append(np.nanmean(ses))
    observed = np.std(means, ddof=1)
    ratio = observed / np.mean(predicted)
    assert 0.5 < ratio < 2.0, (observed, np.mean(predicted))
