"""End-to-end validation: does the folded-normal correction reduce D bias?

The validation has two stages, deliberately on independent random seeds:

1. **Calibrate** -- run Monte Carlo ensembles across a CNR sweep and build a
   :class:`~weighting_wall.truncation.TruncationModel` mapping CNR to the
   truncation fraction, the MSD-point variance, and the additive variance bias.
2. **Evaluate** -- run fresh ensembles, fit D under each scheme using only the
   per-frame CNR plus the calibrated model, and report ``<D_est> / D_true`` as
   a function of CNR_0.

A scheme that works drives the bias curve toward 1 at low CNR, where OLS and
the standard inverse-variance scheme leave D inflated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fit import fit_D
from .montecarlo import run_ensemble
from .simulate import Experiment
from .truncation import TruncationModel, frame_stats
from .weights import SCHEMES, corrected_variance, msd_and_weights


def sweep_stats(exp: Experiment, cnr0_grid: np.ndarray, n_runs: int = 400,
                seed: int = 1, model=None, noise_model=None) -> list:
    """Per-frame statistics from a Monte Carlo CNR sweep.

    Returns one :class:`~weighting_wall.truncation.FrameStats` per CNR_0.  This
    is the raw material both for the truncation model and for the diagnostic
    panel of the POC figure, so callers can reuse it instead of resampling.
    ``model`` selects the profile model (see :mod:`weighting_wall.models`);
    ``noise_model`` selects the noise law (see :mod:`weighting_wall.noise`,
    ``None`` = white).
    """
    stats = []
    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i,
                           model=model, noise_model=noise_model)
        stats.append(frame_stats(ens))
    return stats


def calibrate(exp: Experiment, cnr0_grid: np.ndarray, n_runs: int = 400,
              seed: int = 1, model=None, noise_model=None) -> TruncationModel:
    """Build a truncation model from a Monte Carlo CNR sweep.

    The calibration uses ``exp.D`` to generate synthetic profile sequences, so
    the lookup is implicitly conditioned on that D. For a D-independent
    calibration the recommended approach is :func:`calibrate_d_free`, which is
    a thin wrapper that calls this function with ``D = 0``.
    """
    return TruncationModel.from_stats(
        sweep_stats(exp, cnr0_grid, n_runs=n_runs, seed=seed, model=model,
                    noise_model=noise_model))


def calibrate_d_free(cnr0_grid: np.ndarray, *, sigma0: float = 1.0,
                     tau: float = 1.0, half_window: float = 6.0,
                     n_pixels: int = 241, n_frames: int = 12,
                     t_max: float = 3.0, amplitude0: float = 1.0,
                     n_runs: int = 400, seed: int = 1,
                     model=None, noise_model=None) -> TruncationModel:
    """Build a truncation model **without reference to the application's D**.

    Generates the calibration profile sequence at ``D = 0``, so every frame's
    true variance is ``sigma_0^2`` and the resulting 1-D ``var_bias(CNR)``
    lookup is keyed on per-frame CNR alone. Application code (the slope fit
    in :func:`weighting_wall.weights.msd_and_weights`) consumes the same
    lookup it always has -- no change to the deployment interface.

    Use this whenever the application's D is unknown (real data), variable
    across measurements, or itself heterogeneous (a mixed-D population). In
    the low-D regime where the truncation bias bites, the per-frame variance
    barely changes from ``sigma_0^2`` over the measurement window, so the
    bias calibrated at ``sigma^2 = sigma_0^2`` is the bias the application
    actually sees. Validated by Monte Carlo across D in [0.005, 0.1]; see
    ``scripts/validate_d_zero.py``.

    All parameters except ``cnr0_grid`` describe the experimental geometry
    (pixel grid, frame count, lifetime); the calibration MUST match the
    geometry of the data it will be applied to.
    """
    cal_exp = Experiment(sigma0=sigma0, D=0.0, tau=tau, amplitude0=amplitude0,
                         half_window=half_window, n_pixels=n_pixels,
                         n_frames=n_frames, t_max=t_max)
    return calibrate(cal_exp, cnr0_grid, n_runs=n_runs, seed=seed, model=model,
                     noise_model=noise_model)


@dataclass
class SchemeResult:
    """Bias curve for one weighting scheme across the CNR_0 sweep."""

    scheme: str
    cnr0: np.ndarray
    mean_bias: np.ndarray     # <D_est> / D_true
    median_bias: np.ndarray   # median(D_est) / D_true
    spread: np.ndarray        # std(D_est / D_true)


@dataclass
class FrameDiagnostic:
    """Per-frame residual bias of a width-fitting correction.

    For ``folded_normal`` and ``mas`` the final D estimate is a deterministic
    linear functional of the corrected per-frame variances (the slope-fit
    weights are deterministic in CNR), so the per-frame residual
    ``mean_corrected - true_variance`` fully determines the bias in D. This
    diagnostic exposes that residual frame by frame, which localises where a
    correction over- or under-shoots -- information the collapsed D ratio
    hides. See ``docs/DERIVATION.md``.

    All variances and standard errors are in units of sigma_0^2.

    Standard errors on the means scale roughly as ``SD(sigma_hat^2)/CNR_frame``
    -- they grow sharply at low per-frame CNR because the fit's noise scales
    as ``1/CNR``. Interpret any reported residual against ``residual_se`` to
    see whether it is statistically resolved.
    """

    scheme: str
    cnr0: float
    cnr: np.ndarray             # per-frame CNR
    times: np.ndarray           # frame times
    true_variance: np.ndarray   # sigma^2_true(t)
    mean_raw: np.ndarray        # E[sigma_hat^2] -- raw fitted, uncorrected
    mean_corrected: np.ndarray  # E[sigma_tilde^2] -- after the scheme's correction
    residual: np.ndarray        # mean_corrected - true_variance
    mean_raw_se: np.ndarray     # standard error of mean_raw (= SD/sqrt(n))
    mean_corrected_se: np.ndarray  # standard error of mean_corrected
    residual_se: np.ndarray     # = mean_corrected_se (true_variance is exact)
    n_runs_used: np.ndarray     # finite-fit count contributing to each frame


def per_frame_diagnostic(exp: Experiment, cnr0: float, model: TruncationModel,
                         scheme: str = "folded_normal", n_runs: int = 2000,
                         seed: int = 30000,
                         profile_model=None) -> FrameDiagnostic:
    """Per-frame residual bias of a correction scheme at a given CNR_0.

    Runs a Monte Carlo ensemble at ``cnr0``, applies the ``scheme``'s
    per-frame correction to every realisation, and reports, frame by frame,
    how well the corrected width ``E[sigma_tilde^2(t)]`` recovers
    ``sigma^2_true(t)``.

    ``scheme`` should be a width-fitting scheme (``folded_normal``, ``mas``;
    or ``ols`` / ``inverse_variance`` to see the uncorrected raw bias). The
    cross-frame and joint-ML estimators do not fit a per-frame width and are
    not covered by this diagnostic.
    """
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed,
                       model=profile_model)
    n_frames = exp.n_frames
    corrected = np.full((ens.n_runs, n_frames), np.nan)
    for r in range(ens.n_runs):
        corrected[r] = corrected_variance(scheme, ens.variance[r],
                                          ens.cnr_frame, model)

    mean_raw = np.full(n_frames, np.nan)
    mean_corr = np.full(n_frames, np.nan)
    mean_raw_se = np.full(n_frames, np.nan)
    mean_corr_se = np.full(n_frames, np.nan)
    n_used = np.zeros(n_frames, dtype=int)
    for f in range(n_frames):
        raw = ens.variance[:, f]
        raw = raw[np.isfinite(raw)]
        if raw.size:
            mean_raw[f] = float(raw.mean())
            if raw.size > 1:
                mean_raw_se[f] = float(raw.std(ddof=1)) / np.sqrt(raw.size)
        cor = corrected[:, f]
        cor = cor[np.isfinite(cor)]
        n_used[f] = cor.size
        if cor.size:
            mean_corr[f] = float(cor.mean())
            if cor.size > 1:
                mean_corr_se[f] = float(cor.std(ddof=1)) / np.sqrt(cor.size)

    true_var = exp.true_variance
    return FrameDiagnostic(scheme=scheme, cnr0=float(cnr0),
                           cnr=ens.cnr_frame, times=exp.times,
                           true_variance=true_var, mean_raw=mean_raw,
                           mean_corrected=mean_corr,
                           residual=mean_corr - true_var,
                           mean_raw_se=mean_raw_se,
                           mean_corrected_se=mean_corr_se,
                           residual_se=mean_corr_se,
                           n_runs_used=n_used)


def evaluate(exp: Experiment, cnr0_grid: np.ndarray, model: TruncationModel,
             n_runs: int = 400, seed: int = 20000,
             profile_model=None, noise_model=None) -> dict[str, SchemeResult]:
    """Fit D under every scheme and return the bias curves.

    ``model`` is the calibrated :class:`TruncationModel`; ``profile_model`` is
    the profile model (Gaussian / Voigt / generalized-Gaussian) and must match
    the one used for calibration. ``noise_model`` selects the noise law (see
    :mod:`weighting_wall.noise`, ``None`` = white); to test robustness to a
    noise-law mismatch, pass a ``model`` calibrated under one law and a
    ``noise_model`` of another.
    """
    times = exp.times
    cnr0_grid = np.asarray(cnr0_grid, float)
    n = cnr0_grid.size
    out = {s: {"mean": np.full(n, np.nan),
               "median": np.full(n, np.nan),
               "spread": np.full(n, np.nan)} for s in SCHEMES}

    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i,
                           model=profile_model, noise_model=noise_model)
        for scheme in SCHEMES:
            ratios = []
            for r in range(ens.n_runs):
                msd, w = msd_and_weights(scheme, ens.variance[r],
                                         ens.cnr_frame, model)
                d = fit_D(times, msd, w)
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
