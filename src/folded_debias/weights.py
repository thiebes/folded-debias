"""Schemes for the MSD -> D linear fit.

Four schemes are compared:

* ``ols``               -- ordinary least squares; every frame weighted equally.
* ``inverse_variance``  -- the field-standard scheme, w_i = 1 / Var(MSD_i).
                           It corrects for the unequal noise across frames but
                           leaves the boundary-induced bias intact.
* ``folded_normal``     -- the derived scheme. It subtracts the additive
                           variance bias s^2 from each frame before forming the
                           MSD, then weights by inverse variance.
* ``mas``               -- a Plaszczynski-MAS-style variant of ``folded_normal``.
                           The subtraction is smoothly gated so it vanishes,
                           rather than overshooting, when a frame is noise
                           dominated -- it cannot drive the corrected variance
                           below zero. This removes the deep-regime
                           overcorrection ``folded_normal`` can show.

The folded-normal model (see ``docs/DERIVATION.md``) shows that the fitted
variance carries an additive upward bias equal to the estimation-noise
variance, and that this bias -- not extra scatter -- is what inflates D. A
reweighting of the slope fit cannot remove a mean bias present in every point;
only the per-frame subtraction in ``folded_normal`` / ``mas`` can. They differ
from ``inverse_variance`` *only* by that subtraction, so the comparison
isolates the effect of the bias correction.

Every scheme takes the *per-frame CNR* as input. In the Monte Carlo validation
the CNR is known exactly; on real data it is measured from each profile with
the ``fft-cnr`` package (see :mod:`folded_debias.cnr`).

The ``model`` (a :class:`~folded_debias.truncation.TruncationModel`) can be
either a legacy 1-D, D-aware calibration or the 2-D, D-free one from
:func:`~folded_debias.calibration.calibrate_cnr_sigma2`. With the joint
(CNR, sigma^2) calibration the
per-frame raw fitted variance is also passed into the lookup, so the
correction is width-aware and carries no reference to D.
"""
from __future__ import annotations

import numpy as np

from .truncation import TruncationModel

SCHEMES = ("ols", "inverse_variance", "folded_normal", "mas")


def corrected_variance(scheme: str, variance_row: np.ndarray,
                       cnr_frame: np.ndarray,
                       model: TruncationModel) -> np.ndarray:
    """Apply a scheme's per-frame bias correction to the fitted variances.

    Returns the corrected per-frame variances sigma_tilde^2(t). For ``ols``
    and ``inverse_variance`` this is the input unchanged -- those schemes do
    not correct the points, only the weights. For ``folded_normal`` it is the
    hard subtraction ``sigma_hat^2 - var_bias(CNR)``; for ``mas`` the smoothly
    gated subtraction (see module docstring).

    This is the single place the bias correction is applied.
    :func:`msd_and_weights` calls it to form the MSD, and
    :func:`folded_debias.validate.per_frame_diagnostic` calls it to report
    the per-frame residual bias. Because the slope fit is linear in these
    corrected variances and the weights are deterministic in CNR, the
    per-frame residual ``E[sigma_tilde^2] - sigma^2_true`` fully determines
    the bias in the final D (see ``docs/DERIVATION.md``).
    """
    variance_row = np.asarray(variance_row, float)
    cnr = np.asarray(cnr_frame, float)
    if scheme in ("ols", "inverse_variance"):
        return variance_row
    if scheme == "folded_normal":
        return variance_row - model.var_bias(cnr, variance_row)
    if scheme == "mas":
        # Plaszczynski-MAS-style gated subtraction. The gate (1 - exp(-u)),
        # u = sigma_hat^2 / s^2, is ~1 when the frame is well above the noise
        # (full s^2 subtraction, matching folded_normal) and ~0 when it is
        # noise dominated (no subtraction). corrected = s^2 * (u - 1 + e^-u)
        # is >= 0 for all u >= 0 -- the correction can never overshoot.
        s2 = model.var_bias(cnr, variance_row)
        hi = np.nanmax(s2)
        floor = hi * 1e-6 if np.isfinite(hi) and hi > 0 else 1e-12
        s2 = np.clip(s2, floor, None)
        u = np.clip(variance_row, 0.0, None) / s2
        return variance_row - s2 * (1.0 - np.exp(-u))
    raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")


def msd_and_weights(scheme: str, variance_row: np.ndarray,
                    cnr_frame: np.ndarray, model: TruncationModel):
    """Return ``(msd, weights)`` for one realisation's MSD-vs-t fit.

    Parameters
    ----------
    scheme : one of :data:`SCHEMES`.
    variance_row : array, shape (n_frames,)
        The fitted variances sigma_hat^2(t) for a single realisation.
    cnr_frame : array, shape (n_frames,)
        Per-frame CNR (known in Monte Carlo; measured with ``fft-cnr`` on real
        data).
    model : calibrated :class:`~folded_debias.truncation.TruncationModel`.

    Notes
    -----
    ``folded_normal`` subtracts the calibrated additive variance bias
    ``model.var_bias(CNR, sigma_hat^2)`` from each frame before differencing.
    ``mas`` subtracts the same bias through a smooth gate that switches the
    correction off in deep noise (so the corrected variance never crosses
    zero). ``ols`` and ``inverse_variance`` leave the variances uncorrected.
    The t=0 frame defines the MSD origin and is given zero weight in every
    scheme.
    """
    variance_row = np.asarray(variance_row, float)
    cnr = np.asarray(cnr_frame, float)

    # --- bias correction ------------------------------------------------
    corrected = corrected_variance(scheme, variance_row, cnr, model)
    msd = corrected - corrected[0]

    # --- weights --------------------------------------------------------
    # The weight is 1 / Var(MSD_i). With the legacy 1-D calibration,
    # rel_variance returns the MSD-point variance directly (empirically
    # measured). With the 2-D calibration, rel_variance returns the per-frame
    # Var(sigma_hat^2); the MSD-point variance is then
    # Var(sigma_hat_t^2) + Var(sigma_hat_0^2) (the two fits are independent).
    if scheme == "ols":
        w = np.ones_like(cnr)
    else:
        rv = model.rel_variance(cnr, variance_row)
        if model.is_2d:
            rv = rv + rv[0]
        hi = np.nanmax(rv)
        tiny = hi * 1e-12 if np.isfinite(hi) and hi > 0 else 1e-12
        w = 1.0 / np.clip(rv, tiny, None)

    w = np.asarray(w, float).copy()
    w[0] = 0.0   # the t=0 frame carries no slope information
    return msd, w


def calibration_D_se(times: np.ndarray, msd: np.ndarray, weights: np.ndarray,
                     cnr_frame: np.ndarray, model: TruncationModel,
                     variance_row: np.ndarray | None = None) -> float:
    """SE of a corrected D from the calibration's own Monte Carlo error.

    :func:`~folded_debias.fit.fit_D` is a weighted straight-line fit, so its
    slope is a fixed linear combination of the MSD points, ``b = sum c_i m_i``
    with ``c_i = w_i (S_w t_i - S_wt) / (S_w S_wtt - S_wt^2)``. A calibration
    error ``e_i`` in frame i's subtracted bias moves ``m_i`` by
    ``-(e_i - e_0)``. Since ``sum c_i = 0`` over the fitted frames, the frame-0
    term cancels (the intercept absorbs it), and with independent calibration
    cells

        Var_cal(D) = (1/4) sum_i c_i^2 s_i^2,   s_i = model.var_bias_se(CNR_i).

    Pass the same ``times``, ``msd``, and ``weights`` given to ``fit_D`` so the
    frame set matches, and ``variance_row`` (the raw fitted variances) when the
    model is a joint (CNR, sigma^2) calibration. Combine with the fit's own SE
    as ``sqrt(D_se**2 + calibration_D_se**2)``. This is exact for
    ``folded_normal``; for ``mas`` it ignores the gate's dependence on the bias
    and is approximate. The weights' own dependence on the calibration is
    second order and not included. Sixteen independent calibrations on the
    fft-cnr paper's grid confirm the prediction (observed / predicted variance
    0.98 over 18 draws; ``experiments/dfree-calibration/calibration_se_check.py``);
    the weight term is below 0.06 of it from CNR_0 5 up and about 0.3 of it at
    CNR_0 3. Returns NaN when the model has no SE.
    """
    t = np.asarray(times, float)
    m = np.asarray(msd, float)
    w = np.asarray(weights, float)
    cnr = np.asarray(cnr_frame, float)
    good = np.isfinite(t) & np.isfinite(m) & np.isfinite(w) & (w > 0)
    if good.sum() < 2:
        return np.nan
    tg, wg = t[good], w[good]
    sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
    denom = sw * swtt - swt ** 2
    if denom == 0:
        return np.nan
    c = wg * (sw * tg - swt) / denom
    var_row = None if variance_row is None else np.asarray(variance_row, float)[good]
    s = np.asarray(model.var_bias_se(cnr[good], var_row), float)
    if not np.all(np.isfinite(s)):
        return np.nan
    return float(np.sqrt(np.sum(c ** 2 * s ** 2)) / 2.0)
