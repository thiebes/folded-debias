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
only the per-frame subtraction in ``folded_normal``/``mas`` can. They differ
from ``inverse_variance`` *only* by that subtraction, so the comparison
isolates the effect of the bias correction.

Every scheme takes the *per-frame CNR* as input. In the Monte Carlo validation
the CNR is known exactly; on real data it is measured from each profile with
the ``fft-cnr`` package (see ``weighting_wall.cnr``).
"""
from __future__ import annotations

import numpy as np

from .truncation import TruncationModel

SCHEMES = ("ols", "inverse_variance", "folded_normal", "mas")


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
    model : calibrated :class:`~weighting_wall.truncation.TruncationModel`.

    Notes
    -----
    ``folded_normal`` subtracts the calibrated additive variance bias
    ``model.var_bias(CNR)`` from each frame before differencing. ``mas``
    subtracts the same bias through a smooth gate that switches the correction
    off in deep noise (so the corrected variance never crosses zero). ``ols``
    and ``inverse_variance`` leave the variances uncorrected. The t=0 frame
    defines the MSD origin and is given zero weight in every scheme.
    """
    variance_row = np.asarray(variance_row, float)
    cnr = np.asarray(cnr_frame, float)

    # --- bias correction ------------------------------------------------
    if scheme in ("ols", "inverse_variance"):
        corrected = variance_row
    elif scheme == "folded_normal":
        corrected = variance_row - model.var_bias(cnr)
    elif scheme == "mas":
        # Plaszczynski-MAS-style gated subtraction. The gate (1 - exp(-u)),
        # u = sigma_hat^2 / s^2, is ~1 when the frame is well above the noise
        # (full s^2 subtraction, matching folded_normal) and ~0 when it is
        # noise dominated (no subtraction). corrected = s^2 * (u - 1 + e^-u)
        # is >= 0 for all u >= 0 -- the correction can never overshoot.
        s2 = model.var_bias(cnr)
        hi = np.nanmax(s2)
        floor = hi * 1e-6 if np.isfinite(hi) and hi > 0 else 1e-12
        s2 = np.clip(s2, floor, None)
        u = np.clip(variance_row, 0.0, None) / s2
        corrected = variance_row - s2 * (1.0 - np.exp(-u))
    else:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")
    msd = corrected - corrected[0]

    # --- weights --------------------------------------------------------
    if scheme == "ols":
        w = np.ones_like(cnr)
    else:
        var = model.rel_variance(cnr)
        hi = np.nanmax(var)
        tiny = hi * 1e-12 if np.isfinite(hi) and hi > 0 else 1e-12
        w = 1.0 / np.clip(var, tiny, None)

    w = np.asarray(w, float).copy()
    w[0] = 0.0   # the t=0 frame carries no slope information
    return msd, w
