"""Schemes for the MSD -> D linear fit.

Three schemes are compared:

* ``ols``               -- ordinary least squares; every frame weighted equally.
* ``inverse_variance``  -- the field-standard scheme, w_i = 1 / Var(MSD_i).
                           It corrects for the unequal noise across frames but
                           leaves the boundary-induced bias intact.
* ``folded_normal``     -- the derived scheme. It subtracts the additive
                           variance bias from each frame before forming the
                           MSD, then weights by inverse variance.

The folded-normal model (see ``docs/DERIVATION.md``) shows that the fitted
variance carries an additive upward bias equal to the estimation-noise
variance, and that this bias -- not extra scatter -- is what inflates D. A
reweighting of the slope fit cannot remove a mean bias present in every point;
only the per-frame subtraction in ``folded_normal`` can. ``folded_normal``
therefore differs from ``inverse_variance`` *only* by that subtraction, so
comparing the two isolates the effect of the bias correction.

Every scheme takes the *per-frame CNR* as input. In the Monte Carlo validation
the CNR is known exactly; on real data it is measured from each profile with
the ``fft-cnr`` package (see ``weighting_wall.cnr``).
"""
from __future__ import annotations

import numpy as np

from .truncation import TruncationModel

SCHEMES = ("ols", "inverse_variance", "folded_normal")


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
    ``model.var_bias(CNR)`` from each frame before differencing -- the
    folded-normal correction. ``ols`` and ``inverse_variance`` leave the
    variances uncorrected. The t=0 frame defines the MSD origin and is given
    zero weight in every scheme.
    """
    variance_row = np.asarray(variance_row, float)
    cnr = np.asarray(cnr_frame, float)

    # --- bias correction (folded_normal only) ---------------------------
    if scheme == "folded_normal":
        corrected = variance_row - model.var_bias(cnr)
    elif scheme in ("ols", "inverse_variance"):
        corrected = variance_row
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
