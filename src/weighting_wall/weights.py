"""Weighting schemes for the MSD -> D linear fit.

Three schemes are compared:

* ``ols``               -- ordinary least squares; every frame weighted equally.
* ``inverse_variance``  -- the field-standard scheme, w_i = 1 / Var(MSD_i).
                           It corrects for heteroscedasticity but assumes the
                           MSD sampling distribution is symmetric.
* ``cnr_aware``         -- the proposed scheme.  It keeps the inverse-variance
                           scaling and multiplies in a truncation factor built
                           from the per-frame truncation fraction p(CNR).

Every scheme takes the *per-frame CNR* as input.  In the Monte Carlo validation
the CNR is known exactly; on real data it is measured from each profile with
the ``fft-cnr`` package (see ``weighting_wall.cnr``).

The analytical form of the truncation factor is the open research question of
this project.  ``truncation_factor`` below is a documented first candidate, not
a final result -- see PLAN.md, "Open derivation".
"""
from __future__ import annotations

import numpy as np

from .truncation import TruncationModel

SCHEMES = ("ols", "inverse_variance", "cnr_aware")

# Default exponent of the truncation factor (1 - 2p)^gamma.  Tunable; the
# correct value is part of the derivation still in progress.
DEFAULT_GAMMA = 2.0


def truncation_factor(p: np.ndarray | float, gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    """Reliability factor from the per-frame truncation fraction p.

    Candidate form: ``(1 - 2p) ** gamma``.

    Rationale.  ``p`` is the fraction of the MSD sampling distribution that has
    crossed the D >= 0 wall.  ``1 - 2p`` is the net signed fraction still on the
    correct side: it is 1 for a clean frame (p = 0) and 0 for a frame that is
    pure noise (p = 0.5), where the fitted increment is as likely negative as
    positive.  Raising it to ``gamma`` lets the factor act like an effective
    squared SNR, matching the dimensions of the inverse-variance term.

    This down-weights frames whose contribution to the slope is not merely
    imprecise but systematically pulled upward by the wall -- the leakage that
    inverse-variance weighting alone cannot see.
    """
    p = np.clip(np.asarray(p, float), 0.0, 0.5)
    return np.clip(1.0 - 2.0 * p, 0.0, 1.0) ** gamma


# Optional hard cutoff: frames below this per-frame CNR get zero weight in
# every scheme.  Disabled by default (0.0) so the bias curve shows the full,
# unmasked effect and the weighting -- not a cutoff -- does the work.  TRACE
# recommends a cutoff in practice; expose it but do not lean on it here.
DEFAULT_CNR_FLOOR = 0.0


def make_weights(scheme: str, cnr_frame: np.ndarray, model: TruncationModel,
                 gamma: float = DEFAULT_GAMMA,
                 cnr_floor: float = DEFAULT_CNR_FLOOR) -> np.ndarray:
    """Per-frame weights for one MSD fit.

    Parameters
    ----------
    scheme : one of :data:`SCHEMES`.
    cnr_frame : per-frame CNR (measured or, in Monte Carlo, known).
    model : calibrated :class:`~weighting_wall.truncation.TruncationModel`
        mapping CNR -> (truncation fraction, relative MSD variance).
    gamma : exponent of the truncation factor (``cnr_aware`` only).
    cnr_floor : frames below this per-frame CNR are given zero weight.
    """
    cnr = np.asarray(cnr_frame, float)

    if scheme == "ols":
        w = np.ones_like(cnr)
    elif scheme in ("inverse_variance", "cnr_aware"):
        var = model.rel_variance(cnr)
        tiny = np.nanmax(var) * 1e-12 if np.isfinite(np.nanmax(var)) else 1e-12
        inv_var = 1.0 / np.clip(var, tiny, None)
        if scheme == "inverse_variance":
            w = inv_var
        else:
            w = inv_var * truncation_factor(model.p(cnr), gamma=gamma)
    else:
        raise ValueError(f"unknown scheme {scheme!r}; expected one of {SCHEMES}")

    w = np.asarray(w, float).copy()
    # The t=0 frame defines the MSD origin and carries no slope information;
    # un-measurable low-CNR frames are excluded from every scheme alike.
    w[0] = 0.0
    w[cnr < cnr_floor] = 0.0
    return w
