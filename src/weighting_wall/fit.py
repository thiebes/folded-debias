"""Per-frame width estimation and the MSD -> D linear fit.

This is the fitting hierarchy that the weighting scheme acts on:

1. Each spatial profile is fit with a Gaussian to estimate its variance.
2. The MSD curve, MSD(t) = sigma^2(t) - sigma^2(0), is fit linearly; the
   slope is 2 D.

The variance parameter of a Gaussian fit is bounded below by zero but
unbounded above.  At low CNR the sampling distribution of the fitted variance
is therefore **left-truncated and right-skewed**: noise can throw a fit to an
arbitrarily broad profile, but not below the wall at zero.  That asymmetry,
carried forward through the MSD slope, is what inflates D.

The fitting here mirrors the validated DICE engine (Thiebes & Grumstrup, JCP
2024): the variance is fit directly with a wide lower-to-upper bound, and a
fit that fails to converge falls back to a fixed broad guess rather than being
discarded -- discarding the failures would quietly delete the upper tail that
carries the bias.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


def _gaussian_var(x: np.ndarray, mu: float, var: float, amp: float) -> np.ndarray:
    """Gaussian parameterised by its variance (so the fit estimates sigma^2)."""
    return amp * np.exp(-0.5 * (x - mu) ** 2 / var)


@dataclass(frozen=True)
class FrameFit:
    variance: float       # fitted Gaussian variance, sigma_hat^2
    amplitude: float      # fitted peak amplitude
    at_wall: bool         # True if the variance was pinned near the lower bound
    ok: bool              # True if the optimiser converged


def fit_variance(x: np.ndarray, y: np.ndarray) -> FrameFit:
    """Fit a single noisy profile with a Gaussian; return its variance.

    The variance is bounded to ``[~1 pixel^2, span^2]``.  A failed fit falls
    back to the broad initial guess (the DICE convention) so that the heavy
    upper tail of the variance distribution is preserved, not silently pruned.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    span = float(x[-1] - x[0])

    var_min = (span / n) ** 2          # ~ one pixel squared
    var_max = span ** 2                 # an essentially flat profile
    var0 = (span / 4.0) ** 2            # broad initial guess / fallback

    # Seed the centre from the peak within the central 60% of the window.
    margin = 0.3 * span
    central = (x >= x[0] + margin) & (x <= x[-1] - margin)
    idx = np.where(central)[0]
    if idx.size:
        mu0 = float(x[idx][int(np.argmax(np.abs(y[idx])))])
    else:
        mu0 = float(x[int(np.argmax(np.abs(y)))])
    amp_max = float(np.max(np.abs(y))) or 1.0
    amp0 = amp_max

    p0 = [mu0, var0, amp0]
    lower = [x[0] + margin, var_min, 0.0]
    upper = [x[-1] - margin, var_max, 2.0 * amp_max]
    try:
        popt, _ = curve_fit(_gaussian_var, x, y, p0=p0,
                            bounds=(lower, upper), maxfev=5000)
        var, amp = float(popt[1]), float(popt[2])
        ok = True
    except (RuntimeError, ValueError):
        var, amp, ok = var0, amp0, False

    at_wall = var <= 2.0 * var_min
    return FrameFit(variance=var, amplitude=amp, at_wall=bool(at_wall), ok=ok)


def fit_series_variance(x: np.ndarray, profiles: np.ndarray) -> list[FrameFit]:
    """Fit every frame in a profile sequence."""
    return [fit_variance(x, row) for row in profiles]


@dataclass(frozen=True)
class DFit:
    D: float              # estimated diffusion coefficient (slope / 2)
    intercept: float
    ok: bool


def fit_D(times: np.ndarray, msd: np.ndarray, weights: np.ndarray) -> DFit:
    """Weighted linear fit of MSD(t); the slope is 2 D.

    ``weights`` are the per-point fit weights (w_i, proportional to 1/variance
    in the standard scheme).  Frames with zero or non-finite weight are
    dropped.  This is the single point where every weighting scheme enters.
    """
    t = np.asarray(times, float)
    m = np.asarray(msd, float)
    w = np.asarray(weights, float)

    good = np.isfinite(t) & np.isfinite(m) & np.isfinite(w) & (w > 0)
    if good.sum() < 2:
        return DFit(D=np.nan, intercept=np.nan, ok=False)

    t, m, w = t[good], m[good], w[good]
    # Weighted least squares for a straight line m = a + b t.
    sw = w.sum()
    swt = (w * t).sum()
    swm = (w * m).sum()
    swtt = (w * t * t).sum()
    swtm = (w * t * m).sum()
    denom = sw * swtt - swt ** 2
    if denom == 0:
        return DFit(D=np.nan, intercept=np.nan, ok=False)
    slope = (sw * swtm - swt * swm) / denom
    intercept = (swm - slope * swt) / sw
    return DFit(D=float(slope / 2.0), intercept=float(intercept), ok=True)
