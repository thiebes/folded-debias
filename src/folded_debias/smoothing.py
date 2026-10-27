"""Penalised B-spline (P-spline) smoothing of Monte Carlo calibration cells.

A raw calibration lookup interpolates between individual Monte Carlo cells, so
every looked-up value carries one cell's sampling error, and neighbouring
points jump by that error. A P-spline (cubic B-splines with a difference
penalty on their coefficients) pools many cells into one smooth curve whose
error is much smaller and fully described by a coefficient covariance, which
the calibration SE of D needs because the smoothed values are correlated.

With a ``knee``, the curve is a constant above the knee and the constant plus a
spline below it, with the spline's value and slope forced to zero at the knee
(the last two clamped basis functions are dropped). The constant is fitted
(``intercept=True``) or is zero, for a curve the caller has already offset by a
known asymptote. The calibration uses this for the ratio of the Monte Carlo bias
to its leading-order Fisher value: the ratio tends to a known constant at high
CNR, and the spline carries only the higher-order departure at low CNR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline


@dataclass(frozen=True)
class PSplineFit:
    """A fitted P-spline: coefficients, their covariance, and the smoothing chosen."""

    knots: np.ndarray
    degree: int
    knee: float | None
    intercept: bool
    coef: np.ndarray
    coef_cov: np.ndarray
    lam: float
    edf: float
    phi: float

    def design(self, u: np.ndarray) -> np.ndarray:
        """Design-matrix rows at ``u``; the fitted curve is ``design(u) @ coef``."""
        return _design(np.atleast_1d(np.asarray(u, float)), self.knots,
                       self.degree, self.knee, self.intercept)

    def evaluate(self, u: np.ndarray) -> np.ndarray:
        return self.design(u) @ self.coef


def _clamped_knots(u_lo: float, u_hi: float, n_segments: int, degree: int) -> np.ndarray:
    inner = np.linspace(u_lo, u_hi, n_segments + 1)
    return np.concatenate([np.full(degree, u_lo), inner, np.full(degree, u_hi)])


def _design(u: np.ndarray, knots: np.ndarray, degree: int,
            knee: float | None, intercept: bool) -> np.ndarray:
    """B-spline basis rows; with a knee, the basis minus its last two (plus an intercept).

    Outside the basis domain the rows are clamped to the end values (no knee) or
    carry only the intercept above the knee.
    """
    lo, hi = knots[degree], knots[-degree - 1]
    basis = BSpline.design_matrix(np.clip(u, lo, hi), knots, degree).toarray()
    if knee is None:
        return basis
    spline = basis[:, :-2]
    spline[u >= knee] = 0.0
    return np.column_stack([np.ones(u.size), spline]) if intercept else spline


def _penalty(n_basis: int, diff_order: int, knee: float | None,
             intercept: bool) -> np.ndarray:
    """Difference penalty on the spline coefficients (an intercept is unpenalised).

    With a knee the two dropped end coefficients are zero, so the penalty still
    sees them: it keeps the curve from bending sharply onto its asymptote.
    """
    diff = np.diff(np.eye(n_basis), n=diff_order, axis=0)
    if knee is None:
        return diff.T @ diff
    kept = diff[:, :-2]
    if not intercept:
        return kept.T @ kept
    penalty = np.zeros((n_basis - 1, n_basis - 1))
    penalty[1:, 1:] = kept.T @ kept
    return penalty


def pspline_fit(u: np.ndarray, y: np.ndarray, se: np.ndarray, *,
                n_segments: int = 12, degree: int = 3, diff_order: int = 2,
                knee: float | None = None, intercept: bool = True,
                lam_grid: np.ndarray | None = None) -> PSplineFit:
    """Fit y(u) with inverse-variance weights 1/se^2, smoothing chosen by GCV.

    The coefficient covariance is phi (X'WX + lam P)^-1, the Bayesian form that
    includes the smoothing bias, with phi = max(1, weighted RSS / residual dof)
    so that cell errors understated by ``se`` widen it rather than narrow it.
    """
    u = np.asarray(u, float)
    y = np.asarray(y, float)
    weight = 1.0 / np.asarray(se, float) ** 2
    if lam_grid is None:
        lam_grid = np.logspace(-4, 6, 41)

    u_hi = float(knee) if knee is not None else float(u.max())
    knots = _clamped_knots(float(u.min()), u_hi, n_segments, degree)
    design = _design(u, knots, degree, knee, intercept)
    penalty = _penalty(n_segments + degree, diff_order, knee, intercept)
    gram = design.T @ (weight[:, None] * design)
    rhs = design.T @ (weight * y)
    n_cells = u.size

    best = None
    for lam in lam_grid:
        system = gram + lam * penalty
        coef = np.linalg.solve(system, rhs)
        edf = float(np.trace(np.linalg.solve(system, gram)))
        if edf >= n_cells:
            continue
        rss = float(np.sum(weight * (y - design @ coef) ** 2))
        gcv = n_cells * rss / (n_cells - edf) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, lam, coef, edf, rss, system)
    if best is None:
        raise ValueError("no smoothing parameter leaves residual degrees of freedom; "
                         "use more cells or fewer segments")
    _, lam, coef, edf, rss, system = best
    phi = max(1.0, rss / (n_cells - edf))
    return PSplineFit(knots=knots, degree=degree, knee=knee, intercept=intercept,
                      coef=coef,
                      coef_cov=phi * np.linalg.inv(system), lam=float(lam),
                      edf=edf, phi=phi)
