"""Characterising the truncation: the wall at zero.

For each frame, the Monte Carlo ensemble gives the sampling distribution of the
fitted MSD increment.  Three quantities summarise how the wall at zero distorts
that distribution:

* ``trunc_fraction`` -- the fraction of realisations with MSD <= 0, i.e. the
  probability mass that has crossed to the wrong side of the D >= 0 wall.
* ``rel_variance``   -- the variance of the MSD point, the input the *standard*
  inverse-variance scheme already uses.
* ``skew``           -- the sample skewness, a direct readout of the asymmetry
  that inverse-variance weighting wrongly assumes away.

All three are monotone in per-frame CNR.  :class:`TruncationModel` packages
them against CNR so a weighting scheme can look them up from a per-frame CNR
alone (measured, on real data, with the ``fft-cnr`` package).

The class supports **two calibration modes** with the same API:

* **1-D, D-aware** (legacy): built from :func:`frame_stats` via
  :meth:`TruncationModel.from_stats`. Looks up ``var_bias`` and
  ``rel_variance`` from CNR alone. The calibration implicitly conditions on
  the D used to generate the sweep -- see ``docs/DERIVATION.md``.
* **2-D, D-free** (recommended): built from a
  :class:`~folded_debias.calibration.CalibrationGrid` via
  :meth:`TruncationModel.from_calibration`. Looks up ``var_bias`` and
  per-frame ``rel_variance`` from (CNR, fitted-variance) jointly. The
  calibration carries no reference to *D*; the lookup is queried at each
  frame's own observed width. This is the deployment-ready mode.

The selection between the two is automatic: pass a ``variance`` argument to
:meth:`var_bias` or :meth:`rel_variance` to trigger the 2-D lookup when the
2-D table is present; otherwise the 1-D fallback runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .montecarlo import Ensemble


@dataclass
class FrameStats:
    """Per-frame summary of the MSD sampling distribution."""

    cnr: np.ndarray              # per-frame CNR
    trunc_fraction: np.ndarray   # P(MSD <= 0) per frame
    at_wall_fraction: np.ndarray # fraction of fits pinned at the width wall
    rel_variance: np.ndarray     # Var(MSD) per frame
    skew: np.ndarray             # sample skewness of MSD per frame
    mean_msd: np.ndarray         # mean fitted MSD per frame
    true_msd: np.ndarray         # ground-truth MSD per frame
    var_bias: np.ndarray         # E[sigma_hat^2] - sigma^2_true per frame
    #                              (the additive variance bias; see DERIVATION.md)


def _skew(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size < 3:
        return np.nan
    d = a - a.mean()
    s2 = (d ** 2).mean()
    if s2 == 0:
        return 0.0
    return float((d ** 3).mean() / s2 ** 1.5)


def frame_stats(ens: Ensemble) -> FrameStats:
    """Reduce a Monte Carlo ensemble to per-frame distribution statistics."""
    n_frames = ens.exp.n_frames
    trunc = np.full(n_frames, np.nan)
    rel_var = np.full(n_frames, np.nan)
    skew = np.full(n_frames, np.nan)
    mean_msd = np.full(n_frames, np.nan)
    var_bias = np.full(n_frames, np.nan)
    true_variance = ens.exp.true_variance

    for f in range(n_frames):
        col = ens.msd[:, f]
        col = col[np.isfinite(col)]
        if col.size:
            trunc[f] = float(np.mean(col <= 0.0))
            rel_var[f] = float(np.var(col, ddof=1)) if col.size > 1 else np.nan
            skew[f] = _skew(col)
            mean_msd[f] = float(np.mean(col))
        # Additive variance bias E[sigma_hat^2] - sigma^2_true (DERIVATION.md s.2).
        vcol = ens.variance[:, f]
        vcol = vcol[np.isfinite(vcol)]
        if vcol.size:
            var_bias[f] = float(np.mean(vcol)) - float(true_variance[f])

    at_wall_frac = np.nanmean(ens.at_wall.astype(float), axis=0)
    return FrameStats(cnr=ens.cnr_frame, trunc_fraction=trunc,
                      at_wall_fraction=at_wall_frac, rel_variance=rel_var,
                      skew=skew, mean_msd=mean_msd, true_msd=ens.exp.true_msd,
                      var_bias=var_bias)


def _bilinear(x_grid: np.ndarray, y_grid: np.ndarray, table: np.ndarray,
              x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of ``table[i, j]`` defined on ``x_grid`` x ``y_grid``.

    ``x_grid`` and ``y_grid`` are 1-D ascending; ``table`` has shape
    ``(x_grid.size, y_grid.size)``. Queries outside the grid are clipped to the
    boundary value (i.e. nearest-edge extrapolation in each axis).
    """
    x_arr = np.atleast_1d(np.asarray(x, float))
    y_arr = np.atleast_1d(np.asarray(y, float))

    xc = np.clip(x_arr, x_grid[0], x_grid[-1])
    yc = np.clip(y_arr, y_grid[0], y_grid[-1])

    i = np.clip(np.searchsorted(x_grid, xc) - 1, 0, x_grid.size - 2)
    j = np.clip(np.searchsorted(y_grid, yc) - 1, 0, y_grid.size - 2)

    x_lo = x_grid[i]; x_hi = x_grid[i + 1]
    y_lo = y_grid[j]; y_hi = y_grid[j + 1]
    fx = (xc - x_lo) / np.maximum(x_hi - x_lo, 1e-30)
    fy = (yc - y_lo) / np.maximum(y_hi - y_lo, 1e-30)

    v00 = table[i, j]
    v01 = table[i, j + 1]
    v10 = table[i + 1, j]
    v11 = table[i + 1, j + 1]
    out = (v00 * (1 - fx) * (1 - fy)
           + v10 * fx * (1 - fy)
           + v01 * (1 - fx) * fy
           + v11 * fx * fy)
    # Return scalar if scalar inputs were passed.
    if np.ndim(x) == 0 and np.ndim(y) == 0:
        return float(out[0])
    return out


@dataclass
class TruncationModel:
    """Per-frame CNR (+ optional variance) -> (p, rel_variance, var_bias).

    Two calibration modes share one API; see module docstring.

    Attributes
    ----------
    cnr_grid, p_grid, relvar_grid, varbias_grid : the 1-D lookup arrays
        (always present; the 2-D path also fills them with the variance-grid
        mean of each row so 1-D queries still work).
    variance_grid, varbias_2d, relvar_2d : the 2-D lookups (``None`` for a
        legacy 1-D calibration).
    """

    cnr_grid: np.ndarray
    p_grid: np.ndarray
    relvar_grid: np.ndarray
    varbias_grid: np.ndarray
    # Optional 2-D extensions, set by from_calibration().
    variance_grid: np.ndarray | None = None
    varbias_2d: np.ndarray | None = None
    relvar_2d: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_stats(cls, stats_list: list[FrameStats]) -> "TruncationModel":
        """Legacy 1-D calibration from a CNR sweep of full experiments."""
        cnr = np.concatenate([s.cnr for s in stats_list])
        p = np.concatenate([s.trunc_fraction for s in stats_list])
        rv = np.concatenate([s.rel_variance for s in stats_list])
        vb = np.concatenate([s.var_bias for s in stats_list])
        good = (np.isfinite(cnr) & np.isfinite(p) & np.isfinite(rv)
                & np.isfinite(vb) & (cnr > 0))
        order = np.argsort(cnr[good])
        return cls(cnr_grid=cnr[good][order],
                   p_grid=p[good][order],
                   relvar_grid=rv[good][order],
                   varbias_grid=vb[good][order])

    @classmethod
    def from_calibration(cls, calib) -> "TruncationModel":
        """2-D, D-free calibration from a :class:`CalibrationGrid`.

        Stores the 2-D lookup directly; also computes a 1-D marginal (mean
        over the variance grid) so 1-D queries still work as a fallback.
        ``p_grid`` is set to zeros: the truncation-fraction quantity is a
        relic of the 1-D mode and is unused by ``folded_normal`` / ``mas``.
        """
        cnr = np.asarray(calib.cnr_grid, float)
        var_g = np.asarray(calib.variance_grid, float)
        vb2 = np.asarray(calib.var_bias, float)
        rv2 = np.asarray(calib.rel_variance, float)
        # 1-D marginals: mean over the variance grid, for the no-variance path.
        vb_1d = np.nanmean(vb2, axis=1)
        rv_1d = np.nanmean(rv2, axis=1)
        p_1d = np.zeros_like(cnr)
        order = np.argsort(cnr)
        return cls(
            cnr_grid=cnr[order],
            p_grid=p_1d[order],
            relvar_grid=rv_1d[order],
            varbias_grid=vb_1d[order],
            variance_grid=var_g,
            varbias_2d=vb2[order, :],
            relvar_2d=rv2[order, :],
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------
    @property
    def is_2d(self) -> bool:
        return self.variance_grid is not None and self.varbias_2d is not None

    def p(self, cnr) -> np.ndarray:
        """Interpolated truncation fraction at the given per-frame CNR."""
        return np.interp(cnr, self.cnr_grid, self.p_grid,
                         left=self.p_grid[0], right=self.p_grid[-1])

    def var_bias(self, cnr, variance=None, n_iter: int = 4):
        """Interpolated additive variance bias.

        ``E[sigma_hat^2] - sigma^2_true``. The folded-normal model
        (DERIVATION.md) identifies this as the estimation-noise variance s^2;
        subtracting it per frame removes the bias the MSD slope would
        otherwise inherit.

        With a 2-D calibration **and** a ``variance`` argument (the observed
        raw fitted variance), looks up the bias at (CNR, sigma^2_true)
        jointly; otherwise falls back to the 1-D lookup on CNR alone.

        **Subtlety: the lookup is calibrated against the true variance, but
        the application only knows the biased observed variance.** Querying
        directly at the observed value evaluates the bias at the wrong point,
        which distorts the per-frame correction (especially badly at low CNR,
        where the bias is largest). The fix is a short fixed-point iteration:
        starting from the observed variance, subtract the lookup at the
        current estimate to refine an estimate of the true variance, and
        re-query. Three or four iterations suffice; the bias used for the
        correction is the lookup at the converged estimate.
        """
        if variance is None or not self.is_2d:
            return np.interp(cnr, self.cnr_grid, self.varbias_grid,
                             left=self.varbias_grid[0],
                             right=self.varbias_grid[-1])
        sigma2_obs = np.asarray(variance, float)
        sigma2_true = sigma2_obs.copy()
        # Clip the iterate to the calibration's variance range to keep the
        # bilinear interpolation in bounds. The clip is per-iteration; a
        # frame whose observed variance lies outside the calibration domain
        # still gets the boundary lookup value (no extrapolation).
        var_lo = float(self.variance_grid[0])
        var_hi = float(self.variance_grid[-1])
        for _ in range(int(n_iter)):
            sigma2_true_clipped = np.clip(sigma2_true, var_lo, var_hi)
            bias = _bilinear(self.cnr_grid, self.variance_grid,
                             self.varbias_2d, cnr, sigma2_true_clipped)
            sigma2_true = sigma2_obs - bias
        return sigma2_obs - sigma2_true

    def rel_variance(self, cnr, variance=None):
        """Interpolated variance of the (fitted-variance) point.

        With a 1-D calibration this is the MSD-point variance
        ``Var(sigma_hat^2(t) - sigma_hat^2(0))`` empirically measured per CNR.
        With a 2-D calibration this is the per-frame ``Var(sigma_hat^2)``; the
        MSD-point variance is formed in :mod:`folded_debias.weights` by
        adding the t=0 frame's contribution.
        """
        if variance is None or not self.is_2d:
            return np.interp(cnr, self.cnr_grid, self.relvar_grid,
                             left=self.relvar_grid[0],
                             right=self.relvar_grid[-1])
        return _bilinear(self.cnr_grid, self.variance_grid, self.relvar_2d,
                         cnr, variance)
