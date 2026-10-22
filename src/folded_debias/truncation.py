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
    var_bias_se: np.ndarray | None = None  # Monte Carlo SE of var_bias per frame
    frame_variance: np.ndarray | None = None  # Var(sigma_hat^2) per frame


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
    var_bias_se = np.full(n_frames, np.nan)
    frame_var = np.full(n_frames, np.nan)
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
        if vcol.size > 1:
            # The lookup's own Monte Carlo error; propagated into D by
            # weights.calibration_D_se.
            var_bias_se[f] = float(np.std(vcol, ddof=1) / np.sqrt(vcol.size))
            # The width variance of this frame alone, a function of its own
            # CNR. rel_variance above also carries this ensemble's t = 0 frame,
            # which belongs to the calibration's CNR_0, not the application's.
            frame_var[f] = float(np.var(vcol, ddof=1))

    at_wall_frac = np.nanmean(ens.at_wall.astype(float), axis=0)
    return FrameStats(cnr=ens.cnr_frame, trunc_fraction=trunc,
                      at_wall_fraction=at_wall_frac, rel_variance=rel_var,
                      skew=skew, mean_msd=mean_msd, true_msd=ens.exp.true_msd,
                      var_bias=var_bias, var_bias_se=var_bias_se,
                      frame_variance=frame_var)


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


def _interp_se(x, xp: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """SE of ``np.interp(x, xp, fp)`` when each ``fp[k]`` has independent SE ``sp[k]``.

    The interpolant is ``(1 - f) fp[k] + f fp[k+1]``, so its SE is
    ``sqrt((1 - f)^2 sp[k]^2 + f^2 sp[k+1]^2)``; outside the grid it is the
    edge point's SE, matching np.interp's edge clamping.
    """
    xa = np.clip(np.atleast_1d(np.asarray(x, float)), xp[0], xp[-1])
    k = np.clip(np.searchsorted(xp, xa) - 1, 0, xp.size - 2)
    f = (xa - xp[k]) / np.maximum(xp[k + 1] - xp[k], 1e-30)
    out = np.sqrt((1 - f) ** 2 * sp[k] ** 2 + f ** 2 * sp[k + 1] ** 2)
    return float(out[0]) if np.ndim(x) == 0 else out


def _bilinear_se(x_grid: np.ndarray, y_grid: np.ndarray, se_table: np.ndarray,
                 x, y) -> np.ndarray:
    """SE of :func:`_bilinear` when each table cell has an independent SE."""
    xc = np.clip(np.atleast_1d(np.asarray(x, float)), x_grid[0], x_grid[-1])
    yc = np.clip(np.atleast_1d(np.asarray(y, float)), y_grid[0], y_grid[-1])
    i = np.clip(np.searchsorted(x_grid, xc) - 1, 0, x_grid.size - 2)
    j = np.clip(np.searchsorted(y_grid, yc) - 1, 0, y_grid.size - 2)
    fx = (xc - x_grid[i]) / np.maximum(x_grid[i + 1] - x_grid[i], 1e-30)
    fy = (yc - y_grid[j]) / np.maximum(y_grid[j + 1] - y_grid[j], 1e-30)
    var = (((1 - fx) * (1 - fy)) ** 2 * se_table[i, j] ** 2
           + (fx * (1 - fy)) ** 2 * se_table[i + 1, j] ** 2
           + ((1 - fx) * fy) ** 2 * se_table[i, j + 1] ** 2
           + (fx * fy) ** 2 * se_table[i + 1, j + 1] ** 2)
    out = np.sqrt(var)
    return float(out[0]) if (np.ndim(x) == 0 and np.ndim(y) == 0) else out


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
    varbias_se_grid, varbias_se_2d : Monte Carlo SE of each var_bias point
        (``None`` for a model built or saved without it). They give the
        calibration's own error, which :func:`~folded_debias.weights.calibration_D_se`
        propagates into D.
    framevar_grid : per-frame ``Var(sigma_hat^2)`` on ``cnr_grid`` (1-D
        models from :meth:`from_stats`; ``None`` for models built or saved
        without it, which keep the older ``relvar_grid`` weights). See
        :meth:`frame_variance`.
    """

    cnr_grid: np.ndarray
    p_grid: np.ndarray
    relvar_grid: np.ndarray
    varbias_grid: np.ndarray
    # Optional 2-D extensions, set by from_calibration().
    variance_grid: np.ndarray | None = None
    varbias_2d: np.ndarray | None = None
    relvar_2d: np.ndarray | None = None
    varbias_se_grid: np.ndarray | None = None
    varbias_se_2d: np.ndarray | None = None
    framevar_grid: np.ndarray | None = None

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
        se = fv = None
        if all(s.var_bias_se is not None for s in stats_list):
            se = np.concatenate([s.var_bias_se for s in stats_list])[good][order]
        if all(s.frame_variance is not None for s in stats_list):
            fv = np.concatenate([s.frame_variance for s in stats_list])[good][order]
        return cls(cnr_grid=cnr[good][order],
                   p_grid=p[good][order],
                   relvar_grid=rv[good][order],
                   varbias_grid=vb[good][order],
                   varbias_se_grid=se,
                   framevar_grid=fv)

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
        # Per-cell SE of the mean bias from the cell's sample variance. The
        # bias mean drops fits above the calibration's filter threshold, so
        # this (unfiltered) spread makes the SE slightly conservative.
        se2 = np.sqrt(rv2 / max(int(calib.n_runs), 1))
        n_var = np.sum(np.isfinite(se2), axis=1)
        se_1d = np.sqrt(np.nansum(se2 ** 2, axis=1)) / np.maximum(n_var, 1)
        order = np.argsort(cnr)
        return cls(
            cnr_grid=cnr[order],
            p_grid=p_1d[order],
            relvar_grid=rv_1d[order],
            varbias_grid=vb_1d[order],
            variance_grid=var_g,
            varbias_2d=vb2[order, :],
            relvar_2d=rv2[order, :],
            varbias_se_grid=se_1d[order],
            varbias_se_2d=se2[order, :],
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

        **Choice of key.** ``variance`` is the key that stands in for the
        unknown true sigma^2, and any array can be passed. Keying on each
        frame's own sigma_hat^2 (the usual call) under-corrects when sigma^2
        grows several-fold over the measurement: at low CNR a frame whose fit
        scatters high reads the lookup near the grid edge, and the mean is
        over-subtracted. On the high-N Br-I bed (sigma^2 growing ~64x) this
        left D at 0.89 to 0.96 of the reference at CNR_0 15 to 20, where a
        measurement-level key (a robust line through the measurement's own
        sigma_hat^2) matched the exact key to 0.003. That key assumes a form
        for sigma^2(t), so it is not the default here; see
        ``experiments/halide-mixing-highn/key_sensitivity.py`` and
        ``smoothed_key`` in ``br_halide_highn_perframe.py`` for a worked
        example and its limits (below CNR_0 ~15 no key determined D).
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

    def var_bias_se(self, cnr, variance=None, n_iter: int = 4):
        """Monte Carlo SE of the looked-up :meth:`var_bias`.

        The calibration is a finite Monte Carlo, so every subtracted bias
        carries an error the measurement's own SE does not see. At low CNR it
        dominates: two 300-run calibrations of the same sweep differ by 0.22
        in D/D_ref at D = 0.02 sigma_0^2/tau, CNR_0 = 34, against an
        evaluation SE of 0.04. The SE is propagated through the same
        interpolation weights as the lookup (independent cells assumed; for
        the D-free 1-D grid each point is its own ensemble frame, the frames of
        one ensemble are uncorrelated, and the per-point SE matches the spread
        over 16 independent calibrations). With
        a 2-D model and a ``variance``, it is evaluated at the same
        fixed-point estimate of sigma^2_true that :meth:`var_bias` uses.
        Returns NaN when the model carries no SE (older saved calibrations).
        """
        cnr_arr = np.asarray(cnr, float)
        if variance is None or not self.is_2d:
            if self.varbias_se_grid is None:
                return np.full(np.shape(cnr_arr), np.nan) if np.ndim(cnr_arr) else np.nan
            return _interp_se(cnr_arr, self.cnr_grid, self.varbias_se_grid)
        if self.varbias_se_2d is None:
            return np.full(np.shape(cnr_arr), np.nan) if np.ndim(cnr_arr) else np.nan
        sigma2_obs = np.asarray(variance, float)
        sigma2_true = sigma2_obs - self.var_bias(cnr_arr, sigma2_obs, n_iter)
        sigma2_true = np.clip(sigma2_true, float(self.variance_grid[0]),
                              float(self.variance_grid[-1]))
        return _bilinear_se(self.cnr_grid, self.variance_grid,
                            self.varbias_se_2d, cnr_arr, sigma2_true)

    def frame_variance(self, cnr):
        """Interpolated per-frame ``Var(sigma_hat^2)`` at the given CNR (1-D).

        The slope weights are ``1 / Var(sigma_hat_i^2)`` at the application's
        own frame CNR, frame 0 included (see
        :func:`~folded_debias.weights.msd_and_weights`). The 1-D
        ``relvar_grid`` holds a different quantity, the MSD-point variance
        ``Var(sigma_hat_i^2) + Var(sigma_hat_0^2)`` measured inside each
        calibration ensemble, whose t = 0 frame sits at the calibration's
        CNR_0. Neighbouring grid points come from different ensembles, so that
        lookup is jagged, and each ensemble's own t = 0 point reads exactly 0:
        a frame whose CNR lay next to a calibration CNR_0 drew 5 to 50 times
        its proper weight. Returns ``None`` when the model carries no such grid.
        """
        if self.framevar_grid is None:
            return None
        return np.interp(cnr, self.cnr_grid, self.framevar_grid,
                         left=self.framevar_grid[0], right=self.framevar_grid[-1])

    def rel_variance(self, cnr, variance=None):
        """Interpolated variance of the (fitted-variance) point.

        With a 1-D calibration this is the MSD-point variance
        ``Var(sigma_hat^2(t) - sigma_hat^2(0))`` empirically measured per CNR.
        With a 2-D calibration this is the per-frame ``Var(sigma_hat^2)``, which
        :mod:`folded_debias.weights` uses directly as each frame's weight.
        """
        if variance is None or not self.is_2d:
            return np.interp(cnr, self.cnr_grid, self.relvar_grid,
                             left=self.relvar_grid[0],
                             right=self.relvar_grid[-1])
        return _bilinear(self.cnr_grid, self.variance_grid, self.relvar_2d,
                         cnr, variance)
