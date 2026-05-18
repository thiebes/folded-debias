"""Characterising the truncation: the wall at zero.

For each frame, the Monte Carlo ensemble gives the sampling distribution of the
fitted MSD increment.  Three quantities summarise how the wall at zero distorts
that distribution:

* ``trunc_fraction`` -- the fraction of realisations with MSD <= 0, i.e. the
  probability mass that has crossed to the wrong side of the D >= 0 wall.  This
  is the central new input to the weighting scheme.
* ``rel_variance``   -- the variance of the MSD point, the input the *standard*
  inverse-variance scheme already uses.
* ``skew``           -- the sample skewness, a direct readout of the asymmetry
  that inverse-variance weighting wrongly assumes away.

All three are monotone in per-frame CNR.  ``TruncationModel`` packages them
against CNR so a weighting scheme can look them up from a per-frame CNR alone
(measured, on real data, with the ``fft-cnr`` package).
"""
from __future__ import annotations

from dataclasses import dataclass

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

    for f in range(n_frames):
        col = ens.msd[:, f]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        trunc[f] = float(np.mean(col <= 0.0))
        rel_var[f] = float(np.var(col, ddof=1)) if col.size > 1 else np.nan
        skew[f] = _skew(col)
        mean_msd[f] = float(np.mean(col))

    at_wall_frac = np.nanmean(ens.at_wall.astype(float), axis=0)
    return FrameStats(cnr=ens.cnr_frame, trunc_fraction=trunc,
                      at_wall_fraction=at_wall_frac, rel_variance=rel_var,
                      skew=skew, mean_msd=mean_msd, true_msd=ens.exp.true_msd)


@dataclass
class TruncationModel:
    """Truncation fraction and relative MSD variance as functions of CNR.

    Built by pooling ``FrameStats`` from a CNR sweep.  Given a per-frame CNR it
    returns the truncation fraction p(CNR) and the relative variance v(CNR) by
    monotone interpolation -- the bridge from a measured per-frame CNR to the
    quantities the weighting scheme needs.
    """

    cnr_grid: np.ndarray
    p_grid: np.ndarray
    relvar_grid: np.ndarray

    @classmethod
    def from_stats(cls, stats_list: list[FrameStats]) -> "TruncationModel":
        cnr = np.concatenate([s.cnr for s in stats_list])
        p = np.concatenate([s.trunc_fraction for s in stats_list])
        # Per-frame MSD variance.  Only its value relative to other frames
        # matters for weighting, so no normalisation is needed here.
        rv = np.concatenate([s.rel_variance for s in stats_list])
        good = np.isfinite(cnr) & np.isfinite(p) & np.isfinite(rv) & (cnr > 0)
        order = np.argsort(cnr[good])
        return cls(cnr_grid=cnr[good][order],
                   p_grid=p[good][order],
                   relvar_grid=rv[good][order])

    def p(self, cnr: np.ndarray | float) -> np.ndarray:
        """Interpolated truncation fraction at the given per-frame CNR."""
        return np.interp(cnr, self.cnr_grid, self.p_grid,
                         left=self.p_grid[0], right=self.p_grid[-1])

    def rel_variance(self, cnr: np.ndarray | float) -> np.ndarray:
        """Interpolated relative MSD variance at the given per-frame CNR."""
        return np.interp(cnr, self.cnr_grid, self.relvar_grid,
                         left=self.relvar_grid[0], right=self.relvar_grid[-1])
