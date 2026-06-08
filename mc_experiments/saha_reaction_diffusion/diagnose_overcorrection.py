"""Diagnose the high-spread folded-normal overcorrection on the Saha bed.

At low CNR_0 and high spread, folded-normal pulls the recovered alpha below the
noise-free-fit reference. The shape-mismatch explanation is ruled out (the Saha
emission profile is near-Gaussian, beta ~ 1.93), so this script tests the next
hypothesis: the 2-D calibration over-estimates the per-frame variance bias at
the low-CNR late frames and folded therefore over-subtracts.

Per frame at a stressed operating point it compares two quantities:

* B_actual  -- the real noise-induced Gaussian-fit bias, measured by Monte Carlo
               on the actual Saha profile: E[sigma_hat^2] - sigma^2_ref, where
               sigma^2_ref is the noise-free fit variance.
* B_calib   -- what folded-normal actually subtracts: the mean over the same
               realisations of model.var_bias(cnr, sigma_hat^2) (the 2-D lookup
               with its fixed-point inversion, exactly as `corrected_variance`
               calls it).

If B_calib > B_actual at the low-CNR late frames, the overcorrection is a
calibration-accuracy problem (deep-CNR MC noise), pointing the fix at a denser /
higher-run-count low-CNR calibration rather than a shape-matched estimator.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

import nonfickian_validation as nf  # noqa: E402
from weighting_wall.fit import fit_variance  # noqa: E402


def diagnose(A: float, cnr0: float, power: int = 2, n_mc: int = 4000,
             seed: int = 0) -> None:
    rg = nf.build_regime(A, power)
    model = nf.load_model()
    if model is None:
        sys.exit(f"no calibration at {nf.CALIBRATION_PATH}")
    noise_sigma = rg.peak[0] / cnr0
    cnr_frame = rg.peak / noise_sigma
    rng = np.random.default_rng(seed)

    print(f"A={A:.0f}, CNR_0={cnr0}, emission power={power}, {n_mc} MC fits/frame")
    print(" frame  t(ns)   cnr  sigma2_ref  B_actual  B_calib   verdict")
    for f in range(nf.N_FRAMES):
        ref = rg.fit_var[f]
        clean = rg.clean[f]
        fitted = np.empty(n_mc)
        subtracted = np.empty(n_mc)
        for r in range(n_mc):
            noisy = clean + rng.normal(0.0, noise_sigma, size=clean.shape)
            v = fit_variance(nf.PIXEL_X, noisy).variance
            fitted[r] = v
            subtracted[r] = model.var_bias(float(cnr_frame[f]), v)
        b_actual = float(fitted.mean()) - ref
        b_calib = float(subtracted.mean())
        gap = b_calib - b_actual
        verdict = "OVER" if gap > 0.005 else ("under" if gap < -0.005 else "ok")
        print(f"  {f:2d}   {rg.times[f]:5.2f}  {cnr_frame[f]:5.2f}   {ref:.4f}"
              f"   {b_actual:+.4f}  {b_calib:+.4f}   {verdict} ({gap:+.4f})")


if __name__ == "__main__":
    diagnose(2000.0, 12.0)
    print()
    diagnose(2000.0, 25.0)
