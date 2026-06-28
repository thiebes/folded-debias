"""Locate the analytical floor of the folded-normal correction.

In the folded-normal model the calibrated `var_bias(CNR)` is the variance
of the width estimator, which follows a localization-CRLB scaling at high
CNR:

    var_bias(CNR)  ~=  A / CNR^2

where `A` is set by the experimental geometry (pixel spacing, true width,
profile shape). At low CNR the fit's actual distribution stops being a
folded normal -- the estimator's bounds and the likelihood surface's
structure take over -- and the empirical `var_bias(CNR)` diverges from
the power-law prediction. The crossover marks the **analytical floor**:
below it, no amount of calibration with simple subtraction can debias the
fit, because the underlying model has broken.

This script (a) builds a modest calibration so the var_bias curve covers
deep CNR cleanly, (b) fits the power-law `A / CNR^2` to the high-CNR
portion (where the model is in its valid regime), (c) compares the
empirical curve to the fit, and (d) reports the CNR at which the empirical
curve diverges from the power-law by a stated factor.

The script is intentionally light: 30 log-spaced CNR_0 calibration points,
300 runs per cell. That is enough to *resolve* the curve's shape; we are
not estimating per-cell values precisely.

Usage::

    .venv/Scripts/python.exe scripts/analytical_floor.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from weighting_wall import MODELS, Experiment  # noqa: E402
from weighting_wall.montecarlo import run_ensemble  # noqa: E402
from weighting_wall.truncation import TruncationModel, frame_stats  # noqa: E402


CNR0_RANGE = (2.0, 100.0)
N_CALIB = 30
N_RUNS_CAL = 300
HIGH_CNR_FIT_MIN = 8.0     # fit the power-law on CNR > this
DIVERGENCE_FACTOR = 2.0    # the "floor" is where empirical/fit exceeds this
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


def _calib_one(args):
    cnr0, exp, n_runs, seed, model_name = args
    model = MODELS[model_name]()
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model)
    return frame_stats(ens)


def modest_calibrate(model_name: str) -> TruncationModel:
    cal_exp = Experiment(D=0.0)
    cnr0_grid = np.logspace(np.log10(CNR0_RANGE[0]),
                             np.log10(CNR0_RANGE[1]),
                             N_CALIB)
    args_list = [(float(c), cal_exp, N_RUNS_CAL, 1000 + i, model_name)
                 for i, c in enumerate(cnr0_grid)]
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_calib_one, args): i
                   for i, args in enumerate(args_list)}
        results = [None] * len(args_list)
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return TruncationModel.from_stats(results)


def find_floor(model_name: str, trunc: TruncationModel) -> dict:
    """Fit A/CNR^2 to the high-CNR var_bias; return the floor CNR."""
    cnr = trunc.cnr_grid
    vb = trunc.varbias_grid

    # Keep only positive var_bias points (negative values are MC noise at high CNR
    # where the true bias is essentially zero).
    good = (cnr > 0) & (vb > 0) & np.isfinite(cnr) & np.isfinite(vb)
    cnr, vb = cnr[good], vb[good]

    # Fit the power-law on CNR > HIGH_CNR_FIT_MIN. Power-law: vb = A * CNR^(-2),
    # i.e. log(vb) = log(A) - 2 log(CNR). Fit log(A) by least squares with the
    # exponent fixed at -2.
    fit_mask = cnr > HIGH_CNR_FIT_MIN
    if fit_mask.sum() < 3:
        raise RuntimeError(f"Too few high-CNR points ({fit_mask.sum()}) to fit")
    log_A = np.mean(np.log(vb[fit_mask]) + 2.0 * np.log(cnr[fit_mask]))
    A = float(np.exp(log_A))

    # Compare empirical to power-law prediction, sorted by CNR.
    order = np.argsort(cnr)
    cnr_s = cnr[order]
    vb_s = vb[order]
    pred = A / cnr_s ** 2
    ratio = vb_s / pred

    # The floor is the LOWEST CNR for which ratio is within DIVERGENCE_FACTOR
    # of 1, scanning from high to low CNR (so as soon as ratio leaves the
    # accepted band, we stop). Use both bounds (overshoot and undershoot).
    in_band = (ratio < DIVERGENCE_FACTOR) & (ratio > 1.0 / DIVERGENCE_FACTOR)
    # Walk from highest CNR downward; the floor is the last CNR before the band fails.
    floor = None
    for i in range(len(cnr_s) - 1, -1, -1):
        if not in_band[i]:
            if i + 1 < len(cnr_s):
                floor = float(cnr_s[i + 1])
            break
    if floor is None:
        floor = float(cnr_s[0])  # band held all the way down

    return {
        "model": model_name,
        "A": A,
        "floor_cnr": floor,
        "cnr": cnr_s,
        "var_bias_empirical": vb_s,
        "var_bias_predicted": pred,
        "ratio": ratio,
    }


def report(result: dict) -> None:
    print(f"\n=== {result['model']} ===")
    print(f"  Power-law fit: var_bias(CNR) = {result['A']:.4f} / CNR^2 sigma_0^2"
          f"  (fit range CNR > {HIGH_CNR_FIT_MIN})")
    print(f"  Analytical floor (where empirical/predicted > "
          f"{DIVERGENCE_FACTOR} or < {1/DIVERGENCE_FACTOR:.2g}): "
          f"CNR ~ {result['floor_cnr']:.2f}")
    print(f"\n  CNR     empirical    predicted   ratio")
    for c, e, p, r in zip(result["cnr"], result["var_bias_empirical"],
                            result["var_bias_predicted"], result["ratio"]):
        flag = "" if (1/DIVERGENCE_FACTOR < r < DIVERGENCE_FACTOR) else "  <<<"
        print(f"  {c:6.2f}  {e:11.4f}  {p:11.4f}  {r:6.2f}{flag}")


def main() -> None:
    print(f"Analytical floor of the folded-normal correction")
    print(f"  Calibration: {N_CALIB} log-spaced CNR_0 in "
          f"{CNR0_RANGE}, {N_RUNS_CAL} runs/cell")
    print(f"  Power-law fit on CNR > {HIGH_CNR_FIT_MIN}")
    print(f"  Floor criterion: empirical/predicted outside "
          f"[{1/DIVERGENCE_FACTOR:.2g}, {DIVERGENCE_FACTOR}]\n")

    t_start = time.time()
    for name in ("gaussian", "voigt", "generalized_gaussian"):
        t0 = time.time()
        trunc = modest_calibrate(name)
        result = find_floor(name, trunc)
        report(result)
        print(f"  [{name}: {time.time()-t0:.0f}s]")
    print(f"\n[total {(time.time()-t_start)/60:.1f} min]")


if __name__ == "__main__":
    main()
