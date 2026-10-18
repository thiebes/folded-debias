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
deep CNR cleanly, (b) takes A from the measured estimator spread at high
CNR (the folded-normal model sets var_bias = Var(sigma_hat), so no fit of
the bias itself is needed), (c) pools the per-frame var_bias cells into
log-CNR bins with standard errors, and (d) reports the CNR at which the
resolved bins leave a factor-2 band around A / CNR^2, walking down from
high CNR.

The script is intentionally light: 30 log-spaced CNR_0 calibration points,
300 runs per cell. Single var_bias cells are noise-dominated above CNR ~ 3
(the bias there is below 1e-2 sigma_0^2). An earlier version fitted A to
those unbinned high-CNR cells after dropping the negative ones; that
overestimated A about five-fold (0.37 against about 0.06 sigma_0^2), and a
loop that exited at an out-of-band top cell reported the lowest grid CNR
(0.1) as the floor.

Usage::

    .venv/Scripts/python.exe experiments/correction-floor/analytical_floor.py [MODEL ...]

MODEL is ``gaussian``, ``voigt``, or ``generalized_gaussian:<p>``; the default
runs Gaussian, Voigt, and the generalized-Gaussian at p = 1.5 and p = 4.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import Experiment, model_from_spec, spec_tag  # noqa: E402
from folded_debias.montecarlo import run_ensemble  # noqa: E402
from folded_debias.truncation import TruncationModel, frame_stats  # noqa: E402


CNR0_RANGE = (2.0, 100.0)
N_CALIB = 30
N_RUNS_CAL = 300
SPREAD_CNR_MIN = 8.0      # A from the estimator spread on cells at CNR >= this
N_BINS = 24               # log-CNR bins pooling the per-frame calibration cells
RESOLVE_Z = 3.0           # a bin is resolved if |mean var_bias| > this many SE
DIVERGENCE_FACTOR = 2.0   # the "floor" is where empirical/predicted leaves this band
DEPART_Z = 2.0            # ... by more than this many standard errors of the ratio
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


def _calib_one(args):
    """One calibration cell: per-frame stats plus the per-frame Var(sigma_hat^2)."""
    cnr0, exp, n_runs, seed, model_name = args
    model = model_from_spec(model_name)
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model)
    return frame_stats(ens), np.nanvar(ens.variance, axis=0, ddof=1)


def modest_calibrate(model_name: str):
    """Return the calibration lookup and, per cell, (CNR, Var(sigma_hat^2))."""
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
    stats = [r[0] for r in results]
    spread_cnr = np.concatenate([s.cnr for s in stats])
    spread_var = np.concatenate([r[1] for r in results])
    return TruncationModel.from_stats(stats), spread_cnr, spread_var


def _bin_log_cnr(cnr: np.ndarray, vb: np.ndarray) -> tuple[np.ndarray, ...]:
    """Mean var_bias and its standard error in log-spaced CNR bins.

    One calibration cell's var_bias is a Monte Carlo mean over N_RUNS_CAL fits;
    above CNR ~ 3 its noise exceeds the A/CNR^2 signal, so single cells cannot
    test the model there. Pooling neighbouring cells and taking the standard
    error from their scatter gives a resolved estimate where one exists, and
    says so where it does not. Negative cells stay in: they are the lower half
    of the noise, and dropping them biases every high-CNR bin upward.
    """
    edges = np.logspace(np.log10(cnr.min()), np.log10(cnr.max()) + 1e-9,
                        N_BINS + 1)
    idx = np.digitize(cnr, edges) - 1
    centers, means, ses = [], [], []
    for b in range(N_BINS):
        sel = idx == b
        if sel.sum() < 3:
            continue
        centers.append(float(np.exp(np.mean(np.log(cnr[sel])))))
        means.append(float(vb[sel].mean()))
        ses.append(float(vb[sel].std(ddof=1) / np.sqrt(sel.sum())))
    return np.array(centers), np.array(means), np.array(ses)


def find_floor(model_name: str, trunc: TruncationModel,
               spread_cnr: np.ndarray, spread_var: np.ndarray) -> dict:
    """Estimate A from the estimator spread; locate where var_bias departs.

    The folded-normal model says E[sigma_hat^2] = sigma^2 + Var(sigma_hat),
    so var_bias = Var(sigma_hat). For a small, near-normal width error,
    Var(sigma_hat^2) = 4 sigma^2 Var(sigma_hat), and with the calibration at
    D = 0 the true variance is sigma_0^2 = 1 in every frame, so
    A = CNR^2 Var(sigma_hat^2) / 4. Unlike the mean bias, the spread is
    measured to a few per cent per cell at any CNR, so A is fixed at high CNR
    (>= SPREAD_CNR_MIN), where the small-noise relation holds, without fitting
    the noise-dominated high-CNR bias. The prediction A/CNR^2 is then tested
    against the binned var_bias at lower CNR.
    """
    ok = np.isfinite(spread_cnr) & np.isfinite(spread_var) \
        & (spread_cnr >= SPREAD_CNR_MIN)
    a_cells = spread_cnr[ok] ** 2 * spread_var[ok] / 4.0
    A = float(np.median(a_cells))
    A_spread = float(np.percentile(a_cells, 84) - np.percentile(a_cells, 16)) / 2

    good = np.isfinite(trunc.cnr_grid) & np.isfinite(trunc.varbias_grid) \
        & (trunc.cnr_grid > 0)
    cnr, vb, se = _bin_log_cnr(trunc.cnr_grid[good], trunc.varbias_grid[good])
    pred = A / cnr ** 2
    ratio = vb / pred
    ratio_se = se / pred
    # Resolved = the bin mean differs from zero by more than RESOLVE_Z SE, in
    # either sign: a significantly negative var_bias (the p = 4 fit at low CNR)
    # is a departure from the model, not noise to skip.
    resolved = np.abs(vb) > RESOLVE_Z * se
    # A bin leaves the band only if it lies outside it by more than DEPART_Z
    # of its own standard errors; a bin merely straddling an edge does not.
    above = (ratio - DIVERGENCE_FACTOR) > DEPART_Z * ratio_se
    below = (1.0 / DIVERGENCE_FACTOR - ratio) > DEPART_Z * ratio_se
    in_band = ~(above | below)

    # Cross-check: weighted least-squares A from the resolved bins at CNR >= 1.
    fit = resolved & (vb > 0) & (cnr >= 1.0)
    A_bias = np.nan
    if fit.sum() >= 2:
        g, w = cnr[fit] ** -2.0, se[fit] ** -2.0
        A_bias = float(np.sum(w * g * vb[fit]) / np.sum(w * g * g))

    # Walk down from the highest resolved bin. The floor is the lowest bin
    # reached before the first resolved bin that leaves the band; that bin is
    # the breakdown. Unresolved bins neither set nor break the floor. If the
    # band holds to the bottom of the grid, the breakdown is not located and
    # the lowest bin is an upper bound on it.
    floor, breakdown, reached = np.nan, np.nan, False
    for i in range(len(cnr) - 1, -1, -1):
        if not resolved[i]:
            continue
        if not in_band[i]:
            reached, breakdown = True, float(cnr[i])
            break
        floor = float(cnr[i])

    return {
        "model": model_name,
        "A": A,
        "A_spread": A_spread,
        "A_bias": A_bias,
        "floor_cnr": floor,
        "breakdown_cnr": breakdown,
        "floor_reached": reached,
        "in_band": in_band,
        "cnr": cnr,
        "var_bias_empirical": vb,
        "var_bias_se": se,
        "var_bias_predicted": pred,
        "ratio": ratio,
        "ratio_se": ratio_se,
        "resolved": resolved,
    }


def report(result: dict) -> None:
    print(f"\n=== {result['model']} ===")
    print(f"  A from the estimator spread (CNR >= {SPREAD_CNR_MIN}): "
          f"{result['A']:.4f} sigma_0^2  (cell-to-cell half-width "
          f"{result['A_spread']:.4f})")
    print(f"  A from the resolved var_bias bins at CNR >= 1 (cross-check): "
          f"{result['A_bias']:.4f} sigma_0^2")
    band = f"[{1/DIVERGENCE_FACTOR:.2g}, {DIVERGENCE_FACTOR}]"
    if result["floor_reached"]:
        print(f"  Breakdown: var_bias / (A/CNR^2) leaves {band} by > {DEPART_Z} SE "
              f"at CNR ~ {result['breakdown_cnr']:.2f}; lowest in-band "
              f"resolved bin above it: CNR ~ {result['floor_cnr']:.2f}")
    else:
        print(f"  No breakdown located: every resolved bin is within {band}; "
              f"the breakdown is below CNR ~ {result['floor_cnr']:.2f}")
    print(f"\n  {'CNR':>7}  {'empirical':>10}  {'SE':>8}  {'predicted':>10}"
          f"  {'ratio':>6}  {'+/-':>5}")
    for c, e, s, p, r, rs, ok, inb in zip(
            result["cnr"], result["var_bias_empirical"], result["var_bias_se"],
            result["var_bias_predicted"], result["ratio"], result["ratio_se"],
            result["resolved"], result["in_band"]):
        if not ok:
            flag = "  (unresolved)"
        elif inb:
            flag = ""
        else:
            flag = "  <<<"
        print(f"  {c:7.3f}  {e:10.4f}  {s:8.4f}  {p:10.4f}  {r:6.2f}  "
              f"{rs:5.2f}{flag}")


def main() -> None:
    print(f"Analytical floor of the folded-normal correction")
    print(f"  Calibration: {N_CALIB} log-spaced CNR_0 in "
          f"{CNR0_RANGE}, {N_RUNS_CAL} runs/cell")
    print(f"  A = CNR^2 Var(sigma_hat^2) / 4, median over cells at CNR >= "
          f"{SPREAD_CNR_MIN}")
    print(f"  var_bias pooled into {N_BINS} log-CNR bins; a bin is resolved if "
          f"|mean| exceeds {RESOLVE_Z} SE")
    print(f"  Floor criterion: resolved empirical/predicted outside "
          f"[{1/DIVERGENCE_FACTOR:.2g}, {DIVERGENCE_FACTOR}] by > {DEPART_Z} SE\n")

    # Optional model specs; "generalized_gaussian:<p>" sets the shape (the
    # bare name is p = 2, a Gaussian profile).
    specs = sys.argv[1:] or ["gaussian", "voigt", "generalized_gaussian:1.5",
                             "generalized_gaussian:4"]
    for spec in specs:
        model_from_spec(spec)          # fail fast on a bad spec
    t_start = time.time()
    for name in specs:
        t0 = time.time()
        trunc, spread_cnr, spread_var = modest_calibrate(name)
        result = find_floor(spec_tag(name), trunc, spread_cnr, spread_var)
        report(result)
        print(f"  [{spec_tag(name)}: {time.time()-t0:.0f}s]")
    print(f"\n[total {(time.time()-t_start)/60:.1f} min]")


if __name__ == "__main__":
    main()
