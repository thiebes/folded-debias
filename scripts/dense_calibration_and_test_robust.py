"""Headline-regime validation of the robust multiplicative scheme.

Mirrors ``dense_calibration_and_test.py`` but adds a fifth scheme alongside
the standard four:

  * ols, inverse_variance, folded_normal, mas  -- the standard schemes
                                                  (mean-based 1-D calibration)
  * robust_multiplicative                      -- the new scheme:
        median-bias calibration + MAD-based SE significance gate +
        multiplicative correction sigma_corrected^2 = sigma_hat^2 / (1 + g(CNR)).

Goal: confirm the new scheme reproduces the existing
``mean(D_hat)/D_true in [0.95, 1.05]`` result on Gaussian, Voigt, and
generalized-Gaussian profiles in the poster headline regime. Both
calibrations are built from the SAME Monte Carlo ensembles -- the only
difference is how the per-cell bias and weight are summarised.

Run from the repo root::

    .venv/Scripts/python.exe scripts/dense_calibration_and_test_robust.py
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighting_wall import MODELS, Experiment, SCHEMES  # noqa: E402
from weighting_wall.fit import fit_D  # noqa: E402
from weighting_wall.montecarlo import run_ensemble  # noqa: E402
from weighting_wall.truncation import TruncationModel, frame_stats  # noqa: E402
from weighting_wall.weights import msd_and_weights  # noqa: E402


# --------------------------------------------------------------------------
# Configuration -- matches dense_calibration_and_test.py for direct comparison
# --------------------------------------------------------------------------
D_TRUE = 0.02
CNR0_RANGE = (2.0, 100.0)
N_CALIB_POINTS = 100
N_TEST_POINTS = 100
N_RUNS_CALIB = 1000
N_RUNS_TEST = 1000
GATE_K = 2.0           # significance gate for robust scheme

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# --------------------------------------------------------------------------
# Robust calibration helpers
# --------------------------------------------------------------------------
def _robust_per_frame(ens, true_variance):
    """Compute per-frame (cnr, median_bias, median_SE, rel_variance).

    median_bias = median(sigma_hat^2) - sigma^2_true
    median_SE   = 1.253 * 1.4826 * MAD / sqrt(n)  -- robust SE of the median
    rel_variance = Var(MSD)  -- empirically measured per frame,
                                MSD = sigma_hat^2_f - sigma_hat^2_0
    """
    n_frames = ens.exp.n_frames
    cnr = ens.cnr_frame
    median_bias = np.full(n_frames, np.nan)
    median_se = np.full(n_frames, np.nan)
    rel_var = np.full(n_frames, np.nan)

    var0_col = ens.variance[:, 0]
    for f in range(n_frames):
        vcol = ens.variance[:, f]
        finite = np.isfinite(vcol)
        v = vcol[finite]
        if v.size < 2:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        robust_std = 1.4826 * mad
        median_bias[f] = med - float(true_variance[f])
        median_se[f] = 1.253 * robust_std / np.sqrt(v.size)
        # rel_variance: empirical Var(MSD_f)
        msd_col = ens.variance[:, f] - var0_col
        msd_clean = msd_col[np.isfinite(msd_col)]
        if msd_clean.size > 1:
            rel_var[f] = float(np.var(msd_clean, ddof=1))
    return cnr, median_bias, median_se, rel_var


def _robust_msd_and_weights(variance_row, cnr_frame, cal, sigma0, gate_k=GATE_K):
    """MSD + weights for the robust multiplicative scheme.

    cal is the tuple (cnr_grid, bias_grid, se_grid, rv_grid) -- the 1-D
    lookups built from the calibration ensemble. Multiplicative correction
    with a significance gate; inverse-variance weighting from the empirical
    rel_variance.
    """
    cnr_grid, bias_grid, se_grid, rv_grid = cal
    bias = np.interp(cnr_frame, cnr_grid, bias_grid,
                      left=bias_grid[0], right=bias_grid[-1])
    se = np.interp(cnr_frame, cnr_grid, se_grid,
                    left=se_grid[0], right=se_grid[-1])
    resolved = np.abs(bias) > gate_k * se
    bias_gated = np.where(resolved, bias, 0.0)
    g = np.clip(bias_gated, 0.0, None) / sigma0 ** 2
    corrected = variance_row / (1.0 + g)
    msd = corrected - corrected[0]

    rv = np.interp(cnr_frame, cnr_grid, rv_grid,
                    left=rv_grid[0], right=rv_grid[-1])
    hi = np.nanmax(rv)
    tiny = hi * 1e-12 if np.isfinite(hi) and hi > 0 else 1e-12
    w = 1.0 / np.clip(rv, tiny, None)
    w = np.asarray(w, float).copy()
    w[0] = 0.0
    return msd, w


# --------------------------------------------------------------------------
# Workers (top-level for pickling)
# --------------------------------------------------------------------------
def _calib_one(args):
    """Calibration: return BOTH FrameStats (mean) and robust tuple."""
    cnr0, exp, n_runs, seed, model_name = args
    model = MODELS[model_name]()
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed,
                       model=model)
    fs = frame_stats(ens)
    rb = _robust_per_frame(ens, exp.true_variance)
    return fs, rb


def _test_one(args):
    """Test: evaluate all 5 schemes (4 standard + robust) at one CNR_0."""
    cnr0, exp, n_runs, seed, model_name, trunc, robust_cal = args
    model = MODELS[model_name]()
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed,
                       model=model)
    out = {}
    times = exp.times

    # Standard four
    for scheme in SCHEMES:
        ratios = []
        for r in range(ens.n_runs):
            msd, w = msd_and_weights(scheme, ens.variance[r],
                                     ens.cnr_frame, trunc)
            d = fit_D(times, msd, w)
            if d.ok and np.isfinite(d.D):
                ratios.append(d.D / exp.D)
        ratios = np.asarray(ratios)
        if ratios.size:
            out[scheme] = (float(ratios.mean()),
                           float(np.median(ratios)),
                           float(ratios.std()))
        else:
            out[scheme] = (np.nan, np.nan, np.nan)

    # New robust scheme
    ratios = []
    for r in range(ens.n_runs):
        msd, w = _robust_msd_and_weights(ens.variance[r], ens.cnr_frame,
                                          robust_cal, exp.sigma0)
        d = fit_D(times, msd, w)
        if d.ok and np.isfinite(d.D):
            ratios.append(d.D / exp.D)
    ratios = np.asarray(ratios)
    if ratios.size:
        out["robust"] = (float(ratios.mean()),
                         float(np.median(ratios)),
                         float(ratios.std()))
    else:
        out["robust"] = (np.nan, np.nan, np.nan)
    return cnr0, out


# --------------------------------------------------------------------------
# Parallel orchestrator with progress reporting
# --------------------------------------------------------------------------
def _parallel_with_progress(worker, args_list, label):
    n = len(args_list)
    milestones = max(1, n // 10)
    done = 0
    results = [None] * n
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, args): i for i, args in enumerate(args_list)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if done % milestones == 0 or done == n:
                pct = 100.0 * done / n
                dt = time.time() - t0
                eta = dt * (n - done) / max(done, 1)
                print(f"    [{label}] {done}/{n} ({pct:3.0f}%)  "
                      f"elapsed {dt:5.0f}s  ETA {eta:4.0f}s", flush=True)
    return results


def _build_robust_lookup(robust_results):
    """Concatenate per-CNR_0 robust per-frame tuples into 1-D sorted lookups."""
    cnr = np.concatenate([r[0] for r in robust_results])
    bias = np.concatenate([r[1] for r in robust_results])
    se = np.concatenate([r[2] for r in robust_results])
    rv = np.concatenate([r[3] for r in robust_results])
    good = (np.isfinite(cnr) & np.isfinite(bias) & np.isfinite(se)
            & np.isfinite(rv) & (cnr > 0))
    cnr, bias, se, rv = cnr[good], bias[good], se[good], rv[good]
    order = np.argsort(cnr)
    return cnr[order], bias[order], se[order], rv[order]


SCHEMES_REPORT = SCHEMES + ("robust",)


# --------------------------------------------------------------------------
# Per-shape driver
# --------------------------------------------------------------------------
def run_shape(model_name: str, rng: np.random.Generator) -> None:
    print(f"\n{'=' * 70}\n=== {model_name}  (n_workers = {N_WORKERS}) ===\n"
          f"{'=' * 70}")

    cal_exp = Experiment(D=0.0)
    test_exp = Experiment(D=D_TRUE)
    cnr0_calib = np.logspace(np.log10(CNR0_RANGE[0]),
                              np.log10(CNR0_RANGE[1]),
                              N_CALIB_POINTS)

    # ---------- Calibration (yields BOTH mean and robust lookups) ----------
    print(f"\n  Calibration: {N_CALIB_POINTS} log-spaced CNR_0 in "
          f"[{CNR0_RANGE[0]}, {CNR0_RANGE[1]}], {N_RUNS_CALIB} runs/cell")
    cal_args = [(float(c), cal_exp, N_RUNS_CALIB, 1000 + i, model_name)
                for i, c in enumerate(cnr0_calib)]
    t0 = time.time()
    cal_results = _parallel_with_progress(_calib_one, cal_args, "calib")
    stats_list = [r[0] for r in cal_results]
    robust_per_cell = [r[1] for r in cal_results]
    trunc = TruncationModel.from_stats(stats_list)
    robust_cal = _build_robust_lookup(robust_per_cell)
    cal_time = time.time() - t0
    n_resolved = int(np.sum(np.abs(robust_cal[1]) > GATE_K * robust_cal[2]))
    print(f"  Calibration done [{cal_time:.0f}s]")
    print(f"  Mean lookup spans per-frame CNR {trunc.cnr_grid[0]:.3f} -> "
          f"{trunc.cnr_grid[-1]:.1f} ({trunc.cnr_grid.size} points)")
    print(f"  Robust lookup: {robust_cal[0].size} points; "
          f"{n_resolved} cells with resolved bias (|bias| > {GATE_K}*SE)")

    # ---------- Test ----------
    log_lo, log_hi = np.log10(CNR0_RANGE[0]), np.log10(CNR0_RANGE[1])
    cnr0_test = 10.0 ** rng.uniform(log_lo, log_hi, N_TEST_POINTS)
    cnr0_test.sort()
    print(f"\n  Test: {N_TEST_POINTS} random log-distributed CNR_0 in "
          f"[{CNR0_RANGE[0]}, {CNR0_RANGE[1]}], {N_RUNS_TEST} runs/cell")
    test_args = [(float(c), test_exp, N_RUNS_TEST, 50000 + i, model_name,
                  trunc, robust_cal)
                 for i, c in enumerate(cnr0_test)]
    t0 = time.time()
    test_out = _parallel_with_progress(_test_one, test_args, "test")
    test_time = time.time() - t0
    print(f"  Test done [{test_time:.0f}s]")

    # ---------- Report ----------
    test_cnr0 = np.array([c for c, _ in test_out])
    log_edges = np.linspace(log_lo, log_hi, 11)
    bin_idx = np.clip(np.searchsorted(log_edges[1:-1], np.log10(test_cnr0)),
                      0, 9)

    print(f"\n  Decile report (mean(D_hat) / D_true, averaged within each "
          f"log-CNR_0 decile):")
    header = (f"  {'CNR_0 range':<22}{'n':>4}  " +
              "".join(f"{s[:10]:>11}" for s in SCHEMES_REPORT))
    print(header)
    for b in range(10):
        mask = bin_idx == b
        if not mask.any():
            continue
        edge_lo, edge_hi = 10 ** log_edges[b], 10 ** log_edges[b + 1]
        n_in = int(mask.sum())
        means = {}
        for scheme in SCHEMES_REPORT:
            vals = np.array([test_out[i][1][scheme][0]
                             for i in range(N_TEST_POINTS) if mask[i]])
            means[scheme] = np.nanmean(vals)
        line = f"  [{edge_lo:5.2f}, {edge_hi:5.2f}]  {n_in:>4}  "
        for s in SCHEMES_REPORT:
            line += f"{means[s]:>11.3f}"
        print(line)

    print(f"\n  Overall mean |mean(D_hat)/D_true - 1| across all test points:")
    for scheme in SCHEMES_REPORT:
        vals = np.array([test_out[i][1][scheme][0]
                         for i in range(N_TEST_POINTS)])
        v = np.abs(vals - 1.0)
        v = v[np.isfinite(v)]
        print(f"    {scheme:>22}: {v.mean():.3f}  (median {np.median(v):.3f})")


def main() -> None:
    print(f"Dense parallel D-free calibration and test (with robust scheme)")
    print(f"  D_true = {D_TRUE} sigma_0^2/tau   (L_D/FWHM_0 ~ "
          f"{np.sqrt(2 * D_TRUE) / 2.355:.3f})")
    print(f"  gate_k = {GATE_K}  (robust scheme significance threshold)")
    print(f"  workers = {N_WORKERS}")
    rng = np.random.default_rng(20260601)
    t0 = time.time()
    for name in ("gaussian", "voigt", "generalized_gaussian"):
        run_shape(name, rng)
    print(f"\n[total wall time: {(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
