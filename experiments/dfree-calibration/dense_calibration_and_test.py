"""Dense, parallel, D-free calibration and test.

What this does, per profile model:

1. Build a *dense* D-free calibration: 100 log-spaced CNR_0 values from 2 to
   100, with 1000 noisy realisations per cell. The lookup is the
   concatenation of all per-frame (CNR, var_bias) data from those sweeps,
   which puts substantially more data points at low per-frame CNR than the
   default 7-point sweep.

2. Evaluate the four schemes (ols, inverse_variance, folded_normal, mas)
   on *random* log-distributed CNR_0 values, also 1000 runs per cell. Random
   test CNR_0 values let us see the bias curve as a continuous function
   rather than at the seven calibration points.

3. Report mean(D_hat) / D_true binned in deciles of log-CNR_0, and the
   overall mean |mean(D_hat)/D_true - 1|, for each scheme.

Both phases parallelise across CNR_0 via ProcessPoolExecutor, with progress
reports every 10% completion.

Wall-time estimate: ~5-15 min for Gaussian, ~25-45 min for Voigt,
~5-15 min for generalized-Gaussian on a typical 8-core machine. Total
~50-90 min for all three shapes.

Usage::

    .venv/Scripts/python.exe experiments/dfree-calibration/dense_calibration_and_test.py [MODEL ...]

MODEL is ``gaussian``, ``voigt``, or ``generalized_gaussian:<p>``.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import Experiment, SCHEMES, model_from_spec  # noqa: E402
from folded_debias.fit import fit_D  # noqa: E402
from folded_debias.montecarlo import run_ensemble  # noqa: E402
from folded_debias.truncation import TruncationModel, frame_stats  # noqa: E402
from folded_debias.weights import msd_and_weights  # noqa: E402


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
D_TRUE = 0.02              # sigma_0^2 / tau; L_D / FWHM_0 ~ 0.085
CNR0_RANGE = (2.0, 100.0)  # log-spaced range
N_CALIB_POINTS = 100        # log-spaced calibration CNR_0 values
N_TEST_POINTS = 100         # random log-distributed test CNR_0 values
N_RUNS_CALIB = 5000
N_RUNS_TEST = 1000

# Worker count: leave 1 core for the OS
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# --------------------------------------------------------------------------
# Workers (top-level for pickling)
# --------------------------------------------------------------------------
def _calib_one(args):
    """Run a single-CNR_0 calibration ensemble and return the FrameStats."""
    cnr0, exp, n_runs, seed, model_name = args
    model = model_from_spec(model_name)
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model)
    return frame_stats(ens)


def _test_one(args):
    """Run a single-CNR_0 evaluation ensemble across all schemes."""
    cnr0, exp, n_runs, seed, model_name, trunc = args
    model = model_from_spec(model_name)
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model)
    out = {}
    times = exp.times
    for scheme in SCHEMES:
        ratios = []
        for r in range(ens.n_runs):
            msd, w = msd_and_weights(scheme, ens.variance[r], ens.cnr_frame, trunc)
            d = fit_D(times, msd, w)
            if d.ok and np.isfinite(d.D):
                ratios.append(d.D / exp.D)
        ratios = np.asarray(ratios)
        if ratios.size:
            out[scheme] = (float(ratios.mean()), float(np.median(ratios)),
                           float(ratios.std()))
        else:
            out[scheme] = (np.nan, np.nan, np.nan)
    return cnr0, out


# --------------------------------------------------------------------------
# Parallel orchestrators with progress reporting
# --------------------------------------------------------------------------
def _parallel_with_progress(worker, args_list, label):
    """Submit args_list to worker via ProcessPoolExecutor; report every 10%."""
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
                      f"elapsed {dt:5.0f}s  ETA {eta:4.0f}s",
                      flush=True)
    return results


# --------------------------------------------------------------------------
# Per-shape driver
# --------------------------------------------------------------------------
def run_shape(model_name: str, rng: np.random.Generator) -> None:
    print(f"\n{'=' * 70}\n=== {model_name}  (n_workers = {N_WORKERS}) ===\n"
          f"{'=' * 70}")

    cal_exp = Experiment(D=0.0)         # D-free calibration
    test_exp = Experiment(D=D_TRUE)
    cnr0_calib = np.logspace(np.log10(CNR0_RANGE[0]),
                              np.log10(CNR0_RANGE[1]),
                              N_CALIB_POINTS)

    # ---------- Calibration ----------
    print(f"\n  Calibration: {N_CALIB_POINTS} log-spaced CNR_0 in "
          f"[{CNR0_RANGE[0]}, {CNR0_RANGE[1]}], {N_RUNS_CALIB} runs/cell")
    cal_args = [(float(c), cal_exp, N_RUNS_CALIB, 1000 + i, model_name)
                for i, c in enumerate(cnr0_calib)]
    t0 = time.time()
    stats_list = _parallel_with_progress(_calib_one, cal_args, "calib")
    trunc = TruncationModel.from_stats(stats_list)
    cal_time = time.time() - t0
    print(f"  Calibration done [{cal_time:.0f}s]; "
          f"lookup spans per-frame CNR {trunc.cnr_grid[0]:.3f} -> "
          f"{trunc.cnr_grid[-1]:.1f} ({trunc.cnr_grid.size} data points)")

    # ---------- Test ----------
    # Random log-distributed test CNR_0 values in the same range.
    log_lo, log_hi = np.log10(CNR0_RANGE[0]), np.log10(CNR0_RANGE[1])
    cnr0_test = 10.0 ** rng.uniform(log_lo, log_hi, N_TEST_POINTS)
    cnr0_test.sort()  # for nicer reporting
    print(f"\n  Test: {N_TEST_POINTS} random log-distributed CNR_0 in "
          f"[{CNR0_RANGE[0]}, {CNR0_RANGE[1]}], {N_RUNS_TEST} runs/cell")
    test_args = [(float(c), test_exp, N_RUNS_TEST, 50000 + i, model_name, trunc)
                 for i, c in enumerate(cnr0_test)]
    t0 = time.time()
    test_out = _parallel_with_progress(_test_one, test_args, "test")
    test_time = time.time() - t0
    print(f"  Test done [{test_time:.0f}s]")

    # ---------- Report ----------
    # Bin the test CNR_0s into 10 log-spaced deciles for readability.
    test_cnr0 = np.array([c for c, _ in test_out])
    log_edges = np.linspace(log_lo, log_hi, 11)   # 10 bins
    bin_idx = np.clip(np.searchsorted(log_edges[1:-1], np.log10(test_cnr0)),
                      0, 9)

    print(f"\n  Decile report (mean(D_hat) / D_true, averaged within each "
          f"log-CNR_0 decile):")
    print(f"  {'CNR_0 range':<22}{'n':>4}  "
          f"{'OLS':>10}{'inv-var':>10}{'folded':>10}{'mas':>10}")
    for b in range(10):
        mask = bin_idx == b
        if not mask.any():
            continue
        edge_lo, edge_hi = 10 ** log_edges[b], 10 ** log_edges[b + 1]
        n_in = int(mask.sum())
        means = {}
        for scheme in SCHEMES:
            vals = np.array([test_out[i][1][scheme][0]
                             for i in range(N_TEST_POINTS) if mask[i]])
            means[scheme] = np.nanmean(vals)
        print(f"  [{edge_lo:5.2f}, {edge_hi:5.2f}]  {n_in:>4}  "
              f"{means['ols']:>10.3f}{means['inverse_variance']:>10.3f}"
              f"{means['folded_normal']:>10.3f}{means['mas']:>10.3f}")

    # Overall mean |bias - 1| across all test points
    print(f"\n  Overall mean |mean(D_hat)/D_true - 1| across all test points:")
    for scheme in SCHEMES:
        vals = np.array([test_out[i][1][scheme][0]
                         for i in range(N_TEST_POINTS)])
        v = np.abs(vals - 1.0)
        v = v[np.isfinite(v)]
        print(f"    {scheme:>20}: {v.mean():.3f}  (median {np.median(v):.3f})")


def main() -> None:
    print(f"Dense parallel D-free calibration and test")
    print(f"  D_true = {D_TRUE} sigma_0^2/tau   (L_D/FWHM_0 ~ "
          f"{np.sqrt(2 * D_TRUE) / 2.355:.3f})")
    print(f"  workers = {N_WORKERS}")
    rng = np.random.default_rng(20260521)
    t0 = time.time()
    # Optional model specs on the command line, e.g.
    # "generalized_gaussian:1.5 generalized_gaussian:4". The default list is
    # the committed run; its bare "generalized_gaussian" is p = 2, a Gaussian
    # profile. The test CNR_0 draws come from one shared generator, so a
    # different spec list draws different test points.
    specs = sys.argv[1:] or ["gaussian", "voigt", "generalized_gaussian"]
    for spec in specs:
        model_from_spec(spec)          # fail fast on a bad spec
    for name in specs:
        run_shape(name, rng)
    print(f"\n[total wall time: {(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
