"""Per-profile floor of the folded-normal correction.

The cleanest test of where the correction works: isolated single profiles
at chosen per-profile CNR, with no diffusion, no decay, no frame indexing.
For each test CNR:

1. Build a dense D-free calibration of var_bias(CNR) (matching
   dense_calibration_and_test.py: 100 log-spaced CNR_0 in [2,100], 1000
   runs/cell).
2. Simulate isolated noisy profiles at sigma^2 = sigma_0^2 and the chosen
   per-profile CNR. Many of them (n_runs_test per cell). Fit each.
3. Apply the calibration's var_bias(CNR_test) to get the corrected
   per-profile variance. Measure the residual mean
   E[sigma_tilde^2] - sigma^2_true with its MC standard error.
4. The floor: walking down from high CNR, the deepest CNR reached before
   the first cell whose residual exceeds both 2 * SE and 0.01 sigma_0^2.

If the per-profile correction holds at CNR_test, the per-frame correction
in any application (a diffusion experiment, real data, whatever) holds at
that CNR -- because the correction operates per profile. The diffusion
experiment's slope fit then inherits the per-frame unbiasedness as a
linear consequence.

A residual significantly different from zero says the calibration's mean
estimate at that CNR is off by more than MC noise: either the calibration's
own MC budget there is too thin, or the folded-normal model has broken.
Either way it's the actionable floor with this calibration.

All variances and standard errors in sigma_0^2.

Usage::

    .venv/Scripts/python.exe experiments/correction-floor/per_profile_floor.py [MODEL]

MODEL is ``gaussian``, ``voigt``, or ``generalized_gaussian:<p>``.
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import Experiment, model_from_spec, spec_tag  # noqa: E402
from folded_debias.calibration import _single_frame_profile  # noqa: E402
from folded_debias.montecarlo import run_ensemble  # noqa: E402
from folded_debias.truncation import TruncationModel, frame_stats  # noqa: E402


# --- Calibration (matches the dense run) -----------------------------------
CAL_CNR0_RANGE = (2.0, 100.0)
N_CAL_POINTS = 100
N_RUNS_CAL = 5000

# --- Test sweep ------------------------------------------------------------
TEST_CNR_RANGE = (0.05, 100.0)
N_TEST_POINTS = 30
N_RUNS_TEST = 50_000              # isolated-profile fits per test CNR
AMP = 1.0                          # peak amplitude convention for the test
SIGMA0 = 1.0                       # all variances in sigma_0^2 units

RESID_TOL = 0.01                   # sigma_0^2: a residual must exceed this
                                   # (1% of sigma^2) as well as 2 SE to fail

N_WORKERS = max(1, (os.cpu_count() or 4) - 1)


# --- Workers ---------------------------------------------------------------
def _calib_one(args):
    cnr0, exp, n_runs, seed, model_name = args
    model = model_from_spec(model_name)
    ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed, model=model)
    return frame_stats(ens)


def _test_one(args):
    """Isolated single-profile MC at a chosen (CNR, sigma^2=1).

    Returns (cnr, mean_raw, mean_corrected, residual, residual_se, n_used,
    sd_raw).
    """
    cnr_test, n_runs, seed, model_name, pixel_x, var_bias_lookup_cnr, var_bias_lookup_vals = args
    model = model_from_spec(model_name)
    clean = _single_frame_profile(model, pixel_x, SIGMA0 ** 2, AMP)
    noise_sigma = AMP / cnr_test
    rng = np.random.default_rng(seed)
    fits = np.empty(n_runs)
    for r in range(n_runs):
        noise = rng.normal(0.0, noise_sigma, size=clean.shape)
        ff = model.fit_variance(pixel_x, clean + noise)
        fits[r] = ff.variance
    good = np.isfinite(fits)
    n_used = int(good.sum())
    if n_used < 2:
        return (cnr_test, np.nan, np.nan, np.nan, np.nan, n_used, np.nan)
    raw = fits[good]
    mean_raw = float(raw.mean())
    sd_raw = float(raw.std(ddof=1))
    se_raw = sd_raw / np.sqrt(n_used)
    # Apply the calibration's var_bias(CNR_test) via np.interp on the lookup
    bias = float(np.interp(cnr_test, var_bias_lookup_cnr,
                           var_bias_lookup_vals,
                           left=var_bias_lookup_vals[0],
                           right=var_bias_lookup_vals[-1]))
    corrected = raw - bias
    mean_corr = float(corrected.mean())
    residual = mean_corr - SIGMA0 ** 2
    # residual SE = SE of (corrected mean) = se_raw (since bias is a constant
    # subtraction)
    residual_se = se_raw
    return (cnr_test, mean_raw, mean_corr, residual, residual_se, n_used,
            sd_raw)


# --- Driver ---------------------------------------------------------------
def parallel_with_progress(worker, args_list, label):
    n = len(args_list)
    milestones = max(1, n // 10)
    done = 0
    results = [None] * n
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(worker, args): i
                   for i, args in enumerate(args_list)}
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


def build_dense_calibration(model_name):
    cal_exp = Experiment(D=0.0)
    cnr0_grid = np.logspace(np.log10(CAL_CNR0_RANGE[0]),
                             np.log10(CAL_CNR0_RANGE[1]),
                             N_CAL_POINTS)
    args = [(float(c), cal_exp, N_RUNS_CAL, 1000 + i, model_name)
            for i, c in enumerate(cnr0_grid)]
    print(f"  Dense calibration: {N_CAL_POINTS} CNR_0 in {CAL_CNR0_RANGE}, "
          f"{N_RUNS_CAL} runs/cell ({N_WORKERS} workers)", flush=True)
    stats = parallel_with_progress(_calib_one, args, "cal")
    return TruncationModel.from_stats(stats)


def floor_sweep(model_name, trunc):
    cnr_test_grid = np.logspace(np.log10(TEST_CNR_RANGE[0]),
                                 np.log10(TEST_CNR_RANGE[1]),
                                 N_TEST_POINTS)
    # Use the standard pixel grid for the single-profile MC.
    pixel_x = Experiment().x
    # Pass the lookup arrays so workers can interp without needing the
    # TruncationModel object pickled (also fine to pickle, but arrays are
    # cleaner).
    vb_cnr = trunc.cnr_grid
    vb_vals = trunc.varbias_grid
    args = [(float(c), N_RUNS_TEST, 50000 + i, model_name, pixel_x,
             vb_cnr, vb_vals)
            for i, c in enumerate(cnr_test_grid)]
    print(f"  Test sweep: {N_TEST_POINTS} CNR in {TEST_CNR_RANGE}, "
          f"{N_RUNS_TEST} runs/cell ({N_WORKERS} workers)", flush=True)
    return parallel_with_progress(_test_one, args, "test")


def report(model_name, results):
    print(f"\n=== {model_name}: per-profile floor ===")
    print(f"  {'CNR':>8}{'n':>10}{'E[raw]':>10}{'E[corr]':>10}"
          f"{'resid':>11}{'SE':>9}{'|z|':>7}{'flag':>6}")
    # Sort by CNR for clean display.
    results_sorted = sorted([r for r in results if r is not None],
                            key=lambda r: r[0])

    def fails(residual, residual_se):
        # A cell fails only if its residual is both resolved (|z| > 2) and
        # larger than RESID_TOL. With 50,000 fits the high-CNR SE is ~1e-4
        # sigma_0^2, so significance alone flags the calibration's own
        # interpolation offsets of a few 1e-4 and puts the "floor" at the top
        # of the grid.
        z = abs(residual / residual_se) if residual_se > 0 else np.inf
        return z > 2 and abs(residual) > RESID_TOL

    # Walk from high CNR downward; the floor is the deepest CNR reached before
    # the first failing cell.
    floor_cnr = None
    for cnr, mean_raw, mean_corr, residual, residual_se, n_used, sd in results_sorted[::-1]:
        if fails(residual, residual_se):
            break
        floor_cnr = cnr
    # Now print in ascending CNR
    for cnr, mean_raw, mean_corr, residual, residual_se, n_used, sd in results_sorted:
        z = abs(residual / residual_se) if residual_se > 0 else np.inf
        flag = "<<<" if fails(residual, residual_se) else ""
        print(f"  {cnr:>8.3f}{n_used:>10d}{mean_raw:>10.4f}"
              f"{mean_corr:>10.4f}{residual:>+11.4f}"
              f"{residual_se:>9.4f}{z:>7.2f}{flag:>6}")
    print(f"\n  >>> Per-profile floor (deepest CNR before the first cell with "
          f"|z| > 2 and |residual| > {RESID_TOL} sigma_0^2): CNR ~ {floor_cnr:.3f}")


def rescore(path: str) -> None:
    """Append the floor under the current rule to a committed table.

    Tables written before the floor rule required |residual| > RESID_TOL
    print a significance-only readout. This re-reads the table's rows (the
    residuals and SEs are unchanged) and appends the floor the current rule
    gives, labelled as re-scored, so no multi-hour sweep has to be repeated.
    """
    import re
    text = Path(path).read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if (len(parts) >= 7 and re.match(r"^\d+\.\d+$", parts[0])
                and parts[1].isdigit()):
            cnr, n, raw, corr, res, se = (float(parts[0]), int(parts[1]),
                                          float(parts[2]), float(parts[3]),
                                          float(parts[4]), float(parts[5]))
            rows.append((cnr, raw, corr, res, se, n, 0.0))
    floor = None
    for cnr, _, _, res, se, _, _ in sorted(rows)[::-1]:
        z = abs(res / se) if se > 0 else np.inf
        if z > 2 and abs(res) > RESID_TOL:
            break
        floor = cnr
    note = (f"\n  >>> Re-scored from this table with the current rule (|z| > 2 and "
            f"|residual| > {RESID_TOL} sigma_0^2): per-profile floor CNR ~ {floor:.3f}\n")
    if "Re-scored from this table" not in text:
        Path(path).write_text(text.rstrip("\n") + "\n" + note, encoding="utf-8")
    print(path, note.strip())


def main() -> None:
    if len(sys.argv) > 2 and sys.argv[1] == "--rescore":
        for path in sys.argv[2:]:
            rescore(path)
        return
    # model_name is a spec: "generalized_gaussian:<p>" sets the shape, and the
    # bare "generalized_gaussian" is p = 2 (a Gaussian profile).
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    try:
        model_from_spec(model_name)
    except ValueError as err:
        sys.exit(str(err))
    print(f"Per-profile floor sweep ({spec_tag(model_name)})")
    print(f"  variances in sigma_0^2; tests at sigma^2 = {SIGMA0**2}")
    t0 = time.time()
    trunc = build_dense_calibration(model_name)
    cal_dt = time.time() - t0
    print(f"  Calibration done [{cal_dt:.0f}s]")
    print()
    t1 = time.time()
    results = floor_sweep(model_name, trunc)
    test_dt = time.time() - t1
    print(f"  Test sweep done [{test_dt:.0f}s]")
    report(model_name, results)
    print(f"\n[total {(time.time()-t0)/60:.1f} min]")


if __name__ == "__main__":
    main()
