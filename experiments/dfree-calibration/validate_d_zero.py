"""Validate the D=0 calibration: a 1-D lookup that does not reference the
application's D.

The simplest D-free calibration is the existing 1-D one, run at *D = 0*: the
calibration profiles all have variance = sigma_0^2, the lookup is keyed on
per-frame CNR alone, and the application's D never enters. The lookup
captures the bias at sigma^2 = sigma_0^2, which is the relevant region for
the low-D regime where the truncation bias bites (sigma^2(t) barely changes
across the measurement when D is small).

Procedure mirrors validate_d_free.py: calibrate once at D = 0, evaluate at
several D values, check the bias curves are flat across D.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from weighting_wall import MODELS, Experiment, calibrate, evaluate  # noqa: E402


CNR0_GRID = np.array([5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0])
D_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1)
N_RUNS = 300


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    if model_name not in MODELS:
        sys.exit(f"unknown model {model_name!r}; choose from {sorted(MODELS)}")
    pmodel = MODELS[model_name]()

    cal_exp = Experiment(D=0.0)
    t0 = time.time()
    print(f"D=0 calibration ({model_name}, {N_RUNS} runs/CNR0) ...")
    trunc = calibrate(cal_exp, CNR0_GRID, n_runs=N_RUNS, seed=1, model=pmodel)
    print(f"  done [{time.time() - t0:.0f} s]")

    header = "  D_true   " + "".join(f"CNR0={c:<5g}" for c in CNR0_GRID) \
             + "  mean|bias-1|"
    print("\n" + "=" * len(header))
    print(f"D=0 calibration applied to all D_true values (Gaussian baseline)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    fn_table = np.full((len(D_VALUES), CNR0_GRID.size), np.nan)
    for k, d_true in enumerate(D_VALUES):
        exp = Experiment(D=d_true)
        t1 = time.time()
        results = evaluate(exp, CNR0_GRID, trunc, n_runs=N_RUNS,
                           seed=20000 + k * 1000, profile_model=pmodel)
        fn = results["folded_normal"].mean_bias
        fn_table[k] = fn
        line = f"  {d_true:6.4f}  "
        for v in fn:
            line += f"{v:9.3f}"
        m = np.nanmean(np.abs(fn - 1.0))
        line += f"   {m:.3f}   [{time.time() - t1:.0f} s]"
        print(line)

    print("=" * len(header))
    overall = np.nanmean(np.abs(fn_table - 1.0))
    print(f"\nOverall mean |bias-1| across D x CNR: {overall:.3f}")
    print(f"\n[total: {time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
