"""Render the per-frame poster figure for this experiment set.

The exact recipe that produced the committed ``br_halide_highn_perframe.png`` in
this folder. A fast, figure-only path: it reuses the cached D-aware calibrations
and the per-frame machinery from ``br_halide_highn_perframe`` to compute the three
CNR0 panels (with a lighter seed count than the full
``br_halide_highn_perframe.main`` report run), then calls ``make_figure``. No
Monte Carlo pool (the calibrations are cached in ``../output``), so it is cheap and
safe to re-run; it writes the figure straight into this folder.

    .venv/Scripts/python.exe highn_brI_fit_bound/render_perframe_figure.py
"""
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from pathlib import Path

from br_halide_highn_perframe import (CNR0, FLOOR_CNR, daware_calibration,
                                      make_figure, per_frame_means)
from br_halide_highn_validation import STRIDE, subsample
from br_halide_validation import fit_msd_powerlaw
from br_highn_data import load_all

N_SEEDS_FIG = 300
OUT = Path(HERE) / "br_halide_highn_perframe.png"


def main():
    cases = [subsample(c, STRIDE) for c in load_all()]
    panels = []
    for case in cases:
        d_true, _ = fit_msd_powerlaw(case.t, case.var_true)
        trunc = daware_calibration(d_true, case.name.split("=")[-1],
                                   case.profiles.shape[0])
        mraw, mcor, cnrf = per_frame_means(case, trunc, CNR0, N_SEEDS_FIG)
        use = cnrf >= FLOOR_CNR
        d_n, _ = fit_msd_powerlaw(case.t, mraw, mask=use)
        d_c, _ = fit_msd_powerlaw(case.t, mcor, mask=use)
        panels.append((case, cnrf, mraw, mcor, use, d_true, d_n, d_c))
        print(f"{case.name}: D_naive/true={d_n/d_true:.2f} D_corr/true={d_c/d_true:.2f}")
    make_figure(panels, OUT)


if __name__ == "__main__":
    main()
