"""Compare the 1-D slope weights: per-frame variance against the older MSD-point lookup.

The older weights read ``relvar_grid``, the MSD-point variance measured inside
each calibration ensemble. Its t = 0 term belongs to the calibration's CNR_0,
so the lookup is jagged, and every ensemble's t = 0 point reads exactly 0. The
newer weights read ``framevar_grid`` (per-frame Var(sigma_hat^2)) and form
Var(MSD_i) = v(CNR_i) + v(CNR_0) from the measurement's own frames.

Uses the cached calibrations and evaluation set of ``calibration_se_check.py``
(run its ``generate`` first). Per cell (folded_normal, exact per-frame CNR key,
frames below CNR 1 dropped) and per weighting it reports, averaged over the
calibration draws:

* the mean D/D_in, to show the weights leave the bias where it was;
* the spread of D/D_in over realizations (the precision the weights set);
* the spread of the cell mean over draws against the propagated calibration SE.

    .venv/Scripts/python.exe experiments/dfree-calibration/weights_check.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import Experiment, TruncationModel  # noqa: E402
from folded_debias.truncation import frame_stats  # noqa: E402
from folded_debias.weights import calibration_D_se, msd_and_weights  # noqa: E402

from calibration_se_check import (CNR0_VALUES, D_VALUES, EVAL_RUNS, FLOOR,  # noqa: E402
                                  N_CAL, OUT, _cal_path, _ensembles)


def cell(model: TruncationModel, var: np.ndarray, cnr: np.ndarray, d_true: float):
    """(mean, SD over realizations, calibration SE) of D/D_in in one cell.

    With a 1-D model and the exact key the weights are the same for every
    realization, so the weighted slope is one linear map applied to all rows.
    """
    times = Experiment(D=d_true).times
    _, w = msd_and_weights("folded_normal", var[0], cnr, model)
    w = np.where(cnr >= FLOOR, w, 0.0)
    good = w > 0
    if good.sum() < 2:
        return np.nan, np.nan, np.nan
    corrected = var - model.var_bias(cnr)
    msd = corrected - corrected[:, [0]]
    assert np.all(np.isfinite(msd[:, good]))
    tg, wg = times[good], w[good]
    sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
    c = wg * (sw * tg - swt) / (sw * swtt - swt ** 2)
    ratio = (msd[:, good] @ c) / 2.0 / d_true
    se = calibration_D_se(times, msd[0], w, cnr, model) / d_true
    return ratio.mean(), ratio.std(ddof=1), se


def main() -> None:
    ks = [k for k in range(N_CAL) if _cal_path(k).exists()]
    new = [TruncationModel.from_stats([frame_stats(e) for e in _ensembles(k)])
           for k in ks]
    variants = {"MSD-point lookup": [replace(m, framevar_grid=None) for m in new],
                "per-frame":        new}
    with np.load(OUT / f"eval_runs{EVAL_RUNS}.npz") as z:
        evar, ecnr = z["variance"], z["cnr"]
    nd, nc = len(D_VALUES), len(CNR0_VALUES)
    res = {}
    for name, models in variants.items():
        r = np.full((len(ks), nd, nc, 3), np.nan)
        for j, m in enumerate(models):
            for di, d in enumerate(D_VALUES):
                for ci in range(nc):
                    r[j, di, ci] = cell(m, evar[di, ci], ecnr[di, ci], d)
        res[name] = r
    np.savez(OUT / "weights_check.npz",
             **{k.replace(" ", "_").replace("-", "_"): v for k, v in res.items()})

    head = "          " + "".join(f"{c:>8g}" for c in CNR0_VALUES)
    old, cur = res["MSD-point lookup"], res["per-frame"]
    print(f"{len(ks)} calibration draws x {EVAL_RUNS} evaluation runs per cell; "
          "rows D, columns CNR_0")

    def table(title, arr, fmt="8.3f"):
        print(f"\n{title}\n{head}")
        for di, d in enumerate(D_VALUES):
            print(f"  {d:6.3f}  " + "".join(f"{v:{fmt}}" for v in arr[di]))

    for name, r in res.items():
        table(f"[{name}] mean D/D_in (mean over draws)", r[..., 0].mean(axis=0))
    for name, r in res.items():
        table(f"[{name}] SD of D/D_in over realizations (mean over draws)",
              r[..., 1].mean(axis=0))
    table("SD ratio, per-frame / MSD-point lookup (below 1 = more precise)",
          cur[..., 1].mean(axis=0) / old[..., 1].mean(axis=0))
    # Shift of the mean, in units of the combined SE of the per-frame arm.
    ev = cur[..., 1].mean(axis=0) / np.sqrt(EVAL_RUNS)
    comb = np.hypot(ev, np.sqrt(np.mean(cur[..., 2] ** 2, axis=0)))
    table("Mean shift, (per-frame - MSD-point) / combined SE, mean over draws",
          (cur[..., 0] - old[..., 0]).mean(axis=0) / comb, "8.2f")
    for name, r in res.items():
        obs = r[..., 0].var(axis=0, ddof=1) / np.mean(r[..., 2] ** 2, axis=0)
        print(f"\n[{name}] cell-mean spread over draws / predicted calibration "
              f"variance: mean {np.nanmean(obs):.2f}; per CNR_0 "
              + " ".join(f"{v:.2f}" for v in np.nanmean(obs, axis=0)))


if __name__ == "__main__":
    main()
