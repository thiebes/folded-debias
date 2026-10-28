"""Decompose the criterion-3 result: where does the extra calibration scatter come from?

Run after ``compare_calibrations.py generate`` (it reads the same cache). For each
shape it reports the observed / predicted calibration variance of D (frames with
per-frame CNR >= 1, Fisher arm, all Fisher draws) under three conditions:

* ``as run``         -- each draw's own model, as compare_calibrations scores it;
* ``fixed weights``  -- slope weights from one reference model (the Fisher cells
  pooled over every draw), so only the bias curve varies between draws;
* ``fixed lambda``   -- fixed weights, and the ratio smooth refitted at one
  smoothing parameter for every draw (the geometric median of the per-draw GCV
  choices) instead of choosing it by GCV per draw.

``calibration_D_se`` propagates the covariance of the bias curve at the chosen
smoothing parameter, so it leaves out the scatter that the per-draw choice
itself adds. If the ratio is near 1 at fixed lambda but not as run, that choice
is the missing term. The fixed-lambda refit replicates
``TruncationModel.from_cells``; the script checks that the replica reproduces the
library at the GCV choice before using it.

    .venv/Scripts/python.exe experiments/fisher-calibration/diagnose_criterion3.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import (CalibrationCells, Experiment, TruncationModel,  # noqa: E402
                           leading_order_terms, model_from_spec)
from folded_debias.smoothing import pspline_fit  # noqa: E402
from folded_debias.weights import calibration_D_se, msd_and_weights  # noqa: E402

from compare_calibrations import (CNR0_VALUES, D_VALUES, FLOOR, N_FISHER_DRAWS,  # noqa: E402
                                  OUT, SPECS, _cells_path, _eval_path)
from fisher_check import _pooled_cells  # noqa: E402


def _ratio_model(cells: CalibrationCells, terms, lam: float | None = None,
                 knee_cnr: float = 6.0) -> tuple[TruncationModel, float]:
    """The bias curve of ``TruncationModel.from_cells``, optionally at a fixed lambda."""
    good = (np.isfinite(cells.mean_bias) & np.isfinite(cells.bias_sd)
            & (cells.n_finite > 1))
    cnr = cells.cnr[good]
    u = np.log(cnr)
    n_fits = cells.n_finite[good]
    leading = terms.bias_coef / cnr ** 2
    framevar_ratio = cells.bias_sd[good] ** 2 / (terms.framevar_coef / cnr ** 2)
    pilot = np.exp(pspline_fit(u, np.log(framevar_ratio), np.ones_like(u)).evaluate(u))
    spread = pspline_fit(u, framevar_ratio, pilot * np.sqrt(2.0 / (n_fits - 1)))
    cell_var = np.maximum(spread.evaluate(u), pilot) * terms.framevar_coef / cnr ** 2
    ratio_se = np.sqrt(cell_var / n_fits) / leading
    fit = pspline_fit(u, cells.mean_bias[good] / leading - 1.0, ratio_se,
                      knee=np.log(knee_cnr), intercept=False,
                      lam_grid=None if lam is None else np.array([lam]))
    grid = np.geomspace(cnr.min(), 1e3, 256)
    grid_leading = terms.bias_coef / grid ** 2
    design = fit.design(np.log(grid))
    eigval, eigvec = np.linalg.eigh(fit.coef_cov)
    factor = grid_leading[:, None] * (design @ (eigvec * np.sqrt(np.clip(eigval, 0.0, None))))
    model = TruncationModel(cnr_grid=grid, p_grid=np.zeros_like(grid),
                            relvar_grid=np.ones_like(grid),
                            varbias_grid=grid_leading * (1.0 + design @ fit.coef),
                            varbias_se_grid=np.linalg.norm(factor, axis=1),
                            varbias_cov_factor=factor)
    return model, fit.lam


def _cell(bias_model, weight_model, var, cnr, d_true):
    """Mean D/D_ref over the cell's realizations and its predicted calibration SE."""
    times = Experiment(D=d_true).times
    _, w = msd_and_weights("folded_normal", var[0], cnr, weight_model)
    w = np.where(cnr >= FLOOR, w, 0.0)
    corrected = var - bias_model.var_bias(np.broadcast_to(cnr, var.shape).ravel()).reshape(var.shape)
    msd = corrected - corrected[:, :1]
    good = (w > 0) & np.all(np.isfinite(msd), axis=0)
    tg, wg = times[good], w[good]
    sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
    c = wg * (sw * tg - swt) / (sw * swtt - swt ** 2)
    mean_ratio = float(np.mean(msd[:, good] @ c) / 2.0 / d_true)
    return mean_ratio, calibration_D_se(times, msd[0], w, cnr, bias_model) / d_true


def _obs_over_pred(pairs, weight_models, var, cnr) -> np.ndarray:
    """Observed / predicted calibration variance per (D, CNR_0) cell."""
    s = np.array([[[_cell(bias, weights, var[di, ci], cnr[di, ci], d)
                    for ci in range(len(CNR0_VALUES))] for di, d in enumerate(D_VALUES)]
                  for bias, weights in zip(pairs, weight_models)])
    return s[..., 0].var(axis=0, ddof=1) / s[..., 1].mean(axis=0) ** 2


def main() -> None:
    x = Experiment().x
    lines = ["Criterion 3 decomposition: observed / predicted calibration variance of D, "
             f"Fisher arm, frames with CNR >= {FLOOR}, {N_FISHER_DRAWS} draws",
             "columns: mean over the 35 cells, then the mean over D at each CNR_0 "
             + " ".join(f"{c:g}" for c in CNR0_VALUES), ""]
    for spec in SPECS:
        model = model_from_spec(spec)
        terms = leading_order_terms(model, x, 1.0)
        draws = [CalibrationCells.load(_cells_path(spec, k)) for k in range(N_FISHER_DRAWS)]
        library = [TruncationModel.from_cells(c, terms) for c in draws]
        replica = [_ratio_model(c, terms) for c in draws]
        assert all(np.allclose(m.varbias_grid, r.varbias_grid, rtol=1e-10, atol=0.0)
                   and np.allclose(m.varbias_cov_factor, r.varbias_cov_factor, rtol=1e-8, atol=1e-14)
                   for m, (r, _) in zip(library, replica)), "replica differs from from_cells"
        lams = np.array([lam for _, lam in replica])
        lam_fixed = float(np.exp(np.median(np.log(lams))))
        fixed = [_ratio_model(c, terms, lam_fixed)[0] for c in draws]
        reference = TruncationModel.from_cells(_pooled_cells(spec), terms)
        with np.load(_eval_path(spec)) as z:
            var, cnr = z["variance"], z["cnr"]
        arms = {
            "as run": _obs_over_pred(library, library, var, cnr),
            "fixed weights": _obs_over_pred(library, [reference] * len(draws), var, cnr),
            "fixed lambda": _obs_over_pred(fixed, [reference] * len(draws), var, cnr),
        }
        lines.append(f"=== {spec}: GCV lambda over draws, 5th to 95th percentile "
                     f"{np.quantile(lams, 0.05):.3g} to {np.quantile(lams, 0.95):.3g}; "
                     f"fixed lambda {lam_fixed:.3g}")
        for name, ratio in arms.items():
            lines.append(f"  {name:14s} {ratio.mean():6.2f}   |"
                         + "".join(f"{r:6.2f}" for r in ratio.mean(axis=0)))
        lines.append("")
        print("\n".join(lines[-5:]), flush=True)
    (OUT / "diagnose_criterion3.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
