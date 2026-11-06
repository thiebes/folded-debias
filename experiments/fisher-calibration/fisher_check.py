"""Check the leading-order Fisher term and the width scaling of the bias.

Part 1 (no new Monte Carlo): the Fisher bias scale A = CNR^2 Var(sigma_hat) at
sigma_0 against the Monte Carlo spread of the Fisher calibration cells that
``compare_calibrations.py generate`` cached (A_MC = CNR^2 Var(sigma_hat^2) / 4
at CNR >= 8, where the fit is efficient), for all four shapes.

Part 2: does the bias at a true variance v follow the width table? ``generate``
runs single-frame cells at v / sigma_0^2 in ``TEST_VARIANCES`` (Gaussian and
Voigt, more runs below CNR 3). ``analyse`` compares each cell's mean bias with
three predictions built from the sigma_0 calibration (the cells of every
compare_calibrations draw, pooled):

* ``none``  -- the sigma_0 curve, unscaled (what the D-free lookup subtracts);
* ``lead``  -- the sigma_0 curve times the leading-order width ratio r(v);
* ``table`` -- the width-scaled lookup, q * b_cal(CNR sqrt(q / r)).

It reports chi^2 / dof per width over cells with CNR >= 1 and over all cells,
and the ratio of measured bias to the leading term against the relative
precision variable S = CNR sqrt(v / A(v)).

    $env:WIDTH_DEBIAS_WORKERS = 0
    .venv/Scripts/python.exe experiments/fisher-calibration/fisher_check.py generate
    .venv/Scripts/python.exe experiments/fisher-calibration/fisher_check.py analyse
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import (CalibrationCells, Experiment, BiasLookup,  # noqa: E402
                           calibrate_cells, leading_order_terms, model_from_spec,
                           spec_tag)
from width_debias.bias_lookup import _interp_matrix  # noqa: E402

from compare_calibrations import (N_FISHER_DRAWS, OUT, SPECS, WIDTH_GRID,  # noqa: E402
                                  _cells_path, _seed)

WIDTH_SPECS = ("gaussian", "voigt")
TEST_VARIANCES = (0.25, 0.5, 1.0, 2.0, 4.0, 9.0)
TEST_CNR = np.geomspace(0.5, 20.0, 16)
TEST_RUNS = np.where(TEST_CNR < 3.0, 2000, 500)


def _width_path(spec: str, variance: float):
    return OUT / f"width_{spec_tag(spec)}_v{variance:g}.npz"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    x = Experiment().x
    for s, spec in enumerate(WIDTH_SPECS):
        model = model_from_spec(spec)
        for vi, v in enumerate(TEST_VARIANCES):
            if _width_path(spec, v).exists():
                continue
            t0 = time.time()
            calibrate_cells(TEST_CNR, model=model, pixel_x=x, variance_cal=v,
                            n_runs=TEST_RUNS, seed=_seed(s, 4, vi)).save(_width_path(spec, v))
            print(f"{spec}: width cells v = {v:g} {time.time() - t0:.0f} s", flush=True)


def _pooled_cells(spec: str) -> CalibrationCells | None:
    """The sigma_0 cells of every cached draw, pooled (equal runs per draw)."""
    draws = [CalibrationCells.load(_cells_path(spec, k)) for k in range(N_FISHER_DRAWS)
             if _cells_path(spec, k).exists()]
    if not draws:
        return None
    first = draws[0]
    return CalibrationCells(
        cnr=first.cnr,
        mean_bias=np.mean([c.mean_bias for c in draws], axis=0),
        bias_sd=np.sqrt(np.mean([c.bias_sd ** 2 for c in draws], axis=0)),
        n_finite=np.sum([c.n_finite for c in draws], axis=0),
        n_runs=np.sum([c.n_runs for c in draws], axis=0),
        variance_cal=first.variance_cal, pixel_x=first.pixel_x,
        model_spec=first.model_spec)


def _fisher_vs_monte_carlo(lines: list[str]) -> None:
    x = Experiment().x
    lines.append("Part 1: Fisher bias scale A = CNR^2 Var(sigma_hat) at sigma_0 vs Monte Carlo spread")
    lines.append(f"  closed form, 1-D Gaussian with peak CNR: (2/sqrt(pi)) dx = "
                 f"{2.0 / np.sqrt(np.pi) * (x[1] - x[0]):.5f} sigma_0^2")
    lines.append(f"  {'shape':<26}{'A Fisher':>10}{'A MC':>10}{'ratio':>8}   (MC: CNR >= 8, all draws)")
    for spec in SPECS:
        cells = _pooled_cells(spec)
        if cells is None:
            continue
        a_fisher = leading_order_terms(model_from_spec(spec), x, 1.0).bias_coef
        high = cells.cnr >= 8.0
        a_mc = np.median(cells.cnr[high] ** 2 * cells.bias_sd[high] ** 2 / 4.0)
        lines.append(f"  {spec:<26}{a_fisher:10.4f}{a_mc:10.4f}{a_fisher / a_mc:8.3f}")
    lines.append("")


def _width_scaling(lines: list[str]) -> None:
    x = Experiment().x
    lines.append("Part 2: bias at true variance v against three predictions from the sigma_0 calibration")
    for spec in WIDTH_SPECS:
        cells = _pooled_cells(spec)
        if cells is None or not _width_path(spec, TEST_VARIANCES[0]).exists():
            continue
        model = model_from_spec(spec)
        terms = leading_order_terms(model, x, 1.0, WIDTH_GRID)
        lookup = BiasLookup.from_cells(cells, terms)
        lines.append(f"=== {spec}")
        lines.append(f"  {'v/sigma0^2':>10}{'r(v)':>8}   chi2/dof (CNR >= 1):  none    lead   table"
                     f"   | all CNR:  none    lead   table")
        collapse = []
        for v in TEST_VARIANCES:
            test = CalibrationCells.load(_width_path(spec, v))
            c = test.cnr
            measured = test.mean_bias
            measured_se = test.bias_sd / np.sqrt(test.n_finite)
            ratio_v = float(np.interp(v, terms.variance_grid, terms.bias_scale))
            base = lookup.var_bias(c)
            base_se = lookup.var_bias_se(c)
            scale, c_eq = lookup._width_lookup_point(c, np.full(c.size, v))
            table = scale * np.interp(c_eq, lookup.cnr_grid, lookup.varbias_grid)
            table_se = scale * np.linalg.norm(
                _interp_matrix(c_eq, lookup.cnr_grid) @ lookup.varbias_cov_factor, axis=1)
            predictions = {"none": (base, base_se), "lead": (ratio_v * base, ratio_v * base_se),
                           "table": (table, table_se)}
            chi2 = {}
            for name, (pred, pred_se) in predictions.items():
                z2 = (measured - pred) ** 2 / (measured_se ** 2 + pred_se ** 2)
                chi2[name] = (z2[c >= 1.0].mean(), z2.mean())
            lines.append(f"  {v:>10g}{ratio_v:8.3f}   {'':>18}"
                         + "".join(f"{chi2[n][0]:8.2f}" for n in predictions)
                         + "   | " + " " * 7
                         + "".join(f"{chi2[n][1]:8.2f}" for n in predictions))
            leading = terms.bias_coef * ratio_v / c ** 2
            precision = c * np.sqrt(v / (terms.bias_coef * ratio_v))
            collapse.append((v, precision, measured / leading, measured_se / leading))
        lines.append("  measured bias / leading term against S = CNR sqrt(v / A(v)) (the curves should coincide):")
        for v, precision, rho, rho_se in collapse:
            pairs = "  ".join(f"S {p:5.1f}: {r:5.2f}+/-{e:4.2f}"
                              for p, r, e in zip(precision[::3], rho[::3], rho_se[::3]))
            lines.append(f"    v {v:>4g}: {pairs}")
        lines.append("")


def analyse() -> None:
    lines: list[str] = []
    _fisher_vs_monte_carlo(lines)
    _width_scaling(lines)
    report = "\n".join(lines)
    print(report)
    (OUT / "fisher_check.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"generate": generate, "analyse": analyse}[sys.argv[1] if len(sys.argv) > 1 else "analyse"]()
