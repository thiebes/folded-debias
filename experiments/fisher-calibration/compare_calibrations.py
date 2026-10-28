"""Compare the Fisher-anchored calibration with the raw D-free lookup.

The switch test for ``calibrate_fisher``: at an equal Monte Carlo budget, is it
at least as accurate as ``calibrate_d_free`` and does it carry a smaller
calibration error? The criterion is fixed in ``README.md`` before the run.

``generate`` builds and caches, per profile shape:

* one evaluation set on validate_d_zero's grid (D 0.005 to 0.1, CNR_0 5 to 90,
  ``EVAL_RUNS`` realizations per cell);
* ``N_DRAWS`` independent raw calibrations (``calibrate_d_free`` on CNR_0 5 to
  90, 300 runs: 7 x 300 x 12 = 25,200 fits) and ``N_FISHER_DRAWS`` independent
  sets of Fisher cells (84 CNR from 0.24 to 90, 300 runs: 25,200 fits). The
  budget per draw is equal; the Fisher method gets more draws because its
  predicted calibration SE is under test, and a variance ratio from 8 draws has
  7 degrees of freedom (95% range 0.24 to 2.29 for an honest SE).

Seeds have separate roots per shape, method, and draw, so no two ensembles
share noise (validate_d_zero used the same seeds for every shape).

``analyse`` applies every calibration to the same evaluation realizations, so
the draw-to-draw scatter of a cell mean is the calibration error alone. Arms:
``raw``, ``fisher`` (no width scaling), and ``fisher_w`` (with the width table;
reported, not part of the switch criterion). Each is scored on all frames (the
library default) and on frames with per-frame CNR >= ``FLOOR`` (the regime the
method claims), which the criterion uses.

    $env:FOLDED_DEBIAS_WORKERS = 0
    .venv/Scripts/python.exe experiments/fisher-calibration/compare_calibrations.py generate
    .venv/Scripts/python.exe experiments/fisher-calibration/compare_calibrations.py analyse
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import (CalibrationCells, Experiment, TruncationModel,  # noqa: E402
                           calibrate_cells, calibrate_d_free, leading_order_terms,
                           model_from_spec, spec_tag)
from folded_debias.montecarlo import _resolve_workers, run_ensemble  # noqa: E402
from folded_debias.weights import calibration_D_se, msd_and_weights  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "output" / "fisher-calibration"

SPECS = ("gaussian", "voigt", "generalized_gaussian:1.5", "generalized_gaussian:4")
D_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1)
CNR0_VALUES = (5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0)
RAW_GRID = np.array(CNR0_VALUES)
FISHER_GRID = np.geomspace(0.24, 90.0, 84)
CAL_RUNS = 300
N_DRAWS = int(os.environ.get("N_DRAWS", 8))
N_FISHER_DRAWS = int(os.environ.get("N_FISHER_DRAWS", 64))
EVAL_RUNS = 1000
FLOOR = 1.0
WIDTH_GRID = np.geomspace(0.25, 16.0, 41)
ARMS = ("raw", "fisher", "fisher_w")
SEED_ROOT = 1


def _seed(shape_index: int, role: int, index: int) -> int:
    """Separate seed roots per role (1 raw, 2 Fisher cells, 3 evaluation, 4 width
    cells) and shape.

    ``index`` stays below 10^6, so no two ensembles share a seed; a raw draw's
    sweep adds 0..6 to its root, hence the 1000 spacing of raw draws. Root 0 was
    spent on a Gaussian-only mid-size check of the predicted calibration SE before
    the full run, so the full run starts at root 1 and none of its noise had been
    examined when the criterion was fixed.
    """
    return 100_000_000 * SEED_ROOT + 10_000_000 * role + 1_000_000 * shape_index + index


def _eval_path(spec: str) -> Path:
    return OUT / f"eval_{spec_tag(spec)}_runs{EVAL_RUNS}.npz"


def _raw_path(spec: str, draw: int) -> Path:
    return OUT / f"raw_{spec_tag(spec)}_draw{draw}.npz"


def _cells_path(spec: str, draw: int) -> Path:
    return OUT / f"cells_{spec_tag(spec)}_draw{draw}.npz"


def generate() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for s, spec in enumerate(SPECS):
        model = model_from_spec(spec)
        if not _eval_path(spec).exists():
            t0 = time.time()
            var = np.full((len(D_VALUES), len(CNR0_VALUES), EVAL_RUNS, 12), np.nan)
            cnr = np.full((len(D_VALUES), len(CNR0_VALUES), 12), np.nan)
            for di, d in enumerate(D_VALUES):
                for ci, c0 in enumerate(CNR0_VALUES):
                    ens = run_ensemble(Experiment(D=d), c0, n_runs=EVAL_RUNS,
                                       seed=_seed(s, 3, 100 * di + ci), model=model)
                    var[di, ci], cnr[di, ci] = ens.variance, ens.cnr_frame
            np.savez(_eval_path(spec), variance=var, cnr=cnr)
            print(f"{spec}: evaluation set {time.time() - t0:.0f} s", flush=True)
        for k in range(max(N_DRAWS, N_FISHER_DRAWS)):
            if k < N_DRAWS and not _raw_path(spec, k).exists():
                t0 = time.time()
                calibrate_d_free(RAW_GRID, n_runs=CAL_RUNS, seed=_seed(s, 1, 1000 * k),
                                 model=model).save(_raw_path(spec, k))
                print(f"{spec}: raw draw {k} {time.time() - t0:.0f} s", flush=True)
            if k < N_FISHER_DRAWS and not _cells_path(spec, k).exists():
                t0 = time.time()
                calibrate_cells(FISHER_GRID, model=model, pixel_x=Experiment().x,
                                n_runs=CAL_RUNS, seed=_seed(s, 2, k)).save(
                                    _cells_path(spec, k))
                print(f"{spec}: Fisher cells draw {k} {time.time() - t0:.0f} s", flush=True)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _models(spec: str, draw: int) -> dict[str, TruncationModel | None]:
    """The three arms for one draw; ``raw`` is None beyond the raw draws."""
    model = model_from_spec(spec)
    x = Experiment().x
    cells = CalibrationCells.load(_cells_path(spec, draw))
    raw = _raw_path(spec, draw)
    return {
        "raw": TruncationModel.load(raw) if raw.exists() else None,
        "fisher": TruncationModel.from_cells(cells, leading_order_terms(model, x, 1.0)),
        "fisher_w": TruncationModel.from_cells(
            cells, leading_order_terms(model, x, 1.0, WIDTH_GRID)),
    }


def _cell_scores(model: TruncationModel, var: np.ndarray, cnr: np.ndarray,
                 d_true: float, floor: float | None) -> np.ndarray:
    """(mean D/D_ref, SD of D/D_ref, mean calibration SE / D_ref, coverage) for one cell.

    Vectorised over realizations: with no width key the weights and both SEs
    are the same for every realization of a cell (they depend only on the
    per-frame CNR), and var_bias acts elementwise, so this equals calling
    msd_and_weights, fit_D, and calibration_D_se per realization.
    """
    times = Experiment(D=d_true).times
    _, w = msd_and_weights("folded_normal", var[0], cnr, model)
    if floor is not None:
        w = np.where(cnr >= floor, w, 0.0)
    corrected = var - model.var_bias(np.broadcast_to(cnr, var.shape).ravel(),
                                     var.ravel()).reshape(var.shape)
    msd = corrected - corrected[:, :1]
    good = (w > 0) & np.all(np.isfinite(msd), axis=0)
    tg, wg = times[good], w[good]
    sw, swt, swtt = wg.sum(), (wg * tg).sum(), (wg * tg * tg).sum()
    denom = sw * swtt - swt ** 2
    c = wg * (sw * tg - swt) / denom
    ratio = (msd[:, good] @ c) / 2.0 / d_true
    d_se = np.sqrt(sw / denom) / 2.0 / d_true
    if model.is_width_scaled:
        cal_se = np.array([calibration_D_se(times, m, w, cnr, model, v)
                           for m, v in zip(msd, var)]) / d_true
    else:
        cal_se = np.full(var.shape[0], calibration_D_se(times, msd[0], w, cnr, model)
                         / d_true)
    covered = np.abs(ratio - 1.0) <= 1.96 * np.sqrt(d_se ** 2 + cal_se ** 2)
    return np.array([ratio.mean(), ratio.std(ddof=1), cal_se.mean(), covered.mean()])


def _score_draw(task):
    """All arms, cells, and floor settings for one (shape, draw); picklable."""
    spec, draw = task
    models = _models(spec, draw)
    with np.load(_eval_path(spec)) as z:
        var, cnr = z["variance"], z["cnr"]
    scores = np.full((len(ARMS), 2, len(D_VALUES), len(CNR0_VALUES), 4), np.nan)
    for a, arm in enumerate(ARMS):
        if models[arm] is None:
            continue
        for f, floor in enumerate((None, FLOOR)):
            for di, d in enumerate(D_VALUES):
                for ci in range(len(CNR0_VALUES)):
                    scores[a, f, di, ci] = _cell_scores(models[arm], var[di, ci],
                                                        cnr[di, ci], d, floor)
    return spec, draw, scores


def _score_all(specs) -> dict[str, np.ndarray]:
    """scores[spec] has shape (draw, arm, floor, D, CNR_0, statistic).

    Draws run over the Fisher cells; the raw arm is NaN where a draw has no raw
    calibration.
    """
    tasks = [(spec, k) for spec in specs for k in range(N_FISHER_DRAWS)
             if _cells_path(spec, k).exists()]
    n_workers = _resolve_workers(None, len(tasks))
    if n_workers == 1:
        results = [_score_draw(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(_score_draw, tasks))
    out = {}
    for spec in specs:
        draws = sorted((k, s) for sp, k, s in results if sp == spec)
        if draws:
            out[spec] = np.stack([s for _, s in draws])
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def _table(lines: list[str], title: str, values: np.ndarray, fmt: str) -> None:
    lines.append(f"  {title}")
    lines.append("     D \\ CNR0 " + "".join(f"{c:>8g}" for c in CNR0_VALUES))
    for di, d in enumerate(D_VALUES):
        lines.append(f"    {d:>8g}  " + "".join(format(v, fmt) for v in values[di]))


def _arm_summary(scores: np.ndarray, arm: int, floor: int):
    """Cell bias, its SE, calibration SE (observed, predicted), coverage.

    Uses the draws in which ``arm`` was scored: all Fisher draws, or the first
    ``N_DRAWS`` for the raw arm.
    """
    scored = np.isfinite(scores[:, arm]).any(axis=(1, 2, 3, 4))
    s = scores[scored, arm, floor]               # (draw, D, CNR_0, stat)
    n_draws = s.shape[0]
    mean = s[..., 0]                             # (draw, D, CNR_0)
    eval_se = s[0, ..., 1] / np.sqrt(EVAL_RUNS)  # same evaluation set every draw
    cal_obs = mean.std(axis=0, ddof=1)
    cal_pred = s[..., 2].mean(axis=0)
    bias = mean.mean(axis=0) - 1.0
    bias_se = np.sqrt(eval_se ** 2 + cal_obs ** 2 / n_draws)
    # Pooled over cells with inverse-variance weights; the calibration part of
    # its error is taken from the draw-to-draw scatter of the pooled value, so
    # the correlation between cells sharing one calibration is included.
    weight = 1.0 / bias_se ** 2
    weight /= weight.sum()
    pooled_per_draw = np.tensordot(mean - 1.0, weight, axes=([1, 2], [0, 1]))
    pooled = pooled_per_draw.mean()
    pooled_se = np.sqrt(np.sum(weight ** 2 * eval_se ** 2)
                        + pooled_per_draw.var(ddof=1) / n_draws)
    coverage = s[..., 3].mean(axis=0)
    return dict(bias=bias, bias_se=bias_se, cal_obs=cal_obs, cal_pred=cal_pred,
                pooled=pooled, pooled_se=pooled_se, coverage=coverage)


def analyse() -> None:
    specs = [spec for spec in SPECS if _eval_path(spec).exists()]
    scores = _score_all(specs)
    lines = [
        "Fisher-anchored calibration vs raw D-free lookup (equal budget: 25,200 fits per draw)",
        f"evaluation: {EVAL_RUNS} realizations per cell; independent draws: {N_DRAWS} raw, "
        f"{N_FISHER_DRAWS} Fisher; criterion arm: frames with per-frame CNR >= {FLOOR}; "
        f"seed root {SEED_ROOT}",
        "",
    ]
    d = np.array(D_VALUES)[:, None] * np.ones(len(CNR0_VALUES))
    verdicts = {}
    for spec, sc in scores.items():
        lines.append("=" * 78)
        n_raw = int(np.isfinite(sc[:, 0]).any(axis=(1, 2, 3, 4)).sum())
        lines.append(f"{spec}  ({n_raw} raw draws, {sc.shape[0]} Fisher draws)")
        summary = {(arm, f): _arm_summary(sc, a, f)
                   for a, arm in enumerate(ARMS) for f in (0, 1)}
        for f, label in ((1, f"frames with CNR >= {FLOOR}"), (0, "all frames")):
            lines.append(f"--- {label}")
            for arm in ARMS:
                r = summary[(arm, f)]
                n_out = int(np.sum(np.abs(r["bias"]) > 2 * r["bias_se"]))
                lines.append(f"  [{arm}] pooled bias {r['pooled']:+.4f} +/- {r['pooled_se']:.4f}; "
                             f"cells beyond 2 SE: {n_out} of {r['bias'].size}")
                _table(lines, f"[{arm}] mean D/D_ref - 1 over draws", r["bias"], "+8.3f")
                _table(lines, f"[{arm}] |bias| / SE", np.abs(r["bias"]) / r["bias_se"], "8.1f")
                _table(lines, f"[{arm}] calibration SE (predicted, D/D_ref)", r["cal_pred"], "8.3f")
                _table(lines, f"[{arm}] coverage of the 95% interval", r["coverage"], "8.2f")
            fisher, raw = summary[("fisher", f)], summary[("raw", f)]
            ratio = fisher["cal_pred"] / raw["cal_pred"]
            _table(lines, "calibration SE ratio fisher/raw (predicted)", ratio, "8.2f")
            for arm in ("raw", "fisher"):
                r = summary[(arm, f)]
                obs_pred = np.mean(r["cal_obs"] ** 2 / r["cal_pred"] ** 2)
                lines.append(f"  [{arm}] observed / predicted calibration variance "
                             f"(mean over cells): {obs_pred:.2f}")
            lines.append("")

        f = 1
        fisher, raw = summary[("fisher", f)], summary[("raw", f)]
        n_f = int(np.sum(np.abs(fisher["bias"]) > 2 * fisher["bias_se"]))
        n_r = int(np.sum(np.abs(raw["bias"]) > 2 * raw["bias_se"]))
        ratio = fisher["cal_pred"] / raw["cal_pred"]
        obs_pred = np.mean(fisher["cal_obs"] ** 2 / fisher["cal_pred"] ** 2)
        checks = {
            "1a pooled bias": (abs(fisher["pooled"]) <= 2 * fisher["pooled_se"]
                               or abs(fisher["pooled"]) <= abs(raw["pooled"])),
            "1b cells beyond 2 SE": n_f <= max(4, n_r),
            "2a median cal-SE ratio <= 0.7": np.median(ratio) <= 0.7,
            "2b no D>=0.02 cell ratio > 1.1": bool(np.all(ratio[d >= 0.02] <= 1.1)),
            "3 observed/predicted in [0.7, 1.4]": 0.7 <= obs_pred <= 1.4,
            "4a coverage >= raw - 0.02": bool(np.all(fisher["coverage"] >= raw["coverage"] - 0.02)),
            "4b median coverage (D<=0.02) >= 0.93": np.median(fisher["coverage"][d <= 0.02]) >= 0.93,
        }
        verdicts[spec] = all(checks.values())
        lines.append("  switch criterion (frames with CNR >= 1):")
        lines += [f"    {'pass' if ok else 'fail'}  {name}" for name, ok in checks.items()]
        lines.append("")
    lines.append("=" * 78)
    lines.append("verdict: " + ("calibrate_fisher passes on every shape"
                                if verdicts and all(verdicts.values()) and len(verdicts) == len(SPECS)
                                else "calibrate_fisher does not pass on every shape: "
                                + ", ".join(f"{s} {'pass' if v else 'fail'}"
                                            for s, v in verdicts.items())))
    report = "\n".join(lines)
    print(report)
    (OUT / "compare_calibrations.txt").write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    {"generate": generate, "analyse": analyse}[sys.argv[1] if len(sys.argv) > 1 else "analyse"]()
