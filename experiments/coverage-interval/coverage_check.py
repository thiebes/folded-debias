"""Does the reported confidence interval on D actually cover the truth?

The validation so far checks the *mean* <D_hat>/D_true, not whether a per-
measurement confidence interval on D_hat covers D_true at its nominal rate. This
script does the interval check, on the de-biased estimator (coverage is only
meaningful once the estimator is unbiased -- a confidence interval around a
biased D can never cover the truth, which is why ``inverse_variance`` is shown
here only as the failing contrast).

The standard covariance interval ``D_hat +/- 1.96 * D_se`` (``D_se`` from the
weighted-least-squares slope covariance, ``fit.fit_D``) assumes the MSD points
are independent. They are not: every ``MSD_i = sigma_hat^2_i - sigma_hat^2_0``
shares the fitted t=0 width, so the points are positively correlated, and a
diagonal-weight covariance ignores that cross-covariance and runs too narrow.
The interval therefore under-covers, the more so at low CNR where Var(sigma_hat^2_0)
is large relative to the spread. (Because the weights are calibrated and
deterministic in CNR, this interval has the *same* width for every measurement
at a given CNR_0 -- it does not adapt to the data.)

For each CNR_0 the script reports, over many realisations:

* empirical coverage of the 95% covariance interval (fraction containing
  D_true), for ``folded_normal`` (de-biased) and ``inverse_variance`` (biased);
* the narrowness ratio ``mean(D_se) / SD(D_hat)`` -- below 1 means too narrow;
* the skew of D_hat (its sampling distribution is right-skewed at low CNR, so a
  symmetric interval is doubly wrong there);
* the coverage of a parametric bootstrap interval that resamples each frame's
  ``sigma_hat^2`` (including the shared t=0 frame), which restores the
  correlation the covariance interval drops and yields a percentile interval
  that also follows the skew.

The per-frame variances used to resample are measured from the ensemble here, to
isolate the methodological question (given correct per-frame variances, does the
covariance interval under-cover, and does the bootstrap fix it); in practice they
come from each frame's own fit covariance or the calibration.

Run from the repository root::

    .venv/Scripts/python.exe scripts/coverage_check.py

Set ``FOLDED_DEBIAS_WORKERS=0`` to use all cores. Reuses the standard-geometry
D-free Gaussian calibration cached by ``subdiffusive_alpha.py``; builds it if
absent. Outputs ``output/coverage_check.{png,txt}``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from folded_debias import Experiment, GaussianModel, calibrate_d_free  # noqa: E402
from folded_debias.montecarlo import run_ensemble  # noqa: E402
from folded_debias.truncation import TruncationModel  # noqa: E402
from folded_debias.weights import msd_and_weights  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_subdiffusive.npz"   # shared standard-geometry cache

# Spain CVD (Okabe-Ito); see plot_style.md.
COV_RED = "#D55E00"        # covariance interval (under-covers)
BOOT_BLUE = "#0072B2"      # bootstrap interval (the fix)
IV_PURPLE = "#CC79A7"      # inverse-variance (biased -> no coverage)
NOMINAL_GREY = "#999999"
INCH = 1 / 2.54
FONT = 9

D_TRUE = 0.02
GEOMETRY = dict(sigma0=1.0, tau=1.0, half_window=6.0, n_pixels=241,
                n_frames=12, t_max=3.0)
CNR0_SWEEP = (5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0)
N_RUNS = 2000
N_BOOT = 200
Z = 1.959964               # 95% two-sided normal quantile
CAL_CNR0 = np.geomspace(2.0, 100.0, 60)
N_RUNS_CAL = 1500
SEED = 90000


def load_calibration() -> TruncationModel:
    """Load the shared D-free Gaussian calibration; build and cache if absent."""
    if CAL_PATH.exists() and "framevar_grid" in np.load(CAL_PATH).files:
        d = np.load(CAL_PATH)
        return TruncationModel(cnr_grid=d["cnr_grid"], p_grid=d["p_grid"],
                               relvar_grid=d["relvar_grid"],
                               varbias_grid=d["varbias_grid"],
                               framevar_grid=d["framevar_grid"])
    trunc = calibrate_d_free(CAL_CNR0, n_runs=N_RUNS_CAL, model=GaussianModel(),
                             **GEOMETRY)
    np.savez(CAL_PATH, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid)
    return trunc


def wls_slope(msd, t, w) -> tuple[np.ndarray, float]:
    """Vectorised weighted-LS slope over the last axis; returns (D, D_se).

    ``msd`` has frames on its last axis (any leading shape). ``D_se`` is the
    covariance SE of D and is a scalar -- it depends only on the (deterministic)
    weights and times, not on the data, so every measurement at one CNR_0 shares
    it. Mirrors :func:`folded_debias.fit.fit_D`.
    """
    g = w > 0
    tg, wg = t[g], w[g]
    sw = wg.sum(); swt = (wg * tg).sum(); swtt = (wg * tg * tg).sum()
    denom = sw * swtt - swt ** 2
    swm = msd[..., g] @ wg
    swtm = msd[..., g] @ (wg * tg)
    slope = (sw * swtm - swt * swm) / denom
    return slope / 2.0, float(np.sqrt(sw / denom) / 2.0)


def _skew(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    d = a - a.mean(); s2 = (d ** 2).mean()
    return float((d ** 3).mean() / s2 ** 1.5) if s2 > 0 else 0.0


def coverage_at(cnr0: float, trunc: TruncationModel, seed: int) -> dict:
    """Coverage of the covariance, bootstrap, and inverse-variance intervals."""
    exp = Experiment(D=D_TRUE, **GEOMETRY)
    ens = run_ensemble(exp, cnr0, n_runs=N_RUNS, seed=seed, model=GaussianModel())
    t = exp.times
    cnr = ens.cnr_frame
    var_bias = trunc.var_bias(cnr)                       # (n_frames,), deterministic
    _, w = msd_and_weights("folded_normal", ens.variance[0], cnr, trunc)

    # Point estimates per realisation (de-biased and biased).
    corrected = ens.variance - var_bias
    msd_fn = corrected - corrected[:, [0]]
    D_fn, D_se = wls_slope(msd_fn, t, w)                 # D_se scalar (shared)
    msd_iv = ens.variance - ens.variance[:, [0]]
    D_iv, _ = wls_slope(msd_iv, t, w)

    half = Z * D_se
    cover_cov = float(np.mean(np.abs(D_fn - D_TRUE) <= half))
    cover_iv = float(np.mean(np.abs(D_iv - D_TRUE) <= half))
    sd = float(np.std(D_fn[np.isfinite(D_fn)]))

    # Ensemble-measured per-frame variance Var(sigma_hat^2_i) (the bootstrap's
    # resampling scale, and the attribution input).
    vfr = np.nanvar(ens.variance, axis=0, ddof=1)

    # Parametric bootstrap: resample each frame's sigma_hat^2 (incl. the shared
    # t=0 frame), re-correct, refit; percentile interval over the resamples.
    rng = np.random.default_rng(seed + 7)
    samp = (ens.variance[:, None, :]
            + rng.standard_normal((N_RUNS, N_BOOT, exp.n_frames)) * np.sqrt(vfr))
    cb = samp - var_bias
    msd_b = cb - cb[:, :, [0]]
    D_b, _ = wls_slope(msd_b, t, w)                      # (N_RUNS, N_BOOT)
    lo, hi = np.percentile(D_b, [2.5, 97.5], axis=1)
    cover_boot = float(np.mean((lo <= D_TRUE) & (D_TRUE <= hi)))

    # Attribution: how much the shared sigma_hat^2_0 SHOULD widen the slope CI,
    # from the measured per-frame variances. Diagonal model treats the MSD points
    # as independent (Var(MSD_i) = vfr_i + v0); the full model adds the v0
    # cross-covariance shared by every point. inflation = ratio of slope SEs.
    g = w > 0
    v0 = float(vfr[0]); ti = t[g]; n = int(g.sum())
    diag = vfr[g] + v0
    wi = 1.0 / diag
    sw_a = wi.sum(); swt_a = (wi * ti).sum(); swtt_a = (wi * ti * ti).sum()
    var_diag = sw_a / (sw_a * swtt_a - swt_a ** 2)
    cmat = np.diag(vfr[g]) + v0 * np.ones((n, n))        # C_ii=vfr_i+v0, C_ij=v0
    xmat = np.column_stack([np.ones(n), ti])
    var_full = float(np.linalg.inv(xmat.T @ np.linalg.inv(cmat) @ xmat)[1, 1])
    inflation = float(np.sqrt(var_full / var_diag)) if var_diag > 0 else np.nan

    return dict(cnr0=cnr0, min_cnr=float(cnr.min()), D_se=D_se, sd=sd,
                narrowness=D_se / sd if sd > 0 else np.nan,
                corr_inflation=inflation,
                skew=_skew(D_fn), bias_fn=float(np.median(D_fn)) / D_TRUE,
                bias_iv=float(np.median(D_iv)) / D_TRUE,
                cover_cov=cover_cov, cover_iv=cover_iv, cover_boot=cover_boot)


def write_report(rows: list[dict], path: Path) -> None:
    lines = [
        "=" * 78,
        "Coverage / interval check -- does the reported CI on D cover the truth?",
        "=" * 78,
        f"D_true = {D_TRUE} sigma0^2/tau, Gaussian, standard geometry; "
        f"{N_RUNS} runs/cell, {N_BOOT} bootstrap.",
        "95% intervals. Coverage = fraction containing D_true.",
        "folded_normal is de-biased (coverage meaningful); inverse_variance is "
        "biased (contrast).",
        "narrowness = mean(D_se)/SD(D_hat) (<1 = covariance CI too narrow).",
        "infl = sqrt(Var_full/Var_diag): how much the shared sigma_hat^2_0",
        "       correlation alone should widen the slope CI (>1 = correlation matters).",
        "",
        "  CNR0 | minCNR | bias fn/iv | SD/D | narrow | infl | skew | "
        "cover cov/boot/iv",
        "  (cover nominal = 0.95)",
    ]
    for r in rows:
        lines.append(
            f"  {r['cnr0']:5.0f} | {r['min_cnr']:6.2f} | "
            f"{r['bias_fn']:.2f}/{r['bias_iv']:.2f} | "
            f"{r['sd'] / D_TRUE:5.2f} | {r['narrowness']:6.2f} | "
            f"{r['corr_inflation']:4.2f} | {r['skew']:5.2f} | "
            f"{r['cover_cov']:.2f}/{r['cover_boot']:.2f}/{r['cover_iv']:.2f}")
    report = "\n".join(lines)
    print(report)
    path.write_text(report)


def make_figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    cnr0 = np.array([r["cnr0"] for r in rows])
    fig, (axA, axB) = plt.subplots(1, 2, layout="constrained",
                                   figsize=(18.0 * INCH, 8.0 * INCH))
    axA.axhline(0.95, color=NOMINAL_GREY, lw=1.0, ls=":", label="nominal 0.95")
    axA.plot(cnr0, [r["cover_cov"] for r in rows], "o-", color=COV_RED, lw=2,
             ms=4, label="covariance CI (folded)")
    axA.plot(cnr0, [r["cover_boot"] for r in rows], "s-", color=BOOT_BLUE, lw=2,
             ms=4, label="bootstrap CI (folded)")
    axA.plot(cnr0, [r["cover_iv"] for r in rows], "^-", color=IV_PURPLE, lw=2,
             ms=4, label="covariance CI (inverse-variance, biased)")
    axA.set_ylim(0.0, 1.02)
    axA.set_ylabel("empirical coverage of the 95% interval", fontsize=FONT)

    axB.axhline(1.0, color=NOMINAL_GREY, lw=1.0, ls=":", label="SE = SD")
    axB.plot(cnr0, [r["narrowness"] for r in rows], "o-", color=COV_RED, lw=2,
             ms=4, label=r"covariance SE / SD(D) (deployed)")
    axB.plot(cnr0, [r["corr_inflation"] for r in rows], "s-", color=BOOT_BLUE,
             lw=2, ms=4, label=r"correlation inflation $\sqrt{V_{\rm full}/V_{\rm diag}}$")
    axB.set_ylabel("SE diagnostics (ratio)", fontsize=FONT)

    for ax in (axA, axB):
        ax.set_xscale("log")
        ax.set_xticks(cnr0)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$", fontsize=FONT)
        ax.tick_params(axis="both", which="both", direction="in", top=True,
                       right=True, labelsize=FONT)
        ax.legend(frameon=False, fontsize=FONT, loc="lower right")
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


def main() -> None:
    print(f"Loading/building D-free calibration ...")
    trunc = load_calibration()
    print(f"Coverage sweep: {len(CNR0_SWEEP)} CNR0 x {N_RUNS} runs "
          f"x {N_BOOT} bootstrap ...")
    rows = [coverage_at(float(c), trunc, SEED + 100 * i)
            for i, c in enumerate(CNR0_SWEEP)]
    write_report(rows, OUTPUT / "coverage_check.txt")
    make_figure(rows, OUTPUT / "coverage_check.png")


if __name__ == "__main__":
    main()
