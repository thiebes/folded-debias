"""Per-frame residual + short-window D on the high-N Br-I bed.

The full-3-ns D collapse is brittle here: the peak drops ~147x over the window,
so most late frames fall below CNR ~1 and the slope fit has a thin lever arm.
The honest test is the project's actual metric -- the per-frame mean residual
``E[corrected] - var_true`` over many seeds -- restricted to the *usable window*
(frames whose per-frame CNR stays above the floor). Over that band we ask: does
the correction reduce the upward bias the naive fit carries, and does the
short-window D move toward the truth?

The correction is keyed on (CNR, sigma^2) through a joint D-free calibration
on the data's pixel grid (``cnr_sigma2``, the default). An earlier version used
a 1-D D-aware calibration keyed on CNR alone (``daware``, kept for
reproducing that table); because the spot's variance grows ~64x over the
usable band, that lookup mis-subtracted in the late frames above the floor and
raised the per-frame mean |residual| after correction (see
``cnr_sigma2_calibration``).

Run from the repository root (needs the pshare)::

    .venv/Scripts/python.exe experiments/halide-mixing-highn/br_halide_highn_perframe.py [cnr_sigma2|daware] [CNR0]

Outputs ``output/br_halide_highn_perframe[_daware][_cnr0<N>].{png,txt}`` and
caches the calibrations under ``output/``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "halide-mixing-brI-legacy"))
sys.path.insert(0, os.path.dirname(__file__))

from folded_debias import (CalibrationGrid, GaussianModel,  # noqa: E402
                           calibrate_cnr_sigma2)
from folded_debias.simulate import Experiment  # noqa: E402
from folded_debias.truncation import TruncationModel  # noqa: E402
from folded_debias.validate import calibrate  # noqa: E402
from folded_debias.weights import corrected_variance  # noqa: E402

from folded_debias.fit import fit_variance as _fit_gaussian  # noqa: E402

from br_halide_validation import fit_msd_powerlaw  # noqa: E402
from br_halide_highn_validation import STRIDE, subsample  # noqa: E402
from br_highn_data import SIGMA0_HINT_NM, load_all  # noqa: E402

# Tighten the width fit's upper bound from the window span^2 to a physical cap
# (G * sigma0)^2: the spot grows ~8x in sigma over the window, so G = 12 clears
# every real frame (sigma capped at 300 nm) while bounding the low-CNR fit's
# noise-wandering, which the wide 3000 nm window otherwise leaves unbounded.
# Applied identically to the data fit and the calibration (the var_bias must
# come from the same bound).
G_MAX = 12.0
MAX_VAR = (G_MAX * SIGMA0_HINT_NM) ** 2


def fit_each_row(x, profiles):
    """Bounded Gaussian width fit per frame, with the tightened max_var bound."""
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = _fit_gaussian(x, profiles[i], max_var=MAX_VAR)
        if r.ok and np.isfinite(r.variance):
            out[i] = r.variance
    return out

OUTPUT = Path(__file__).resolve().parents[2] / "output"  # repo-root output/ (cached calibrations)
OUTPUT.mkdir(exist_ok=True)

CORR_BLUE = "#0072B2"
NAIVE_RED = "#D55E00"
TRUTH_BLACK = "#000000"
FLOOR_GREY = "#999999"
INCH = 1 / 2.54
FONT = 9

CNR0 = 10.0          # initial CNR for the per-frame panel
CNR0_TABLE = (8.0, 15.0, 30.0)   # short-window D reported at these CNR_0
N_SEEDS = 800
# Joint-lookup key: "perseed" (each frame's own sigma_hat^2, library default)
# or "smoothed" (a robust line through the measurement; see smoothed_key).
KEY = os.environ.get("HIGHN_KEY", "perseed")
FLOOR_CNR = 1.3      # project floor: folded-normal over-subtracts below ~1.3
CAL_CNR0 = np.geomspace(1.2, 60.0, 30)
N_RUNS_CAL = 600
# Joint (CNR, sigma^2) calibration: sigma spans the true widths of the usable
# band (25 nm at t = 0 to ~125 nm where the CNR reaches the floor) with margin.
SIGMA_GRID_NM = np.linspace(20.0, 220.0, 26)
# Dense below CNR 3, where the bias and its Monte Carlo noise are largest: a
# 600-run, 30-point grid left a few-hundred-nm^2 gap between the floor (1.3) and
# CNR 2 even with an exact sigma^2 key.
CAL2_CNR = np.concatenate([np.geomspace(1.0, 3.0, 20, endpoint=False),
                           np.geomspace(3.0, 60.0, 20)])
N_RUNS_CAL2 = int(os.environ.get("HIGHN_CAL2_RUNS", 4000))


def daware_calibration(d_true: float, frac: str, n_frames: int) -> TruncationModel:
    """D-aware calibration at this fraction's D and geometry; cached per fraction."""
    path = OUTPUT / f"cal_highn_daware_g{int(G_MAX)}_x{frac}.npz"
    if path.exists() and "framevar_grid" in np.load(path).files:
        d = np.load(path)
        return TruncationModel(cnr_grid=d["cnr_grid"], p_grid=d["p_grid"],
                               relvar_grid=d["relvar_grid"],
                               varbias_grid=d["varbias_grid"],
                               framevar_grid=d["framevar_grid"])
    exp = Experiment(sigma0=25.0, D=float(d_true), tau=1.0, half_window=60.0,
                     n_pixels=600, n_frames=n_frames, t_max=3.0)
    trunc = calibrate(exp, CAL_CNR0, n_runs=N_RUNS_CAL,
                      model=GaussianModel(max_var=MAX_VAR))
    np.savez(path, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid,
             framevar_grid=trunc.framevar_grid)
    return trunc


def cnr_sigma2_calibration(pixel_x: np.ndarray) -> TruncationModel:
    """Joint (CNR, sigma^2) calibration on the data's pixel grid; cached.

    The 1-D D-aware lookup above is keyed on CNR alone, but this bed's spot
    grows ~8x in sigma (~64x in sigma^2) over the usable band, and var_bias
    scales with sigma^2. Its calibration reaches each CNR along its own
    peak-decay path (amplitude ~ exp(-t/tau)), while the data's CNR also falls
    as sigma_0/sigma(t) as the spot spreads, so one CNR value pools calibration
    frames from sigma^2 = 625 to ~38,000 nm^2. np.interp through those
    unsmoothed points returns erratic var_bias in the late frames above the
    floor (e.g. 2031, 7773, 529 nm^2 at CNR 1.71, 1.56, 1.48 for x = 0.25) and
    over-subtracts on average. Keying on (CNR, sigma^2), with the fixed-point
    inversion in TruncationModel.var_bias, removes the path dependence. The
    grid is independent of D, so one calibration serves every fraction.
    """
    path = OUTPUT / (f"cal_highn_cnrsig2_g{int(G_MAX)}.npz" if N_RUNS_CAL2 == 600
                   else f"cal_highn_cnrsig2_g{int(G_MAX)}_n{N_RUNS_CAL2}_dense.npz")
    if path.exists():
        grid = CalibrationGrid.load(path)
        if grid.pixel_x is None or not np.allclose(grid.pixel_x, pixel_x):
            raise RuntimeError(f"{path.name} was built on a different pixel grid")
        return TruncationModel.from_calibration(grid)
    grid = calibrate_cnr_sigma2(
        GaussianModel(max_var=MAX_VAR), pixel_x=pixel_x,
        sigma_grid=SIGMA_GRID_NM, cnr_grid=CAL2_CNR, n_runs=N_RUNS_CAL2,
        seed=4242, model_kwargs={"max_var": MAX_VAR}, verbose=True,
        n_workers=int(os.environ.get("FOLDED_DEBIAS_WORKERS") or 0) or None)
    grid.save(path)
    return TruncationModel.from_calibration(grid)


def per_frame_means(case, trunc, cnr0, n_seeds):
    """Mean raw and corrected fitted variance per frame over many seeds + per-frame CNR."""
    peak0 = float(case.profiles[0].max())
    ns = peak0 / cnr0
    cnrf = case.profiles.max(axis=1) / ns
    nfr = case.profiles.shape[0]
    raw = np.full((n_seeds, nfr), np.nan)
    cor = np.full((n_seeds, nfr), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(7919 * int(cnr0) + s)
        noisy = case.profiles + rng.normal(0.0, ns, case.profiles.shape)
        vh = fit_each_row(case.x, noisy)
        raw[s] = vh
        if KEY == "smoothed" and trunc.is_2d:
            cor[s] = vh - trunc.var_bias(cnrf, smoothed_key(case.t, vh, cnrf))
        else:
            cor[s] = corrected_variance("folded_normal", vh, cnrf, trunc)
    return np.nanmean(raw, axis=0), np.nanmean(cor, axis=0), cnrf


def smoothed_key(t, vh, cnrf):
    """Per-measurement sigma^2 key for the joint lookup: a robust line through
    this measurement's own fitted variances above the floor.

    Keying the joint lookup on each frame's own sigma_hat^2 fails at low CNR:
    the fit scatters far toward the grid edge, the fixed-point inversion reads a
    large bias there, and the mean over measurements is over-subtracted (x = 0.5,
    CNR0 = 10: -1274 nm^2 between the floor and CNR 2, against +815 naive). A
    straight-line (Theil-Sen) fit of sigma_hat^2 against t uses the whole
    measurement and brings that band to +264 nm^2. It assumes sigma^2(t) is
    linear, true for this bed's normal diffusion; it is a trend model, so it is
    an option here, not the library default.
    """
    from scipy.stats import theilslopes
    use = (cnrf >= FLOOR_CNR) & np.isfinite(vh)
    if use.sum() < 3:
        return vh
    slope, icpt, _, _ = theilslopes(vh[use], t[use])
    return np.where(cnrf >= FLOOR_CNR, icpt + slope * t, vh)


def short_window_D(case, trunc, cnr0, n_seeds):
    """Naive vs corrected D over the usable window (CNR >= floor)."""
    mraw, mcor, cnrf = per_frame_means(case, trunc, cnr0, n_seeds)
    use = cnrf >= FLOOR_CNR
    d_naive, a_naive = fit_msd_powerlaw(case.t, mraw, mask=use)
    d_corr, a_corr = fit_msd_powerlaw(case.t, mcor, mask=use)
    return dict(n_use=int(use.sum()), cnr_min=float(cnrf[use].min()) if use.any() else np.nan,
                t_max_use=float(case.t[use].max()) if use.any() else np.nan,
                d_naive=d_naive, a_naive=a_naive, d_corr=d_corr, a_corr=a_corr)


def main():
    # "cnr_sigma2" (default) keys the correction on (CNR, sigma^2); "daware"
    # reproduces the earlier 1-D D-aware lookup, which mis-subtracts here (see
    # cnr_sigma2_calibration).
    mode = sys.argv[1] if len(sys.argv) > 1 else "cnr_sigma2"
    if mode not in ("cnr_sigma2", "daware"):
        sys.exit("usage: br_halide_highn_perframe.py [cnr_sigma2|daware]")
    tag = "" if mode == "cnr_sigma2" else "_daware"
    if mode == "cnr_sigma2":
        tag += f"_n{N_RUNS_CAL2}" + ("_smoothed" if KEY == "smoothed" else "")
    global CNR0
    if len(sys.argv) > 2:
        CNR0 = float(sys.argv[2])
        tag += f"_cnr0{CNR0:g}"
    cases = [subsample(c, STRIDE) for c in load_all()]
    if not cases:
        print("No high-N maps found.")
        return
    label = (f"joint (CNR, sigma^2) D-free calibration, {N_RUNS_CAL2} runs/cell, "
             f"{KEY} key" if mode == "cnr_sigma2"
             else "1-D D-aware calibration, CNR-only lookup")

    lines = ["=" * 76,
             f"High-N Br-I: per-frame residual + short-window D ({label})",
             f"floor CNR = {FLOOR_CNR}; n_seeds = {N_SEEDS}; panel CNR0 = {CNR0:.0f}",
             f"width-fit bound: max_var = (G*sigma0)^2, G = {G_MAX:.0f} "
             f"(sigma capped at {G_MAX * SIGMA0_HINT_NM:.0f} nm)",
             "=" * 76, ""]
    panels = []
    for case in cases:
        d_true, a_true = fit_msd_powerlaw(case.t, case.var_true)
        if mode == "cnr_sigma2":
            trunc = cnr_sigma2_calibration(case.x)
        else:
            trunc = daware_calibration(d_true, case.name.split("=")[-1],
                                       case.profiles.shape[0])

        mraw, mcor, cnrf = per_frame_means(case, trunc, CNR0, N_SEEDS)
        use = cnrf >= FLOOR_CNR
        resid_naive = mraw - case.var_true
        resid_corr = mcor - case.var_true
        d_n, _ = fit_msd_powerlaw(case.t, mraw, mask=use)
        d_c, _ = fit_msd_powerlaw(case.t, mcor, mask=use)
        panels.append((case, cnrf, mraw, mcor, use, d_true, d_n, d_c))

        lines.append("-" * 76)
        lines.append(f"=== {case.name} ===  truth D={d_true:.4g} nm^2/ns, alpha={a_true:.3f}")
        lines.append(f"  usable window at CNR0={CNR0:.0f}: {use.sum()} frames, "
                     f"t<= {case.t[use].max():.2f} ns, CNR down to {cnrf[use].min():.2f}")
        lines.append(f"  mean |residual| over usable band: naive "
                     f"{np.nanmean(np.abs(resid_naive[use])):.0f} -> corrected "
                     f"{np.nanmean(np.abs(resid_corr[use])):.0f} nm^2 "
                     f"(mean signed: {np.nanmean(resid_naive[use]):+.0f} -> "
                     f"{np.nanmean(resid_corr[use]):+.0f} nm^2)")
        # The noise correction addresses distortion (1) only; var_true also
        # carries the shape / emission-to-carrier gap (the noise-free fit's
        # offset from the carrier second moment). Report the residual against
        # the noise-free fit of the same profile as the clean metric.
        nf = fit_each_row(case.x, case.profiles)
        lines.append(f"  vs noise-free fit (noise only): mean |residual| naive "
                     f"{np.nanmean(np.abs((mraw - nf)[use])):.0f} -> corrected "
                     f"{np.nanmean(np.abs((mcor - nf)[use])):.0f} nm^2 (mean signed: "
                     f"{np.nanmean((mraw - nf)[use]):+.0f} -> "
                     f"{np.nanmean((mcor - nf)[use]):+.0f} nm^2); noise-free fit "
                     f"minus var_true over the band: {np.nanmean((nf - case.var_true)[use]):+.0f} nm^2")
        lines.append("  per frame (usable band): t (ns) | CNR | var_true | "
                     "resid naive | resid corrected | noise-free fit - var_true  (nm^2)")
        for i in np.flatnonzero(use):
            lines.append(f"    {case.t[i]:5.2f} | {cnrf[i]:5.2f} | "
                         f"{case.var_true[i]:7.0f} | {resid_naive[i]:+7.0f} | "
                         f"{resid_corr[i]:+7.0f} | {nf[i] - case.var_true[i]:+7.0f}")
        lines.append("  short-window D (usable band):  CNR0 | D naive/corr (ratio to truth) "
                     "| alpha naive/corr")
        for c0 in CNR0_TABLE:
            r = short_window_D(case, trunc, c0, N_SEEDS)
            rn = r["d_naive"] / d_true if np.isfinite(r["d_naive"]) else np.nan
            rc = r["d_corr"] / d_true if np.isfinite(r["d_corr"]) else np.nan
            lines.append(f"    {c0:5.0f} | {r['d_naive']:.4g}/{r['d_corr']:.4g} "
                         f"({rn:.2f}/{rc:.2f}) | {r['a_naive']:.3f}/{r['a_corr']:.3f}"
                         f"  [{r['n_use']} frames]")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT / f"br_halide_highn_perframe{tag}.txt").write_text(report)
    make_figure(panels, OUTPUT / f"br_halide_highn_perframe{tag}.png")


def make_figure(panels, path):
    """Poster figure: per-frame fitted variance vs time, naive vs corrected.

    Spanish-flag palette (venue-appropriate, plot_style.md): blue = corrected
    (the reference the fit should recover), red = naive (the salient bias),
    black line = carrier truth. The shaded band marks frames below the CNR floor
    that are excluded from the slope fit. The time axis is focused on the usable
    window plus the bias onset; the deep-noise tail is dropped.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # Spanish-flag palette (plot_style.md): blue = corrected (recovered
    # reference), red = naive (the salient bias), indigo = carrier truth.
    truth_indigo = "#540d6e"
    naive_red = "#c60b1e"
    corr_blue = "#0077c8"
    floor_grey = "#bdbdbd"
    font = 10
    S = 1.0e6            # nm^2 -> micron^2
    XMAX = 2.0           # common time axis (ns)
    YMAX = 0.030         # common variance axis (micron^2); clears panel (a)'s data
    letters = "abcdefgh"

    n = len(panels)
    # Shared y-axis, panels abutting: values only on the left, tick marks on all.
    fig, axes = plt.subplots(1, n, layout="constrained", sharey=True,
                             figsize=(5.0 * n * INCH, 6.0 * INCH), squeeze=False)
    for i, (ax, (case, cnrf, mraw, mcor, use, d_true, d_n, d_c)) in enumerate(
            zip(axes[0], panels)):
        t = case.t
        m = t <= XMAX
        t_floor = t[~use].min() if (~use).any() else XMAX
        if t_floor < XMAX:
            ax.axvspan(t_floor, XMAX, color=floor_grey, alpha=0.30, lw=0)
        ax.plot(t[m], case.var_true[m] / S, "-", color=truth_indigo, lw=1.6,
                label="reference", zorder=3)
        ax.plot(t[m], mraw[m] / S, "o", color=naive_red, ms=3.5,
                label="naive fit", zorder=4)
        ax.plot(t[m], mcor[m] / S, "s", color=corr_blue, ms=3.5,
                label="corrected", zorder=5)
        ax.set_xlim(0, XMAX)
        ax.set_ylim(0, YMAX)
        ax.set_xticks([0, 1, 2])
        ax.yaxis.set_major_locator(MultipleLocator(0.005))
        # Inward ticks, bottom + left sides only (no top/right).
        ax.tick_params(axis="both", which="both", direction="in",
                       top=False, right=False, labelsize=font)
        if i != 0:
            ax.tick_params(axis="y", labelleft=False)   # keep marks, drop values
        ax.set_xlabel("time (ns)", fontsize=font)
        ax.text(0.06, 0.94, f"({letters[i]})", ha="left", va="top",
                fontsize=font, fontweight="bold", transform=ax.transAxes)
    axes[0][0].set_ylabel(r"fitted $\sigma^2$  ($\mu$m$^2$)", fontsize=font)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=font, loc="outside lower center", ncol=3)
    fig.patch.set_facecolor("w"); fig.patch.set_alpha(1)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
