"""Per-frame residual + short-window D on the high-N Br-I bed.

The full-3-ns D collapse is brittle here: the peak drops ~147x over the window,
so most late frames fall below CNR ~1 and the slope fit has a thin lever arm.
The honest test is the project's actual metric -- the per-frame mean residual
``E[corrected] - var_true`` over many seeds -- restricted to the *usable window*
(frames whose per-frame CNR stays above the floor). Over that band we ask: does
the correction reduce the upward bias the naive fit carries, and does the
short-window D move toward the truth?

The calibration is built D-aware at each fraction's fitted D (the quick,
sound choice for a known-D validation bed; the D-free 2-D (CNR, sigma^2)
calibration is the separate real-data path). It is cached per fraction.

Run from the repository root (needs the pshare)::

    FOLDED_DEBIAS_WORKERS=0 .venv/Scripts/python.exe experiments/halide-mixing-highn/br_halide_highn_perframe.py

Outputs ``output/br_halide_highn_perframe.{png,txt}`` and caches each D-aware
calibration to ``output/cal_highn_daware_x<frac>.npz``.
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

from folded_debias import GaussianModel  # noqa: E402
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
FLOOR_CNR = 1.3      # project floor: folded-normal over-subtracts below ~1.3
CAL_CNR0 = np.geomspace(1.2, 60.0, 30)
N_RUNS_CAL = 600


def daware_calibration(d_true: float, frac: str, n_frames: int) -> TruncationModel:
    """D-aware calibration at this fraction's D and geometry; cached per fraction."""
    path = OUTPUT / f"cal_highn_daware_g{int(G_MAX)}_x{frac}.npz"
    if path.exists():
        d = np.load(path)
        return TruncationModel(cnr_grid=d["cnr_grid"], p_grid=d["p_grid"],
                               relvar_grid=d["relvar_grid"],
                               varbias_grid=d["varbias_grid"])
    exp = Experiment(sigma0=25.0, D=float(d_true), tau=1.0, half_window=60.0,
                     n_pixels=600, n_frames=n_frames, t_max=3.0)
    trunc = calibrate(exp, CAL_CNR0, n_runs=N_RUNS_CAL,
                      model=GaussianModel(max_var=MAX_VAR))
    np.savez(path, cnr_grid=trunc.cnr_grid, p_grid=trunc.p_grid,
             relvar_grid=trunc.relvar_grid, varbias_grid=trunc.varbias_grid)
    return trunc


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
        cor[s] = corrected_variance("folded_normal", vh, cnrf, trunc)
    return np.nanmean(raw, axis=0), np.nanmean(cor, axis=0), cnrf


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
    cases = [subsample(c, STRIDE) for c in load_all()]
    if not cases:
        print("No high-N maps found.")
        return
    lines = ["=" * 76,
             "High-N Br-I: per-frame residual + short-window D (D-aware calibration)",
             f"floor CNR = {FLOOR_CNR}; n_seeds = {N_SEEDS}; panel CNR0 = {CNR0:.0f}",
             f"width-fit bound: max_var = (G*sigma0)^2, G = {G_MAX:.0f} "
             f"(sigma capped at {G_MAX * SIGMA0_HINT_NM:.0f} nm)",
             "=" * 76, ""]
    panels = []
    for case in cases:
        d_true, a_true = fit_msd_powerlaw(case.t, case.var_true)
        trunc = daware_calibration(d_true, case.name.split("=")[-1], case.profiles.shape[0])

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
                     f"{np.nanmean(np.abs(resid_corr[use])):.0f} nm^2")
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
    (OUTPUT / "br_halide_highn_perframe.txt").write_text(report)
    make_figure(panels, OUTPUT / "br_halide_highn_perframe.png")


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
