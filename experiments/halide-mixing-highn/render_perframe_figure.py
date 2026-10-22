"""Render the per-frame figure for the high-N Br-I bed, with its results table.

Three panels, one per Br fraction (x = 0.25, 0.5, 0.75), at initial CNR 15,
the lowest initial CNR at which the corrected D does not depend on the
joint-lookup key (``key_sensitivity.py``). Each panel shows, per frame, the mean
fitted variance over the seeds minus the Gaussian fit of the noise-free profile,
for the naive fit and for the corrected fit, with standard errors. That residual
isolates the noise bias, which is what the correction removes. The separate
offset between the noise-free fit and the carrier reference (the shape and
emission-to-carrier gap, which no noise correction addresses) is drawn as its
own line.

The correction uses the dense joint (CNR, sigma^2) calibration and the
iterated measurement-level key (``iterated_key_correction``). The table
``br_halide_highn_perframe.txt`` gives the short-window D against the carrier
reference and against the noise-free fit, with standard errors from resampling
the seeds.

    .venv/Scripts/python.exe experiments/halide-mixing-highn/render_perframe_figure.py

Needs the pshare (collapsed maps) and the dense calibration in ``output/``
(``output/build_highn_dense_cal.py`` or ``br_halide_highn_perframe.cnr_sigma2_calibration``).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "halide-mixing-brI-legacy"))

CNR0_FIG = 15.0
N_SEEDS_FIG = int(os.environ.get("HIGHN_FIG_SEEDS", 800))
N_BOOT = 200
OUT_PNG = Path(HERE) / "br_halide_highn_perframe.png"
OUT_TXT = Path(HERE) / "br_halide_highn_perframe.txt"
INCH = 1 / 2.54


def _panel(frac):
    """Per-seed naive and corrected variances for one map at CNR0_FIG."""
    import br_halide_highn_perframe as hp
    from br_halide_highn_validation import STRIDE, subsample
    from br_highn_data import load_all
    case = [subsample(c, STRIDE) for c in load_all() if c.name.endswith(frac)][0]
    trunc = hp.cnr_sigma2_calibration(case.x)
    ns = float(case.profiles[0].max()) / CNR0_FIG
    cnrf = case.profiles.max(axis=1) / ns
    raw = np.full((N_SEEDS_FIG, cnrf.size), np.nan)
    cor = np.full_like(raw, np.nan)
    for s in range(N_SEEDS_FIG):
        rng = np.random.default_rng(7919 * int(CNR0_FIG) + s)
        vh = hp.fit_each_row(case.x, case.profiles + rng.normal(0.0, ns, case.profiles.shape))
        raw[s] = vh
        cor[s] = hp.iterated_key_correction(case.t, vh, cnrf, trunc)
    nf = hp.fit_each_row(case.x, case.profiles)
    return frac, case.t, case.var_true, nf, cnrf, raw, cor


def _d(t, traj, use):
    from br_halide_validation import fit_msd_powerlaw
    d, _ = fit_msd_powerlaw(t, traj, mask=use)
    return d


def main():
    import br_halide_highn_perframe as hp
    from br_halide_validation import fit_msd_powerlaw

    if hp.N_RUNS_CAL2 != 4000:
        sys.exit("expects the dense calibration (HIGHN_CAL2_RUNS unset or 4000)")
    with ProcessPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(_panel, ["0.25", "0.5", "0.75"]))

    rng = np.random.default_rng(57)
    lines = ["=" * 76,
             f"High-N Br-I per-frame figure: CNR0 = {CNR0_FIG:g}, {N_SEEDS_FIG} seeds, "
             "dense joint calibration, iterated key",
             f"floor CNR {hp.FLOOR_CNR}; residuals in nm^2 against the noise-free fit; "
             f"D SE from {N_BOOT} resamples of the seeds",
             "=" * 76]
    panels = []
    for frac, t, var_true, nf, cnrf, raw, cor in results:
        use = cnrf >= hp.FLOOR_CNR
        d_ref, _ = fit_msd_powerlaw(t, var_true)
        d_nf = _d(t, nf, use)
        mean_raw, mean_cor = np.nanmean(raw, axis=0), np.nanmean(cor, axis=0)
        se_raw = np.nanstd(raw, axis=0, ddof=1) / np.sqrt(len(raw))
        se_cor = np.nanstd(cor, axis=0, ddof=1) / np.sqrt(len(cor))
        dn, dc = _d(t, mean_raw, use), _d(t, mean_cor, use)
        boot = np.array([[_d(t, np.nanmean(a[idx], axis=0), use) for a in (raw, cor)]
                         for idx in (rng.integers(0, len(raw), len(raw))
                                     for _ in range(N_BOOT))])
        sen, sec = np.nanstd(boot, axis=0, ddof=1)
        rn, rc = (mean_raw - nf)[use], (mean_cor - nf)[use]
        lines += [
            f"\nx = {frac}: {use.sum()} usable frames, t <= {t[use].max():.2f} ns, "
            f"CNR down to {cnrf[use].min():.2f}",
            f"  residual vs noise-free fit: mean |naive| {np.nanmean(np.abs(rn)):.0f}, "
            f"mean |corrected| {np.nanmean(np.abs(rc)):.0f} nm^2; signed "
            f"{np.nanmean(rn):+.0f} -> {np.nanmean(rc):+.0f} nm^2",
            f"  carrier gap (noise-free fit - reference) over the band: mean "
            f"{np.nanmean((nf - var_true)[use]):+.0f} nm^2",
            f"  short-window D / D_ref:      naive {dn / d_ref:.3f} ({sen / d_ref:.3f}), "
            f"corrected {dc / d_ref:.3f} ({sec / d_ref:.3f})",
            f"  short-window D / D_noisefree: naive {dn / d_nf:.3f} ({sen / d_nf:.3f}), "
            f"corrected {dc / d_nf:.3f} ({sec / d_nf:.3f})",
            f"  D_ref = {d_ref:.4g} nm^2/ns; D of the noise-free fit = {d_nf:.4g} nm^2/ns",
        ]
        panels.append((frac, t, cnrf, use, mean_raw - nf, se_raw, mean_cor - nf,
                       se_cor, nf - var_true))
    report = "\n".join(lines)
    print(report)
    OUT_TXT.write_text(report + "\n")
    make_residual_figure(panels, OUT_PNG)


def make_residual_figure(panels, path):
    """Per-frame residual against the noise-free fit, naive and corrected.

    Spanish-flag set: flag red = naive (the bias), blue = corrected, indigo =
    the carrier gap, flag yellow = the zero line and the sub-floor band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    red, blue, indigo, yellow = "#c60b1e", "#0077c8", "#540d6e", "#ffc400"
    font = 9
    xmax = max(p[1][p[3]].max() for p in panels) + 0.1
    # Scale y to the usable frames: the sub-floor frames are excluded from the
    # fit, are not plotted, and are marked only by the shaded band.
    ylim = 1.15 * max(np.nanmax(np.abs(np.concatenate([
        (p[4] + np.sign(p[4]) * p[5])[p[3]], (p[6] + np.sign(p[6]) * p[7])[p[3]],
        p[8][p[3]]]))) for p in panels)
    fig, axes = plt.subplots(1, len(panels), layout="constrained", sharex=True,
                             sharey=True, figsize=(17 * INCH, 6.5 * INCH))
    for i, (ax, (frac, t, cnrf, use, rn, sen, rc, sec, gap)) in enumerate(zip(axes, panels)):
        m = use & (t <= xmax)
        t_floor = t[~use].min() if (~use).any() else xmax
        if t_floor < xmax:
            ax.axvspan(t_floor, xmax, color=yellow, alpha=0.18, lw=0)
        ax.axhline(0.0, color=yellow, lw=1.2, zorder=1)
        ax.plot(t[m], gap[m], "-", color=indigo, lw=2, label="carrier gap", zorder=2)
        ax.errorbar(t[m], rn[m], yerr=sen[m], fmt="o", color=red, ms=3, lw=0.8,
                    label="naive", zorder=3)
        ax.errorbar(t[m], rc[m], yerr=sec[m], fmt="s", color=blue, ms=3, lw=0.8,
                    label="corrected", zorder=4)
        ax.set_xlim(0, xmax)
        ax.set_ylim(-ylim, ylim)
        ax.set_facecolor("white")
        ax.tick_params(axis="both", which="both", direction="in", top=False,
                       right=False, labelsize=font)
        ax.text(0.04, 0.95, f"({'abc'[i]})", ha="left", va="top", fontsize=font,
                fontweight="bold", transform=ax.transAxes)
        ax.set_xlabel(r"$t \, (\text{ns})$", fontsize=font)
    axes[0].set_ylabel(r"$\Delta\sigma^2 \, (\text{nm}^2)$", fontsize=font)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=font, loc="outside lower center", ncol=3)
    fig.patch.set_alpha(0)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
