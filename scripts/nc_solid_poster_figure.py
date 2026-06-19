"""Poster figure: noise bias of the MSD exponent in the nanocrystal-solid regime.

One representative curve (the C16 / `small` bed) rather than three near-identical
ones, over the realistic measurement regime CNR_0 >= 30 (no one runs an MSD
experiment below that). The point is the residual upward bias that survives well
into "good" contrast: at CNR_0 ~ 30 the apparent exponent reads ~0.09 high, falls
below ~0.01 by CNR_0 ~ 90, and clears into Monte Carlo noise near CNR_0 ~ 200. The
curve smooths only by honest averaging of many INDEPENDENT runs (noise is drawn
afresh per CNR_0; draws are never shared across CNR_0, which would impose a false
smoothness). Error bars are the bootstrap standard error of the bias exponent over
the seeds -- the per-seed-exponent s.d. / sqrt(n_seeds) underestimates it, because
the exponent of the mean trajectory is a nonlinear functional of the seeds. Below
CNR_0 ~ 30 the exponent climbs to the ballistic ceiling (alpha = 2); that is
off-figure and noted in the caption.

Reuses the parallel sweep machinery in `nc_solid_alpha_vs_cnr.py`. Heavy
(one sample x dense grid x many seeds); run with WEIGHTING_WALL_WORKERS=0 to use
all cores. Pin the BLAS thread count to 1 as well: the fits are tiny nonlinear
least-squares that BLAS threading does not help, and one BLAS thread per worker
times all cores oversubscribes and can exhaust memory (OpenBLAS allocation
failure) -- most acute for the inset's low-CNR_0 points, whose ill-conditioned
fits iterate to maxfev::

    $env:WEIGHTING_WALL_WORKERS = 0       # PowerShell; all cores
    $env:OPENBLAS_NUM_THREADS = 1
    $env:OMP_NUM_THREADS = 1
    .venv/Scripts/python.exe scripts/nc_solid_poster_figure.py

The two sweeps cache to ``output/nc_solid_poster_{figure,inset}.npz``; with both
present the script only redraws (delete a cache to recompute). Outputs
``output/nc_solid_poster_figure.{png,txt}``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter, ScalarFormatter  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from curved_text import curved_text  # noqa: E402

import nc_solid_alpha_vs_cnr as base  # noqa: E402  (reuse compute_row / run_tasks)
from weighting_wall.noise import WhiteNoise  # noqa: E402
from mc_experiments.disordered_nc_solid.nc_data import load_case  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
inch = 1 / 2.54
# Spanish-flag palette (see the plot style guide): red marks the
# salient measured series, yellow (gualda) the reference value.
FLAG_RED = "#C60B1E"
FLAG_YELLOW = "#FFC400"

SAMPLE = "small"          # C16 -- the most subdiffusive, representative of the regime
N_SEEDS = 64000           # honest averaging: more independent runs shrink the
                          # per-point Monte Carlo scatter (~1/sqrt(N)) so the
                          # curve smooths without correlating noise across CNR_0
# A real MSD experiment is not run below CNR_0 ~ 30, so the figure covers the
# realistic regime 30-250: there the apparent exponent still carries a residual
# upward bias that only clears near CNR_0 ~ 200. (Below 30 it climbs to the
# ballistic ceiling; the caption notes this, off-figure.)
CNR_GRID = tuple(float(v) for v in np.unique(np.round(
    np.logspace(np.log10(30), np.log10(250), 20))))

# Inset: the off-panel low-CNR_0 behaviour, where the estimate climbs to the
# ballistic ceiling. It only has to show the divergence, not resolve a 0.01
# residual, so it uses a coarser grid down to CNR_0 = 3 and far fewer seeds.
INSET_GRID = tuple(float(v) for v in np.unique(np.round(
    np.logspace(np.log10(3), np.log10(250), 14))))
INSET_SEEDS = 4000


def _bias_task(packed):
    """Apparent MSD exponent at one CNR_0 from INDEPENDENT noise realisations.

    Each seed draws its own white-noise pattern, and the seed is offset by CNR_0
    so neighbouring points are statistically independent: the curve smooths only by
    honest averaging of more runs, never by sharing noise draws across CNR_0. The
    point estimate is the exponent of the seed-mean MSD (the bias that survives
    averaging many measurements). The error bar is the bootstrap standard error of
    that exponent over the seeds -- the spread of the per-seed exponent divided by
    sqrt(N) underestimates it, because the exponent of the mean trajectory is a
    nonlinear functional of the seeds, not their average. Returns
    ``(cnr0, bias_alpha, bootstrap_se)``.
    """
    x, t, profiles, cnr0, n_seeds = packed
    white = WhiteNoise()
    base_seed = int(round(cnr0)) * 1_000_003          # independent across CNR_0
    rows = np.empty((n_seeds, t.size))
    for s in range(n_seeds):
        rows[s] = base.fit_each_row(x, white.sample(profiles, cnr0,
                                                     np.random.default_rng(base_seed + s)))
    _, a_bias = base.fit_powerlaw(t, np.nanmean(rows, axis=0))
    rb = np.random.default_rng(base_seed + 7)
    boot = np.empty(200)
    for b in range(boot.size):
        idx = rb.integers(0, n_seeds, n_seeds)
        _, boot[b] = base.fit_powerlaw(t, np.nanmean(rows[idx], axis=0))
    return float(cnr0), float(a_bias), float(np.nanstd(boot))


def _run_sweep(grid, n_seeds, cache_path):
    """Run (or load from cache) the CNR_0 sweep; return (cnr, alpha, unc, a_true)."""
    meta = dict(sample=SAMPLE, n_seeds=n_seeds, grid=tuple(grid), scheme="indep-boot")
    if cache_path.exists() and \
            dict(np.load(cache_path, allow_pickle=True)["meta"].item()) == meta:
        d = np.load(cache_path, allow_pickle=True)
        print(f"Loaded {cache_path.name} (delete to recompute).")
        return d["cnr"], d["alpha"], d["unc"], float(d["a_true"])
    from concurrent.futures import ProcessPoolExecutor
    case = load_case(SAMPLE, denoise=True)
    _, a_true = base.fit_powerlaw(case.t, case.var_true)
    tasks = [(case.x, case.t, case.profiles, c, n_seeds) for c in grid]
    nw = base._n_workers()
    print(f"Running {len(tasks)} CNR_0 x {n_seeds} independent seeds on "
          f"{nw} worker(s) -> {cache_path.name}...")
    if nw == 1:
        out = [_bias_task(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            out = list(ex.map(_bias_task, tasks))
    cnr = np.array([o[0] for o in out])
    alpha = np.array([o[1] for o in out])
    unc = np.array([o[2] for o in out])
    np.savez(cache_path, cnr=cnr, alpha=alpha, unc=unc, a_true=a_true, meta=meta)
    return cnr, alpha, unc, a_true


def main() -> None:
    # cache the expensive sweeps so plot tweaks do not recompute them
    cnr, alpha, unc, a_true = _run_sweep(CNR_GRID, N_SEEDS,
                                         OUTPUT / "nc_solid_poster_figure.npz")
    icnr, ialpha, _, _ = _run_sweep(INSET_GRID, INSET_SEEDS,
                                    OUTPUT / "nc_solid_poster_inset.npz")
    lines = [f"NC-solid poster figure: {SAMPLE}/C16, N_SEEDS={N_SEEDS}, "
             f"true alpha={a_true:.3f}",
             "  CNR_0 | alpha_bias +/- unc | residual (alpha-true)"]
    for c, a, u in zip(cnr, alpha, unc):
        lines.append(f"  {c:6.0f} | {a:.3f} +/- {u:.3f} | {a - a_true:+.3f}")
    (OUTPUT / "nc_solid_poster_figure.txt").write_text("\n".join(lines))
    print("\n".join(lines))

    # --- figure: residual bias over the realistic regime (CNR_0 >= 30) ------
    fig, ax = plt.subplots(layout="constrained", figsize=(11 * inch, 8 * inch))
    fs = 10
    ax.set_xscale("log")
    ax.set_xlim(cnr.min() * 0.93, cnr.max() * 1.07)
    ax.set_ylim(a_true - 0.008, max(alpha) + 0.012)

    # flag-yellow reference line at the bed's noise-free exponent, labelled in
    # black; "nominal" (not "true") since it is the simulation bed's input value
    ax.axhline(a_true, color=FLAG_YELLOW, lw=2.6, zorder=1)
    ax.text(cnr.min() * 0.97, a_true + 0.0016, rf"nominal $\alpha \approx {a_true:.2f}$",
            fontsize=fs, color="black", ha="left", va="bottom")

    # measured points in flag red: markers with error bars only, no connecting
    # line (each point is an independent Monte Carlo estimate, not a fitted curve)
    ax.errorbar(cnr, alpha, yerr=unc, fmt="o", color=FLAG_RED, ms=4, lw=0,
                elinewidth=1.2, capsize=2, zorder=4)

    # invisible smooth fit through the points (residual decays as a power law in
    # CNR_0), used only as the guide curve for the label that follows the points
    def _resid(c, amp, expo):
        return a_true + amp * c ** (-expo)
    (amp, expo), _ = curve_fit(_resid, cnr, alpha, p0=[100.0, 2.0], maxfev=20000)
    seg = (cnr >= 34) & (cnr <= 125)
    xlab = np.logspace(np.log10(cnr[seg].min()), np.log10(cnr[seg].max()), 200)
    curved_text(ax, xlab, _resid(xlab, amp, expo), r"estimated $\alpha$ from MSD fit",
                pos=0.5, anchor="center", offset=11.0, color=FLAG_RED, fontsize=fs,
                zorder=6)

    ax.set_xlabel(r"initial contrast-to-noise CNR$_0$", fontsize=fs)
    ax.set_ylabel(r"MSD exponent $\alpha$", fontsize=fs)
    ax.set_xticks([30, 50, 100, 200])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs)

    # inset: the off-panel low-CNR_0 behaviour -- as contrast drops below the
    # measured regime the estimate climbs to the ballistic ceiling (alpha = 2)
    axin = ax.inset_axes([0.46, 0.52, 0.50, 0.44])
    axin.set_facecolor("white")
    axin.set_zorder(5)
    axin.set_xscale("log")
    axin.axhline(2.0, ls=(0, (4, 3)), color="#555555", lw=1.0)
    axin.axhline(a_true, color=FLAG_YELLOW, lw=1.6)
    axin.axvspan(30.0, icnr.max(), color="#c9c9c9", alpha=0.35, lw=0)
    axin.plot(icnr, ialpha, "o", color=FLAG_RED, ms=2.6)
    axin.set_xlim(icnr.min() * 0.85, icnr.max() * 1.15)
    axin.set_ylim(a_true - 0.05, 2.10)
    axin.text(icnr.min(), 1.95, "ballistic", fontsize=fs - 3,
              color="#555555", ha="left", va="top")
    axin.text((30.0 * icnr.max()) ** 0.5, 1.5, "main panel", fontsize=fs - 3,
              color="#6f6f6f", ha="center", va="center")
    axin.set_xticks([3, 10, 30, 100])
    axin.xaxis.set_major_formatter(ScalarFormatter())
    axin.xaxis.set_minor_formatter(NullFormatter())
    axin.set_yticks([1.0, 1.5, 2.0])
    axin.set_xlabel(r"CNR$_0$", fontsize=fs - 2, labelpad=1)
    axin.set_ylabel(r"$\alpha$", fontsize=fs - 2, labelpad=2)
    axin.tick_params(axis="both", which="both", direction="in", top=True,
                     right=True, labelsize=fs - 3)

    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    path = OUTPUT / "nc_solid_poster_figure.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
