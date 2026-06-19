"""Poster figure: noise bias of the MSD exponent in the nanocrystal-solid regime.

One representative curve (the C16 / `small` bed) rather than three near-identical
ones, on a dense CNR_0 grid with many noise realisations so the curve is smooth,
plus an inset that zooms into the high-contrast approach -- to see whether the
bias fully vanishes as CNR_0 grows or plateaus slightly above the true exponent.
The inset carries error bars (the bias estimate's own uncertainty,
per-seed-s.d. / sqrt(n_seeds)) so a residual can be told from Monte Carlo scatter.

Reuses the parallel sweep machinery in `nc_solid_alpha_vs_cnr.py`. Heavy
(one sample x dense grid x many seeds); run with WEIGHTING_WALL_WORKERS=0::

    .venv/Scripts/python.exe scripts/nc_solid_poster_figure.py

Outputs ``output/nc_solid_poster_figure.{png,txt}``.
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

import nc_solid_alpha_vs_cnr as base  # noqa: E402  (reuse compute_row / run_tasks)
from weighting_wall.noise import WhiteNoise  # noqa: E402
from mc_experiments.disordered_nc_solid.nc_data import load_case  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
inch = 1 / 2.54
BLUE, GREY = "#003f7f", "#9a9a9a"

SAMPLE = "small"          # C16 -- the most subdiffusive, representative of the regime
N_SEEDS = 8000            # enough to resolve a ~0.01 residual at high CNR
# dense grid, log-spaced, denser at the high-CNR end where the residual lives
CNR_GRID = tuple(float(v) for v in np.unique(np.round(
    np.logspace(np.log10(3), np.log10(220), 22))))


def main() -> None:
    # cache the expensive sweep so plot tweaks do not recompute it
    cache = OUTPUT / "nc_solid_poster_figure.npz"
    meta = dict(sample=SAMPLE, n_seeds=N_SEEDS, grid=tuple(CNR_GRID))
    if cache.exists() and dict(np.load(cache, allow_pickle=True)["meta"].item()) == meta:
        d = np.load(cache, allow_pickle=True)
        cnr, alpha, unc, a_true = d["cnr"], d["alpha"], d["unc"], float(d["a_true"])
        print("Loaded cache (delete the .npz to recompute).")
    else:
        case = load_case(SAMPLE, denoise=True)
        _, a_true = base.fit_powerlaw(case.t, case.var_true)
        white = WhiteNoise()
        tasks = [((c,), case.x, case.t, case.profiles, c, N_SEEDS, white, 0)
                 for c in CNR_GRID]
        print(f"Running {len(tasks)} CNR_0 x {N_SEEDS} seeds ({SAMPLE}/C16) on "
              f"{base._n_workers()} worker(s)...")
        res = base.run_tasks(tasks)
        cnr = np.array(CNR_GRID)
        alpha = np.array([res[(c,)]["a_bias_naive"] for c in CNR_GRID])
        unc = np.array([res[(c,)]["a_seed_sd"] for c in CNR_GRID]) / np.sqrt(N_SEEDS)
        np.savez(cache, cnr=cnr, alpha=alpha, unc=unc, a_true=a_true, meta=meta)
        lines = [f"NC-solid poster figure: {SAMPLE}/C16, N_SEEDS={N_SEEDS}, "
                 f"true alpha={a_true:.3f}",
                 "  CNR_0 | alpha_bias +/- unc | residual (alpha-true)"]
        for c, a, u in zip(cnr, alpha, unc):
            lines.append(f"  {c:6.0f} | {a:.3f} +/- {u:.3f} | {a - a_true:+.3f}")
        (OUTPUT / "nc_solid_poster_figure.txt").write_text("\n".join(lines))
        print("\n".join(lines))

    # --- figure: main panel + high-CNR inset --------------------------------
    fig, ax = plt.subplots(layout="constrained", figsize=(11 * inch, 9 * inch))
    fs = 10
    ax.axhline(2.0, color=GREY, lw=1.0, ls=":")
    ax.text(3.1, 1.97, "ballistic ceiling", fontsize=fs - 1, color="#6b6b6b",
            va="top")
    ax.axhline(a_true, color=GREY, lw=1.2)
    ax.text(200, a_true + 0.02, "true exponent", fontsize=fs - 1, color="#5a5a5a",
            ha="right", va="bottom")
    ax.plot(cnr, alpha, "-o", color=BLUE, ms=4, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel(r"initial contrast-to-noise  CNR$_0$", fontsize=fs)
    ax.set_ylabel(r"apparent MSD exponent  $\alpha$", fontsize=fs)
    ax.set_ylim(0.6, 2.05)
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs)

    # inset: zoom into the high-CNR approach, with the bias-estimate error bars
    hi = cnr >= 25
    axin = ax.inset_axes([0.46, 0.46, 0.5, 0.46])
    axin.axhline(a_true, color=GREY, lw=1.0)
    axin.errorbar(cnr[hi], alpha[hi], yerr=unc[hi], fmt="-o", color=BLUE, ms=3,
                  lw=1.5, capsize=2)
    axin.set_xscale("log")
    axin.set_ylim(a_true - 0.02, max(alpha[hi]) + 0.02)
    axin.set_xticks([30, 50, 100, 200])
    axin.xaxis.set_major_formatter(ScalarFormatter())
    axin.xaxis.set_minor_formatter(NullFormatter())
    axin.tick_params(axis="both", which="both", direction="in", labelsize=fs - 2)
    axin.set_xlabel(r"CNR$_0$", fontsize=fs - 2)
    axin.set_title("high-contrast approach (residual bias)", fontsize=fs - 2)

    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    path = OUTPUT / "nc_solid_poster_figure.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
