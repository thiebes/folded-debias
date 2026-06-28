"""Per-sweep and de-paritied-merged MSD views for the real-data measurements.

Each measurement is two interleaved delay sweeps (a coarse and a fine sweep,
merged by Enrique), detected by the single time reset in the acquisition order.
The two sweeps carry different sigma^2(0) baselines, so how they are zeroed
matters for D. This script produces two views per measurement, from the existing
``manifest.csv`` (the correction is already applied there -- no Monte Carlo is
rerun):

* **per-sweep** (``msd_persweep.png``): each sweep re-zeroed to *its own* t0,
  with its own inverse-variance D. Two genuinely separate diffusion
  measurements of the same sample, plotted and fit independently.
* **de-parity merged** (``msd_deparity.png``): both sweeps re-zeroed to their
  own t0 and then combined into a single MSD curve with one fit -- the
  legitimate multi-dataset merge.

Both views keep the per-frame CNR floor and the sustained-tail truncation. The
original global-baseline ``msd_corrected.png`` is left untouched for comparison.

Usage::

    .venv/Scripts/python.exe scripts/realdata_sweep_views.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from weighting_wall.fit import fit_D  # noqa: E402
from realdata_diagnostics import (MEASUREMENTS, _find_canonical,  # noqa: E402
                                  truncate_degraded_tail)

WIP = r"P:\jthiebes\WIP\realdata_bias_diagnostics"
BLUE = "#0072B2"      # sweep 0 (coarse) / merged
VERM = "#D55E00"      # sweep 1 (fine)
GOLD = "#f7941e"
GRAY = "0.6"
INCH = 1 / 2.54
LABELS = ["mixed_tmdc_comparison_all_0", "mixed_tmdc_pureS_250701",
          "mixed_tmdc_26_32_250716", "mixed_tmdc_50_50_250702",
          "WSe2_Power_var_250224_0p020"]


def _read_manifest(label):
    rows = list(csv.DictReader(open(os.path.join(WIP, label, "manifest.csv"))))
    g = lambda key: np.array([float(r[key]) for r in rows])
    return (g("time_ns"), g("var_naive_px2"), g("se_naive_px2"),
            g("var_corr_px2"), g("se_corr_px2"),
            g("keep_flag").astype(bool), g("cnr_white"))


def _folder(label):
    return next(f for lbl, f in MEASUREMENTS.values() if lbl == label)


def _batch_labels(label, t_sorted):
    """Batch index per time-sorted frame, from the acquisition-order time reset."""
    a = np.load(_find_canonical(_folder(label)), allow_pickle=True).flat[0]
    t_orig = np.asarray(a["time_arr(ns)"], float)
    batch_of, b = {}, 0
    for i, tt in enumerate(t_orig):
        if i > 0 and t_orig[i] < t_orig[i - 1]:
            b += 1
        batch_of[round(float(tt), 3)] = b
    return np.array([batch_of[round(float(tt), 3)] for tt in t_sorted])


def _fitD(t, msd, se, keep):
    """Re-zero already done by caller; CNR floor + sustained-tail, inv-var fit."""
    retained, cut = truncate_degraded_tail(t, msd, se, keep)
    w = (1.0 / np.maximum(se, 1e-9) ** 2) * retained
    return fit_D(t, msd, w), retained, cut


def _ax(fs=8):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(layout="constrained", figsize=(9 * INCH, 7 * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w")
    fig.patch.set_alpha(1)
    return fig, ax


def per_sweep(label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, vn, sn, vc, sc, keep, cnr = _read_manifest(label)
    batch = _batch_labels(label, t)
    fig, ax = _ax()
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    colors = {0: BLUE, 1: VERM}
    names = {0: "coarse", 1: "fine"}
    out_rows = []
    ylo, yhi = [], []
    for b in sorted(set(batch)):
        sel = batch == b
        tb, vcb, scb, vnb, snb, kb = (t[sel], vc[sel], sc[sel],
                                      vn[sel], sn[sel], keep[sel])
        msd_c = vcb - vcb[0]                      # re-zero to this sweep's t0
        msd_n = vnb - vnb[0]
        dC, ret, cut = _fitD(tb, msd_c, scb, kb)
        dN, _, _ = _fitD(tb, msd_n, snb, kb)
        c = colors.get(b, GRAY)
        ax.errorbar(tb[ret], msd_c[ret], yerr=scb[ret], fmt="o", color=c, ms=4,
                    capsize=2, lw=1, label=f"sweep {b} ({names.get(b,'')}): "
                                           f"D={dC.D:.2f}")
        if (kb & ~ret).any():
            ax.plot(tb[kb & ~ret], msd_c[kb & ~ret], "s", mfc="none", mec=c, ms=5)
        tt = np.linspace(tb[ret].min(), tb[ret].max(), 40) if ret.any() else tb
        ax.plot(tt, dC.intercept + 2 * dC.D * tt, "-", color=c, lw=1.6)
        if ret.any():
            ylo.append(float(np.min(msd_c[ret]))); yhi.append(float(np.max(msd_c[ret])))
        out_rows.append((b, dN.D, dC.D, int(ret.sum()), cut))
    if ylo:
        lo, hi = min(ylo), max(yhi); pad = 0.3 * (hi - lo + 1e-9)
        ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("time after pump (ns)", fontsize=8)
    ax.set_ylabel(r"MSD $\sigma^2(t)-\sigma^2(0)$ (px$^2$)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    out = os.path.join(WIP, label, "msd_persweep.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out, out_rows


def deparity_merged(label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, vn, sn, vc, sc, keep, cnr = _read_manifest(label)
    batch = _batch_labels(label, t)
    # per-batch baseline (each sweep's own t0), then merge into one MSD curve
    base_c = np.zeros_like(vc); base_n = np.zeros_like(vn)
    for b in set(batch):
        sel = batch == b
        base_c[sel] = vc[sel][0]; base_n[sel] = vn[sel][0]
    msd_c = vc - base_c; msd_n = vn - base_n
    dC, ret, cut = _fitD(t, msd_c, sc, keep)
    dN, _, _ = _fitD(t, msd_n, sn, keep)
    fig, ax = _ax()
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax.errorbar(t[ret], msd_c[ret], yerr=sc[ret], fmt="o", color=BLUE, ms=4,
                capsize=2, lw=1, label=f"corrected (de-parity): D={dC.D:.2f}")
    if (keep & ~ret).any():
        ax.plot(t[keep & ~ret], msd_c[keep & ~ret], "s", mfc="none", mec=VERM,
                ms=5, label="cut (degraded tail)")
    if (~keep).any():
        ax.plot(t[~keep], np.clip(msd_c[~keep], *ax.get_ylim()), "o", mfc="none",
                mec=GRAY, ms=4, label="CNR < floor")
    tt = np.linspace(t[ret].min(), t[ret].max(), 50)
    ax.plot(tt, dC.intercept + 2 * dC.D * tt, "-", color=BLUE, lw=2.0)
    ax.plot(tt, dN.intercept + 2 * dN.D * tt, "--", color=GOLD, lw=1.3,
            label=f"naive (de-parity): D={dN.D:.2f}")
    if ret.any():
        lo, hi = float(np.min(msd_c[ret])), float(np.max(msd_c[ret]))
        pad = 0.3 * (hi - lo + 1e-9); ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("time after pump (ns)", fontsize=8)
    ax.set_ylabel(r"MSD $\sigma^2(t)-\sigma^2(0)$ (px$^2$)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    out = os.path.join(WIP, label, "msd_deparity.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out, dN.D, dC.D, int(ret.sum()), cut


def main():
    print("=== per-sweep D (corrected, inv-var, CNR floor + sustained tail) ===")
    for label in LABELS:
        out, rows = per_sweep(label)
        for b, dn, dc, nret, cut in rows:
            cs = "-" if not np.isfinite(cut) else f"{cut:.1f}ns"
            print(f"  {label:30s} sweep {b}: n_ret={nret:2d} cut={cs:6s} "
                  f"D naive {dn:6.2f}  corrected {dc:6.2f}")
        print(f"    -> {out}")
    print("\n=== de-parity merged D ===")
    for label in LABELS:
        out, dn, dc, nret, cut = deparity_merged(label)
        cs = "-" if not np.isfinite(cut) else f"{cut:.1f}ns"
        print(f"  {label:30s} n_ret={nret:2d} cut={cs:6s} "
              f"D naive {dn:6.2f}  corrected {dc:6.2f}")
        print(f"    -> {out}")


if __name__ == "__main__":
    main()
