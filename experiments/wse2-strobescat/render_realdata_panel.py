"""Canonical poster real-data panel (station 8): de-parity merged, full-range,
in physical um^2 units.

MoSSe alloy (Enrique's `mixed_tmdc/26_32/250716`, sample "250716_MoSSe_mystery").
De-parity merged reduction (each interleaved sweep re-zeroed to its own t0),
converted to um^2 via the sample's pixel->um scale (his sigma(um)^2 / my px fit,
a well-defined constant here). The message: on real data a measurable ~5% upward
bias in D, concentrated in the low-CNR late frames that are normally discarded.

Numbers-only (no fit lines): naive sigma^2 (red, open) vs corrected (blue,
filled), error bars, D and the ratio in the legend. Every frame is shown; the
single below-floor frame (CNR < 1.3, which over-subtracts -- the correction's
documented limitation) is flagged and excluded from the fit, so the reported D is
the floor-valid, ~5% result (including it would give a spurious 41% from the
over-subtraction, not the clean bias).

Usage::

    .venv/Scripts/python.exe experiments/wse2-strobescat/render_realdata_panel.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from realdata_diagnostics import (MEASUREMENTS, FIT_HALF_SIGMA, FLOOR,  # noqa: E402
    load_measurement, per_frame_cnr, build_or_load_calibration, decide_noise_law)
from weighting_wall.fit import fit_variance, fit_D  # noqa: E402

RED = "#c60b1e"        # naive / his sigma^2 (open)
BLUE = "#0077c8"       # corrected (filled)
GRAY = "0.6"
INCH = 1 / 2.54
OUT = os.path.join(os.path.dirname(__file__), "figure_candidates",
                   "station8_realdata-MoSSe-26_32.png")


def build():
    label, folder = MEASUREMENTS["mixed_tmdc_26_32_250716"]
    m = load_measurement(folder)
    t, prof, npix, center, batch = (m["t"], m["profiles"], m["npix"],
                                    m["center"], m["batch"])
    sig_um, err_um = m["sigma_um"], m["errsigma_um"]
    rough = float(np.sqrt(fit_variance(np.arange(npix, dtype=float),
                                       prof[0]).variance))
    half = max(6, int(round(FIT_HALF_SIGMA * rough)))
    lo, hi = max(0, center - half), min(npix, center + half)
    win = prof[:, lo:hi]
    cnr, _, shot, amp, gain, read = per_frame_cnr(win)
    nl = decide_noise_law(gain, read, amp)
    xs = np.arange(hi - lo, dtype=float)
    var_naive = np.array([fit_variance(xs, p).variance for p in win])
    sig0_px2 = float(var_naive[0])
    se_naive = var_naive * 2.0 * (err_um / sig_um)
    tr = build_or_load_calibration(np.sqrt(sig0_px2), hi - lo, nl)
    var_corr = var_naive - tr.var_bias(cnr) * sig0_px2
    # de-parity: each sweep re-zeroed to its own t0
    base_n = np.zeros_like(var_naive); base_c = np.zeros_like(var_corr)
    for b in set(batch.tolist()):
        s = batch == b
        base_n[s] = var_naive[s][0]; base_c[s] = var_corr[s][0]
    msd_n = var_naive - base_n; msd_c = var_corr - base_c
    # pixel->um scale (um^2/px^2): his sigma^2 vs my px fit, high-CNR frames
    scale = float(np.median((sig_um ** 2 / var_naive)[cnr > 5]))
    keep = cnr >= FLOOR
    w = (1.0 / np.maximum(se_naive, 1e-9) ** 2) * keep
    d_n = fit_D(t, msd_n, w); d_c = fit_D(t, msd_c, w)
    return dict(t=t, cnr=cnr, keep=keep, scale=scale,
                msd_n=msd_n * scale, msd_c=msd_c * scale,
                se=se_naive * scale, D_n=d_n.D * scale, D_c=d_c.D * scale, nl=nl)


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t, k, se = r["t"], r["keep"], r["se"]
    fs = 9
    fig, ax = plt.subplots(layout="constrained", figsize=(16 * INCH, 11 * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_facecolor("w"); fig.patch.set_alpha(1)
    ratio = r["D_c"] / r["D_n"]
    # naive: all frames (red open); corrected: floor-valid frames (blue filled)
    ax.errorbar(t[k], r["msd_n"][k], yerr=se[k], fmt="o", mfc="none", mec=RED,
                ecolor=RED, ms=4, capsize=2, lw=0.8,
                label=f"naive $\\sigma^2$:  D = {r['D_n']:.4f} $\\mu$m$^2$/ns")
    ax.errorbar(t[k], r["msd_c"][k], yerr=se[k], fmt="o", color=BLUE,
                ecolor=BLUE, ms=3.5, capsize=2, lw=0.8,
                label=f"corrected:  D = {r['D_c']:.4f} $\\mu$m$^2$/ns  "
                      f"(ratio {ratio:.2f})")
    if (~k).any():
        ax.errorbar(t[~k], r["msd_n"][~k], yerr=se[~k], fmt="o", mfc="none",
                    mec=GRAY, ecolor=GRAY, ms=4, capsize=2, lw=0.8,
                    label=f"CNR < {FLOOR} (below floor; excluded from fit)")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    # y-range from the floor-valid points so the rise stays readable
    lo = float(np.nanmin((r["msd_c"][k] - se[k])))
    hiv = float(np.nanmax((r["msd_n"][k] + se[k])))
    pad = 0.12 * (hiv - lo + 1e-9)
    ax.set_ylim(lo - pad, hiv + pad)
    ax.set_xlabel("time after pump (ns)", fontsize=fs)
    ax.set_ylabel(r"MSD  $\sigma^2(t)-\sigma^2(0)$  ($\mu$m$^2$)", fontsize=fs)
    ax.legend(frameon=False, fontsize=fs, loc="upper left")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200); plt.close(fig)


def main():
    r = build()
    plot(r)
    print(f"material: MoSSe alloy (250716_MoSSe_mystery)")
    print(f"scale = {r['scale']:.5f} um^2/px^2 ; noise law {r['nl']}")
    print(f"fit over CNR>={FLOOR}: naive D = {r['D_n']:.4f}  corrected D = "
          f"{r['D_c']:.4f} um^2/ns  ratio {r['D_c']/r['D_n']:.3f}")
    print(f"kept {int(r['keep'].sum())}/{r['t'].size} frames; "
          f"t {r['t'][r['keep']].min():.2f}..{r['t'][r['keep']].max():.2f} ns; "
          f"CNR {r['cnr'][r['keep']].min():.2f}..{r['cnr'][r['keep']].max():.1f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
