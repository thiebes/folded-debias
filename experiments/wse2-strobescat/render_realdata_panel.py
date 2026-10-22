"""Real-data panel: de-parity merged, full-range,
in physical um^2 units.

MoSSe alloy (Enrique's `mixed_tmdc/26_32/250716`, sample "250716_MoSSe_mystery").
De-parity merged reduction (each interleaved sweep re-zeroed to its own t0),
converted to um^2 with the 0.060 um pixel (``UM_PER_PX``;
not the ratio of his sigma(um) to the px fit, which depends on the fit). The message: on real data a measurable ~5% upward
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
    UM_PER_PX, load_measurement, per_frame_cnr, build_or_load_calibration, decide_noise_law)
from folded_debias.fit import fit_variance, fit_D  # noqa: E402

# Spanish-flag palette (plot_style.md): flag red = the salient/biased series,
# flag blue = the reference/corrected series.
RED = "#c60b1e"        # naive / his sigma^2 (open)
BLUE = "#0077c8"       # corrected (filled)
GRAY = "0.6"
INCH = 1 / 2.54
T_FIT = 2.0            # early diffusive window (ns) shown and fit
OUT = os.path.join(os.path.dirname(__file__),
                   "realdata_panel_MoSSe-26_32.png")


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
    scale = UM_PER_PX ** 2             # um^2 per px^2
    # fit + display over the early diffusive window (t <= 2 ns), CNR floor.
    fit_mask = (cnr >= FLOOR) & (t <= T_FIT)
    w = (1.0 / np.maximum(se_naive, 1e-9) ** 2) * fit_mask
    d_n = fit_D(t, msd_n, w); d_c = fit_D(t, msd_c, w)
    # time-binning (inverse-variance weighted) to cut the per-frame scatter and
    # the even/odd parity wobble; the systematic bias survives averaging, the
    # noise does not -- so the correction shows on every binned point.
    tb, nb, cb, sdn, sdc, sem = _time_bins(t[fit_mask], msd_n[fit_mask],
                                           msd_c[fit_mask], se_naive[fit_mask],
                                           n_bins=6)
    return dict(t=t, cnr=cnr, fit_mask=fit_mask, scale=scale,
                D_n=d_n.D * scale, D_c=d_c.D * scale, nl=nl,
                tb=tb, nb=nb * scale, cb=cb * scale,
                sdn=sdn * scale, sdc=sdc * scale, sem=sem * scale)


def _time_bins(t, mn, mc, se, n_bins=7):
    """Inverse-variance weighted means in ~equal-count time bins.
    Returns bin mean time, binned naive/corrected MSD, the within-bin spread
    (std of the frames -- the scatter carried as the error bar), and the SE of
    the mean (the uncertainty on the mean)."""
    order = np.argsort(t)
    t, mn, mc, se = t[order], mn[order], mc[order], se[order]
    edges = np.linspace(0, t.size, n_bins + 1).astype(int)
    tb, nb, cb, sd_n, sd_c, se_m = [], [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        wi = 1.0 / np.maximum(se[a:b], 1e-9) ** 2
        tb.append(float(np.average(t[a:b], weights=wi)))
        nb.append(float(np.average(mn[a:b], weights=wi)))
        cb.append(float(np.average(mc[a:b], weights=wi)))
        sd_n.append(float(np.std(mn[a:b])))
        sd_c.append(float(np.std(mc[a:b])))
        se_m.append(float(1.0 / np.sqrt(wi.sum())))
    return (np.array(tb), np.array(nb), np.array(cb), np.array(sd_n),
            np.array(sd_c), np.array(se_m))


def plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tb, nb, cb, sdn, sdc = r["tb"], r["nb"], r["cb"], r["sdn"], r["sdc"]
    fs = 9
    fig, ax = plt.subplots(layout="constrained", figsize=(16 * INCH, 11 * INCH))
    ax.tick_params(axis="both", which="both", direction="in", top=True,
                   right=True, labelsize=fs, length=4, width=1)
    fig.patch.set_alpha(0.0)            # transparent margin (poster overlay)
    ax.set_facecolor("white")           # opaque white plot area
    ax.patch.set_alpha(1.0)
    # time-binned means at their true bin time (no x-dodge); error bars carry the
    # within-bin scatter (no raw cloud). naive = red open ring, corrected = blue
    # filled dot -- both at the same time, only sigma^2 differs. D values are in
    # the caption, not the legend.
    ax.errorbar(tb, nb, yerr=sdn, fmt="o", mfc="none", mec=RED, ecolor=RED,
                ms=8, capsize=3, lw=1.1, mew=1.4, label=r"naive $\sigma^2$")
    ax.errorbar(tb, cb, yerr=sdc, fmt="o", color=BLUE, ecolor=BLUE, ms=5,
                capsize=3, lw=1.1, label="corrected")
    ax.axhline(0, color="0.8", lw=0.8, zorder=0)
    lo = float(np.nanmin(cb - sdc)); hiv = float(np.nanmax(nb + sdn))
    pad = 0.15 * (hiv - lo + 1e-9)
    ax.set_ylim(lo - pad, hiv + pad)
    ax.set_xlim(-0.3, T_FIT + 0.2)
    ax.set_xlabel("time after pump (ns)", fontsize=fs)
    ax.set_ylabel(r"MSD  $\sigma^2(t)-\sigma^2(0)$  ($\mu$m$^2$)", fontsize=fs)
    ax.legend(frameon=False, fontsize=fs, loc="upper left")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="none")   # margin transparent, axes white
    plt.close(fig)


def main():
    r = build()
    plot(r)
    print(f"material: MoSSe alloy (250716_MoSSe_mystery)")
    print(f"scale = {r['scale']:.5f} um^2/px^2 ; noise law {r['nl']}")
    fm = r["fit_mask"]
    print(f"fit over CNR>={FLOOR} & t<={T_FIT}ns: naive D = {r['D_n']:.4f}  "
          f"corrected D = {r['D_c']:.4f} um^2/ns  ratio {r['D_c']/r['D_n']:.3f}")
    print(f"{int(fm.sum())}/{r['t'].size} frames; "
          f"t {r['t'][fm].min():.2f}..{r['t'][fm].max():.2f} ns; "
          f"CNR {r['cnr'][fm].min():.2f}..{r['cnr'][fm].max():.1f}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
