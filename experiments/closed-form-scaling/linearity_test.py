"""Test the scaling form of the fitted-variance bias.

The bias of the fitted variance is, to leading order, the width's own
estimation variance, Var(sigma_hat), which for an efficient fit is the
Cramer-Rao (Fisher) value. For a 1-D Gaussian profile with white noise, peak
CNR and pixel pitch dx, far from the window edges,

  var_bias = (2 / sqrt(pi)) * sigma * dx / CNR^2,
  g = var_bias / sigma^2 = (2 / sqrt(pi)) / s^2 = 1.13 / s^2,   s = CNR * sqrt(sigma / dx).

So bias / sigma^2 is a function of s alone, not of CNR alone: s captures both
the noise level (CNR) and the spatial resolution (pixels per sigma). Near the
window edge the profile is truncated and the exact Fisher value, computed on
the actual pixel grid (``width_debias.fisher.leading_order_terms``), exceeds
the closed form; the script reports both. An earlier version compared with the
truncated-normal form lambda(-s)/s, which falls exponentially in s where the
measured bias falls as 1/s^2; the scaling variable was right, the form was not.

This script runs a D=0 calibration sweep at several sigma_0 values with FIXED
absolute geometry (n_pixels, half_window in physical units), with independent
seeds per sigma_0, and plots bias/sigma_0^2 against:

  Panel A: CNR alone                 -- should NOT collapse if sigma matters
  Panel B: s = CNR sqrt(sigma_0/dx)  -- SHOULD collapse onto 1.13/s^2
  Panel C: ratio empirical / exact Fisher -- where the leading term stops holding

Each cell carries its Monte Carlo SE (SD / sqrt(n)), so agreement is judged in
SE units (z = (g_mean - g_Fisher) / SE).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from width_debias import GaussianModel, leading_order_terms  # noqa: E402
from width_debias.montecarlo import run_ensemble  # noqa: E402
from width_debias.simulate import Experiment  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[2] / "output"
OUTPUT.mkdir(exist_ok=True)

# Fixed absolute geometry: matches Marc-style application, where dx is set
# by the experimental pixel grid and sigma_true varies frame to frame.
ABS_HALF_WINDOW = 1.5    # um, +/- half-window
N_PIXELS = 251           # odd so a pixel sits at x=0
DX = 2 * ABS_HALF_WINDOW / (N_PIXELS - 1)

# sigma_0 spans an order of magnitude. At sigma_0 = 0.5 um the window ends at
# 3 sigma; at sigma_0 = 0.05 um it has 30 sigma of headroom.
SIGMA0_GRID = np.array([0.05, 0.10, 0.20, 0.50])  # um

CNR_GRID = np.geomspace(0.5, 200.0, 18)
N_RUNS = 500
CLOSED_FORM = 2.0 / np.sqrt(np.pi)   # g = CLOSED_FORM / s^2
Z_S_MIN = (5.0, 10.0)                # report chi^2/dof against Fisher for s above these


def fisher_g(cnr: np.ndarray, sigma0: float, x: np.ndarray) -> np.ndarray:
    """Exact leading-order g = var_bias / sigma_0^2 on the actual pixel grid."""
    a = leading_order_terms(GaussianModel(), x, sigma0 ** 2).bias_coef
    return a / cnr ** 2 / sigma0 ** 2


def calibrate_with_both_stats(cnr0_grid, sigma0, half_window, n_pixels,
                              n_runs, n_frames=12, seed=1):
    """Run a D=0 calibration sweep, returning per-frame mean and median bias.

    Replicates calibrate_d_free's MC structure but records the mean (the
    standard calibration) with its SE, and the median (a robust summary that
    suppresses the broken-fit heavy tail) of sigma_hat^2 - sigma^2_true at each
    (CNR, frame) cell.

    Returns (each (M,), ascending in CNR): cnr, mean, mean SE, median, and the
    experiment's pixel grid.
    """
    exp = Experiment(sigma0=sigma0, D=0.0, half_window=half_window,
                     n_pixels=n_pixels, n_frames=n_frames)
    true_var = exp.true_variance  # all = sigma_0^2 at D=0

    rows = []
    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i)
        for f in range(n_frames):
            vcol = ens.variance[:, f]
            vcol = vcol[np.isfinite(vcol)]
            if vcol.size < 2:
                continue
            rows.append((float(ens.cnr_frame[f]), float(np.mean(vcol)) - true_var[f],
                         float(np.std(vcol, ddof=1) / np.sqrt(vcol.size)),
                         float(np.median(vcol)) - true_var[f]))
    rows = np.array(sorted(rows))
    return rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3], exp.x


def main():
    print("Fitted-variance bias scaling test")
    print("=" * 70)
    print(f"Fixed geometry: n_pixels = {N_PIXELS}, "
          f"half_window = {ABS_HALF_WINDOW} um, dx = {DX:.4f} um")
    print(f"sigma_0 sweep: {SIGMA0_GRID} um")
    print(f"CNR grid: {len(CNR_GRID)} points from "
          f"{CNR_GRID[0]:.2f} to {CNR_GRID[-1]:.2f}")
    print(f"MC runs per CNR/sigma_0 cell: {N_RUNS}")

    results = {}
    for k, sigma0 in enumerate(SIGMA0_GRID):
        # half_window in sigma_0 units, chosen so absolute half-window is fixed
        half_window_in_sigma0 = ABS_HALF_WINDOW / sigma0
        print(f"\nsigma_0 = {sigma0} um  (half_window = "
              f"{half_window_in_sigma0:.1f} sigma_0, "
              f"sigma_0/dx = {sigma0/DX:.1f} pixels/sigma)", flush=True)
        cnr, vb_mean, vb_se, vb_med, x = calibrate_with_both_stats(
            CNR_GRID, sigma0=sigma0, half_window=half_window_in_sigma0,
            n_pixels=N_PIXELS, n_runs=N_RUNS, seed=1 + 1000 * k)
        results[sigma0] = (cnr, vb_mean, vb_se, vb_med, fisher_g(cnr, sigma0, x))

    # ----- Text summary -----
    lines = [
        "Fitted-variance bias scaling test",
        f"Geometry: n_pixels = {N_PIXELS}, half_window = +/- {ABS_HALF_WINDOW} um, "
        f"dx = {DX:.4f} um",
        f"MC: {N_RUNS} runs per (CNR, sigma_0) cell; seeds 1 + 1000 k + i for sigma_0 index k",
        "",
        "For each (sigma_0, CNR) cell:",
        "  g_mean   = mean bias / sigma_0^2 (SE: its Monte Carlo standard error)",
        "  s        = CNR * sqrt(sigma_0 / dx)",
        f"  g_closed = {CLOSED_FORM:.4f} / s^2 (Fisher, 1-D Gaussian far from the window edge)",
        "  g_fisher = exact Fisher value on this pixel grid (includes window truncation)",
        "  z        = (g_mean - g_fisher) / SE",
        "",
    ]
    for sigma0 in SIGMA0_GRID:
        cnr, vb_mean, vb_se, vb_med, g_fisher = results[sigma0]
        g_mean, g_se, g_med = vb_mean / sigma0 ** 2, vb_se / sigma0 ** 2, vb_med / sigma0 ** 2
        s = cnr * np.sqrt(sigma0 / DX)
        g_closed = CLOSED_FORM / s ** 2
        z = (g_mean - g_fisher) / g_se
        lines.append(f"--- sigma_0 = {sigma0} um  (R = sigma_0/dx = {sigma0/DX:.2f}; "
                     f"exact / closed Fisher = {np.median(g_fisher / g_closed):.3f}) ---")
        for s_min in Z_S_MIN:
            sel = s >= s_min
            lines.append(f"  s >= {s_min:g}: {int(sel.sum())} cells, chi^2/dof against exact "
                         f"Fisher {np.mean(z[sel] ** 2):.2f}, beyond 2 SE {int(np.sum(np.abs(z[sel]) > 2))}")
        lines.append("  CNR       s         g_mean    SE        g_median  g_closed  g_fisher  "
                     "mean/fisher  z")
        for c, sv, gm, ge, gd, gc, gf, zz in zip(cnr, s, g_mean, g_se, g_med, g_closed,
                                                 g_fisher, z):
            lines.append(f"  {c:8.3f}  {sv:8.3f}  {gm:8.4g}  {ge:8.2g}  {gd:8.4g}  "
                         f"{gc:8.4g}  {gf:8.4g}  {gm / gf:10.3g}  {zz:+6.2f}")
        lines.append("")

    text = "\n".join(lines)
    text_path = OUTPUT / "linearity_test.txt"
    text_path.write_text(text + "\n", encoding="utf-8")
    print(f"Summary saved: {text_path}")

    # ----- Plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(no matplotlib; skipping plot)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(SIGMA0_GRID)))
    s_plot = np.geomspace(0.3, 500, 300)

    for row, stat_name, stat_idx in [(0, "mean", 1), (1, "median", 3)]:
        # Panel A: g_empirical = bias / sigma_0^2 vs CNR
        ax = axes[row, 0]
        for color, sigma0 in zip(colors, SIGMA0_GRID):
            data = results[sigma0]
            g = data[stat_idx] / sigma0 ** 2
            ax.loglog(data[0], np.clip(np.abs(g), 1e-6, None), color=color,
                      marker='o', markersize=4, lw=1,
                      label=f"$\\sigma_0$ = {sigma0} um")
        ax.set_xlabel("per-frame CNR")
        ax.set_ylabel(f"$g_{{{stat_name}}} = $ var_bias / $\\sigma_0^2$")
        ax.legend(fontsize=8, loc="upper right")

        # Panel B: g_empirical vs s
        ax = axes[row, 1]
        for color, sigma0 in zip(colors, SIGMA0_GRID):
            data = results[sigma0]
            g = data[stat_idx] / sigma0 ** 2
            s = data[0] * np.sqrt(sigma0 / DX)
            ax.loglog(s, np.clip(np.abs(g), 1e-6, None), color=color,
                      marker='o', markersize=4, lw=1,
                      label=f"$\\sigma_0$ = {sigma0} um")
        ax.loglog(s_plot, CLOSED_FORM / s_plot ** 2, "k--", lw=2,
                  label=r"Fisher $1.13/s^2$")
        ax.set_xlabel(r"$s = $ CNR $\sqrt{\sigma_0/dx}$")
        ax.set_ylabel(f"$g_{{{stat_name}}} = $ var_bias / $\\sigma_0^2$")
        ax.legend(fontsize=8, loc="upper right")

        # Panel C: empirical / exact Fisher
        ax = axes[row, 2]
        for color, sigma0 in zip(colors, SIGMA0_GRID):
            data = results[sigma0]
            g = data[stat_idx] / sigma0 ** 2
            s = data[0] * np.sqrt(sigma0 / DX)
            ax.semilogx(s, g / data[4], color=color, marker='o', markersize=4, lw=1,
                        label=f"$\\sigma_0$ = {sigma0} um")
        ax.axhline(1.0, color="k", ls="--", alpha=0.6, label="Fisher")
        ax.set_xlabel(r"$s$")
        ax.set_ylabel(f"$g_{{{stat_name}}}$ / $g_{{Fisher}}$")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_yscale("symlog", linthresh=1.0)

    fig_path = OUTPUT / "linearity_test.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()
