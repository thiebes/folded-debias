"""Test the scaling form of the folded-normal bias.

The dimensional / CRLB analysis (Thiebes, 2026-05-31) gives:

  Std(sigma_hat^2) ~ sigma^{3/2} * sqrt(dx) / CNR   [CRLB, Fisher info for sigma^2]
  alpha = sigma^2 / Std(sigma_hat^2) = CNR * sqrt(sigma/dx) / 1.226
  bias / sigma^2 = lambda(-alpha) / alpha           [folded-normal mean shift]

So bias / sigma^2 is theoretically a function of  s = CNR * sqrt(sigma/dx)  alone,
NOT CNR alone. The variable s captures both the noise level (CNR) and the
spatial resolution (pixels per sigma).

This script runs calibrate_d_free at multiple sigma_0 values with FIXED
absolute geometry (n_pixels, half_window in physical units) and plots
bias/sigma_0^2 against two x-axes:

  Panel A: vs CNR alone           -- should NOT collapse if sigma matters
  Panel B: vs s = CNR sqrt(sigma_0/dx) -- SHOULD collapse if the analytical
                                          scaling holds
  Panel C: ratio empirical / analytical -- where the model breaks

If Panel B collapses, the analytical scaling is validated and the
multiplicative correction generalizes to a one-dimensional lookup on s
instead of on CNR.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighting_wall.montecarlo import run_ensemble  # noqa: E402
from weighting_wall.simulate import Experiment  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

# Fixed absolute geometry: matches Marc-style application, where dx is set
# by the experimental pixel grid and sigma_true varies frame to frame.
ABS_HALF_WINDOW = 1.5    # um, +/- half-window
N_PIXELS = 251           # odd so a pixel sits at x=0
DX = 2 * ABS_HALF_WINDOW / (N_PIXELS - 1)

# sigma_0 spans two orders of magnitude. At sigma_0 = 1.0, the profile
# extends 1.5 sigma; at sigma_0 = 0.05, it has 30 sigma of headroom.
SIGMA0_GRID = np.array([0.05, 0.10, 0.20, 0.50])  # um

CNR_GRID = np.geomspace(0.5, 200.0, 18)
N_RUNS = 500


def lambda_func(z):
    """Inverse Mills ratio: phi(z) / (1 - Phi(z))."""
    return norm.pdf(z) / (1.0 - norm.cdf(z))


def analytical_bias_over_sigma2(s):
    """bias / sigma^2 from the folded-normal CRLB model, as a function of s."""
    s = np.asarray(s, float)
    # bias / sigma^2 = lambda(-s) / s  (from the derivation)
    return lambda_func(-s) / s


def calibrate_with_both_stats(cnr0_grid, sigma0, half_window, n_pixels,
                               n_runs, n_frames=12, seed=1):
    """Run a D=0 calibration sweep, returning per-frame mean AND median bias.

    Replicates calibrate_d_free's MC structure but records both the mean
    (the standard calibration) and the median (a robust summary that
    suppresses the broken-fit heavy tail) of sigma_hat^2 - sigma^2_true at
    each (CNR, frame) cell.

    Returns:
        cnr  : (M,) array of per-frame CNR values, ascending
        mean : (M,) array of mean(sigma_hat^2 - sigma^2_true) at each cnr
        med  : (M,) array of median(sigma_hat^2 - sigma^2_true) at each cnr
    """
    exp = Experiment(sigma0=sigma0, D=0.0, half_window=half_window,
                     n_pixels=n_pixels, n_frames=n_frames)
    true_var = exp.true_variance  # all = sigma_0^2 at D=0

    cnr_list = []
    mean_list = []
    med_list = []
    for i, cnr0 in enumerate(cnr0_grid):
        ens = run_ensemble(exp, float(cnr0), n_runs=n_runs, seed=seed + i)
        for f in range(n_frames):
            cnr_f = float(ens.cnr_frame[f])
            vcol = ens.variance[:, f]
            vcol = vcol[np.isfinite(vcol)]
            if vcol.size == 0:
                continue
            cnr_list.append(cnr_f)
            mean_list.append(float(np.mean(vcol)) - true_var[f])
            med_list.append(float(np.median(vcol)) - true_var[f])
    cnr_arr = np.array(cnr_list)
    mean_arr = np.array(mean_list)
    med_arr = np.array(med_list)
    order = np.argsort(cnr_arr)
    return cnr_arr[order], mean_arr[order], med_arr[order]


def main():
    print("Folded-normal bias scaling test")
    print("=" * 70)
    print(f"Fixed geometry: n_pixels = {N_PIXELS}, "
          f"half_window = {ABS_HALF_WINDOW} um, dx = {DX:.4f} um")
    print(f"sigma_0 sweep: {SIGMA0_GRID} um")
    print(f"CNR grid: {len(CNR_GRID)} points from "
          f"{CNR_GRID[0]:.2f} to {CNR_GRID[-1]:.2f}")
    print(f"MC runs per CNR/sigma_0 cell: {N_RUNS}")

    results = {}
    for sigma0 in SIGMA0_GRID:
        # half_window in sigma_0 units, chosen so absolute half-window is fixed
        half_window_in_sigma0 = ABS_HALF_WINDOW / sigma0
        print(f"\nsigma_0 = {sigma0} um  (half_window = "
              f"{half_window_in_sigma0:.1f} sigma_0, "
              f"sigma_0/dx = {sigma0/DX:.1f} pixels/sigma)")
        cnr_arr, vb_mean, vb_med = calibrate_with_both_stats(
            CNR_GRID, sigma0=sigma0,
            half_window=half_window_in_sigma0,
            n_pixels=N_PIXELS, n_runs=N_RUNS,
        )
        results[sigma0] = (cnr_arr, vb_mean, vb_med)
        print(f"  mean   varbias range: [{vb_mean.min():.4g}, "
              f"{vb_mean.max():.4g}] um^2")
        print(f"  median varbias range: [{vb_med.min():.4g}, "
              f"{vb_med.max():.4g}] um^2")
        print(f"  g_mean   = bias/sigma_0^2 range: "
              f"[{vb_mean.min()/sigma0**2:.4g}, "
              f"{vb_mean.max()/sigma0**2:.4g}]")
        print(f"  g_median = bias/sigma_0^2 range: "
              f"[{vb_med.min()/sigma0**2:.4g}, "
              f"{vb_med.max()/sigma0**2:.4g}]")

    # ----- Plot -----
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(no matplotlib; skipping plot)")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(SIGMA0_GRID)))
    s_plot = np.geomspace(0.3, 500, 300)
    g_ana_plot = analytical_bias_over_sigma2(s_plot)

    for row, stat_name, stat_idx in [(0, "mean", 1), (1, "median", 2)]:
        # Panel A: g_empirical = bias / sigma_0^2 vs CNR
        ax = axes[row, 0]
        for color, sigma0 in zip(colors, SIGMA0_GRID):
            data = results[sigma0]
            cnr, vb = data[0], data[stat_idx]
            g = vb / sigma0 ** 2
            ax.loglog(cnr, np.clip(np.abs(g), 1e-6, None), color=color,
                      marker='o', markersize=4, lw=1,
                      label=f"$\\sigma_0$ = {sigma0} um")
        ax.set_xlabel("per-frame CNR")
        ax.set_ylabel(f"$g_{{{stat_name}}} = $ var_bias / $\\sigma_0^2$")
        ax.set_title(f"({chr(ord('a') + row*3)}) {stat_name}: g vs CNR\n"
                     "(linear-in-$\\sigma^2$ predicts collapse)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both")

        # Panel B: g_empirical vs s
        ax = axes[row, 1]
        for color, sigma0 in zip(colors, SIGMA0_GRID):
            data = results[sigma0]
            cnr, vb = data[0], data[stat_idx]
            g = vb / sigma0 ** 2
            s = cnr * np.sqrt(sigma0 / DX)
            ax.loglog(s, np.clip(np.abs(g), 1e-6, None), color=color,
                      marker='o', markersize=4, lw=1,
                      label=f"$\\sigma_0$ = {sigma0} um")
        ax.loglog(s_plot, np.clip(g_ana_plot, 1e-6, None), "k--", lw=2,
                  label=r"analytical $\lambda(-s)/s$")
        ax.set_xlabel(r"$s = $ CNR $\sqrt{\sigma_0/dx}$")
        ax.set_ylabel(f"$g_{{{stat_name}}} = $ var_bias / $\\sigma_0^2$")
        ax.set_title(f"({chr(ord('a') + row*3 + 1)}) {stat_name}: g vs $s$\n"
                     "(CRLB universal scaling predicts collapse)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both")

        # Panel C: empirical / analytical
        ax = axes[row, 2]
        for color, sigma0 in zip(colors, SIGMA0_GRID):
            data = results[sigma0]
            cnr, vb = data[0], data[stat_idx]
            g_emp = vb / sigma0 ** 2
            s = cnr * np.sqrt(sigma0 / DX)
            g_ana = analytical_bias_over_sigma2(s)
            ratio = g_emp / np.where(g_ana > 0, g_ana, np.nan)
            ax.semilogx(s, ratio, color=color, marker='o', markersize=4, lw=1,
                        label=f"$\\sigma_0$ = {sigma0} um")
        ax.axhline(1.0, color="k", ls="--", alpha=0.6, label="theory")
        ax.set_xlabel(r"$s$")
        ax.set_ylabel(f"$g_{{{stat_name}}}$ / $g_{{analytical}}$")
        ax.set_title(f"({chr(ord('a') + row*3 + 2)}) {stat_name}: deviation from theory")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3, which="both")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1e4)

    fig.suptitle("Mean (top) vs median (bottom) calibration: "
                 "does median bias collapse onto the CRLB scaling?",
                 fontsize=13)
    fig_path = OUTPUT / "linearity_test.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nFigure saved: {fig_path}")

    # ----- Text summary -----
    lines = []
    lines.append("Folded-normal bias scaling test")
    lines.append(f"Geometry: n_pixels = {N_PIXELS}, "
                 f"half_window = +/- {ABS_HALF_WINDOW} um, dx = {DX:.4f} um")
    lines.append(f"MC: {N_RUNS} runs per (CNR, sigma_0) cell")
    lines.append("")
    lines.append("For each (sigma_0, CNR) cell, three views:")
    lines.append("  g_emp     = bias_cal / sigma_0^2     (dimensionless)")
    lines.append("  s         = CNR * sqrt(sigma_0 / dx) (dimensionless arg)")
    lines.append("  g_ana(s)  = lambda(-s) / s           (theory)")
    lines.append("")
    for sigma0 in SIGMA0_GRID:
        cnr, vb_mean, vb_med = results[sigma0]
        g_mean = vb_mean / sigma0 ** 2
        g_med = vb_med / sigma0 ** 2
        s = cnr * np.sqrt(sigma0 / DX)
        g_ana = analytical_bias_over_sigma2(s)
        lines.append(f"--- sigma_0 = {sigma0} um  "
                     f"(R = sigma_0/dx = {sigma0/DX:.2f}) ---")
        lines.append("  CNR       s         g_mean    g_median  g_ana     "
                     "mean/ana  med/ana")
        for c, sv, gm, gd, ga in zip(cnr, s, g_mean, g_med, g_ana):
            r_mean = gm / ga if ga > 0 else np.nan
            r_med = gd / ga if ga > 0 else np.nan
            lines.append(f"  {c:8.3f}  {sv:8.3f}  {gm:8.4g}  {gd:8.4g}  "
                         f"{ga:8.4g}  {r_mean:8.3g}  {r_med:8.3g}")
        lines.append("")

    text = "\n".join(lines)
    text_path = OUTPUT / "linearity_test.txt"
    text_path.write_text(text)
    print(f"Summary saved: {text_path}")


if __name__ == "__main__":
    main()
