"""Diagnostic plot of a saved 2-D (CNR, sigma^2) calibration.

Shows the var_bias and rel_variance surfaces as heatmaps over the (CNR, sigma^2)
grid, with the Marc-data trajectory overlaid when applicable.

Usage::

    .venv/Scripts/python.exe scripts/plot_calibration.py output/calibration_marc_gaussian.npz
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weighting_wall import CalibrationGrid  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"


def _marc_trajectories():
    """Return (label, per_frame_CNR, per_frame_sigma2) for several CNR_0
    values, computed from Marc's noiseless diffmap. Returns empty list if
    Marc's module is not importable.
    """
    try:
        from mc_experiments.marc_mn_trapping.diffusion_map import (
            Parameters, diffusion, x_axis,
        )
        from weighting_wall.fit import fit_variance as fit_gaussian
    except ImportError:
        return []
    par = Parameters()
    diffmap = diffusion(par)
    x_arr = x_axis(par)
    peak0 = float(diffmap[0].max())
    peaks_per_frame = diffmap.max(axis=1)
    # Noiseless Gaussian-fit variance per frame -- "true variance" the
    # application's calibration lookup is queried at.
    noiseless_var = np.array([
        fit_gaussian(x_arr, row).variance for row in diffmap
    ])
    trajs = []
    for cnr0 in (5, 20, 50, 100, 200):
        noise_sigma = peak0 / cnr0
        cnr_frame = peaks_per_frame / noise_sigma
        trajs.append((f"CNR$_0$={cnr0}", cnr_frame, noiseless_var))
    return trajs


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: plot_calibration.py <path-to-calibration.npz>")
    path = Path(sys.argv[1])
    cal = CalibrationGrid.load(path)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # X axis: CNR (log), Y axis: sigma^2 (linear)
    extent = [np.log10(cal.cnr_grid[0]), np.log10(cal.cnr_grid[-1]),
              cal.variance_grid[0], cal.variance_grid[-1]]

    # ---------- Panel A: var_bias ----------
    ax = axes[0]
    # Clip to [0, max] for visualisation -- negative values are MC noise
    vb_pos = np.clip(cal.var_bias.T, 0.0, None)
    # Symmetric-log-like color scale: linear up to small clip, log above
    pcm = ax.imshow(
        vb_pos, origin="lower", aspect="auto", extent=extent,
        cmap="viridis",
        norm=plt.matplotlib.colors.LogNorm(
            vmin=max(1e-6, np.nanmin(vb_pos[vb_pos > 0])),
            vmax=np.nanmax(vb_pos)),
    )
    fig.colorbar(pcm, ax=ax, label=r"var_bias ($\mu$m$^2$)")
    ax.set_xlabel(r"log$_{10}$(CNR)")
    ax.set_ylabel(r"$\sigma^2$  ($\mu$m$^2$)")
    ax.set_title(f"(a) var_bias  ({cal.model_name}, "
                 f"{cal.n_runs} runs/cell)")

    # Overlay Marc-data trajectories
    for label, cnr_traj, var_traj in _marc_trajectories():
        # Cull anything outside the calibration grid
        mask = ((cnr_traj >= cal.cnr_grid[0])
                & (cnr_traj <= cal.cnr_grid[-1])
                & (var_traj >= cal.variance_grid[0])
                & (var_traj <= cal.variance_grid[-1]))
        if mask.any():
            ax.plot(np.log10(cnr_traj[mask]), var_traj[mask],
                    "-", lw=1.0, color="white", alpha=0.6)
            ax.plot(np.log10(cnr_traj[mask]), var_traj[mask],
                    "o", ms=3, mec="white", mfc="none", label=label)
    ax.legend(fontsize=7, loc="lower right",
              title="Marc per-frame trajectories")

    # ---------- Panel B: rel_variance ----------
    ax = axes[1]
    rv = np.clip(cal.rel_variance.T, 0.0, None)
    pcm = ax.imshow(
        rv, origin="lower", aspect="auto", extent=extent,
        cmap="magma",
        norm=plt.matplotlib.colors.LogNorm(
            vmin=max(1e-9, np.nanmin(rv[rv > 0])),
            vmax=np.nanmax(rv)),
    )
    fig.colorbar(pcm, ax=ax, label=r"Var($\hat\sigma^2$)  ($\mu$m$^4$)")
    ax.set_xlabel(r"log$_{10}$(CNR)")
    ax.set_ylabel(r"$\sigma^2$  ($\mu$m$^2$)")
    ax.set_title(f"(b) rel_variance  ({cal.model_name}, "
                 f"{cal.n_runs} runs/cell)")

    for label, cnr_traj, var_traj in _marc_trajectories():
        mask = ((cnr_traj >= cal.cnr_grid[0])
                & (cnr_traj <= cal.cnr_grid[-1])
                & (var_traj >= cal.variance_grid[0])
                & (var_traj <= cal.variance_grid[-1]))
        if mask.any():
            ax.plot(np.log10(cnr_traj[mask]), var_traj[mask],
                    "-", lw=1.0, color="white", alpha=0.6)
            ax.plot(np.log10(cnr_traj[mask]), var_traj[mask],
                    "o", ms=3, mec="white", mfc="none")

    fig.suptitle(
        f"Calibration diagnostic: {path.name}  "
        f"(n_pixels={cal.pixel_x.size if cal.pixel_x is not None else '?'}, "
        f"amplitude={cal.amplitude})")

    out_path = OUTPUT / f"{path.stem}_diagnostic.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
