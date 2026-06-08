"""Validate the folded-normal correction on Marc Melendez's diffusion maps.

Marc supplied noiseless diffusion maps and ground-truth variance curves for four
Mn-doping cases (undoped, 0.51%, 0.81%, 4.04% Mn) from the Magdaleno et al. 2025
reaction-diffusion model. This script runs the per-frame correction directly on
his published maps -- no regeneration through our port -- which is the strongest
realistic-transport test of the method: the transport is genuinely non-Fickian
(two trapping channels), and the ground truth is the model author's own
sigma^2(t).

For each case and each CNR_0 we add white noise to the noiseless map, fit each
frame's Gaussian variance, apply the subtractive folded-normal correction through
the joint (CNR, sigma^2) calibration (``calibration_marc_gaussian_filtered.npz``,
which carries the fixed-point inversion needed for the large variance spread of
the undoped case), and average over seeds. We then compare the recovered variance
to the ground truth, both over all frames and over the working regime
(per-frame CNR >= 1, where the per-profile floor places the correction).

Run from the repository root::

    .venv/Scripts/python.exe scripts/marc_diffmap_validation.py

Outputs ``output/marc_diffmap_validation.{png,txt}``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from weighting_wall import CalibrationGrid  # noqa: E402
from weighting_wall.fit import fit_variance as fit_gaussian  # noqa: E402
from weighting_wall.truncation import TruncationModel  # noqa: E402
from weighting_wall.weights import corrected_variance  # noqa: E402

from mc_experiments.marc_mn_trapping.marc_data import load_all  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_marc_gaussian_filtered.npz"

DICE_BLUE = "#003f7f"
DICE_GOLD = "#f7941e"

CNR_GRID = (10.0, 20.0, 50.0, 100.0, 200.0)
FIG_CNR0 = 100.0          # CNR_0 shown in the per-case trajectory figure
N_SEEDS = 15
FLOOR_CNR = 1.0           # per-frame CNR above which the correction is reliable


def fit_each_row(x: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    """Gaussian variance fit of every row; NaN where the fit fails."""
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = fit_gaussian(x, profiles[i])
        if getattr(r, "ok", True) and np.isfinite(r.variance):
            out[i] = r.variance
    return out


def validation_pass(case, trunc: TruncationModel, cnr0: float, n_seeds: int):
    """Per-frame raw and folded-normal-corrected variance, averaged over seeds."""
    peak0 = float(case.diffmap[0].max())
    noise_sigma = peak0 / cnr0
    cnr_frame = case.diffmap.max(axis=1) / noise_sigma

    n_frames = case.diffmap.shape[0]
    raw = np.full((n_seeds, n_frames), np.nan)
    cor = np.full((n_seeds, n_frames), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(int(cnr0) * 1000 + s)
        noisy = case.diffmap + rng.normal(0.0, noise_sigma, case.diffmap.shape)
        var_hat = fit_each_row(case.x, noisy)
        raw[s] = var_hat
        cor[s] = corrected_variance("folded_normal", var_hat, cnr_frame, trunc)
    return (np.nanmean(raw, axis=0), np.nanmean(cor, axis=0), cnr_frame)


def _residual(recovered: np.ndarray, truth: np.ndarray,
              mask: np.ndarray | None = None) -> float:
    d = np.abs(recovered - truth)
    if mask is not None:
        d = d[mask]
    return float(np.nanmean(d))


def main() -> None:
    if not CAL_PATH.exists():
        print(f"Calibration not found: {CAL_PATH}")
        print("Build it with scripts/build_calibration.py (tag "
              "marc_gaussian_filtered) first.")
        return

    cal = CalibrationGrid.load(CAL_PATH)
    trunc = TruncationModel.from_calibration(cal)
    cases = load_all()

    lines = []
    lines.append("=" * 70)
    lines.append("Marc diffusion-map validation -- folded-normal correction")
    lines.append("(joint 2-D calibration; subtractive folded_normal + "
                 "fixed-point)")
    lines.append("=" * 70)
    lines.append(f"calibration: {CAL_PATH.name}")
    lines.append(f"n_seeds per CNR_0: {N_SEEDS};  reliable regime: "
                 f"per-frame CNR >= {FLOOR_CNR}")
    lines.append("")

    lines.append("Reference = the NOISELESS Gaussian fit (isolates the "
                 "truncation correction).")
    lines.append("Shape gap = |noiseless fit - true 2nd moment|: the "
                 "Gaussian-fit-vs-moment mismatch the correction cannot touch.")
    lines.append("")

    fig_rows = {}     # case.name -> (cnr_frame, raw, cor, nf) at FIG_CNR0
    for case in cases:
        # Noise-free Gaussian fit per frame: the reference the truncation
        # correction must recover. With no noise there is no fold, so this is
        # the bias-free target for the *estimator we actually use* -- distinct
        # from the true second moment, which a Gaussian fit of a non-Gaussian
        # profile does not equal (the "shape gap" below).
        nf = fit_each_row(case.x, case.diffmap)
        lines.append("-" * 70)
        lines.append(f"=== {case.name} ===   noiseless fit: "
                     f"[{np.nanmin(nf):.4g}, {np.nanmax(nf):.4g}] um^2   "
                     f"(true 2nd-moment max {case.var_true.max():.4g})")
        lines.append("  CNR_0 | min CNR | frames CNR>=1 | mean|resid vs "
                     "noiseless fit| (raw / corrected) | shape gap")
        for cnr0 in CNR_GRID:
            raw, cor, cnr_frame = validation_pass(case, trunc, cnr0, N_SEEDS)
            work = cnr_frame >= FLOOR_CNR
            n_work = int(work.sum())
            r_raw = _residual(raw, nf, work)
            r_cor = _residual(cor, nf, work)
            shape_gap = _residual(nf, case.var_true, work)
            lines.append(f"  {cnr0:5.0f} | {cnr_frame.min():7.2f} | "
                         f"{n_work:4d}/{cnr_frame.size} | "
                         f"{r_raw:8.4g}  /  {r_cor:8.4g} | {shape_gap:8.4g}")
            if cnr0 == FIG_CNR0:
                fig_rows[case.name] = (cnr_frame, raw, cor, nf)
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT / "marc_diffmap_validation.txt").write_text(report)

    # --- Figure: recovered vs true variance per case at FIG_CNR0 ----------
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
    for ax, case in zip(axes.ravel(), cases):
        cnr_frame, raw, cor, nf = fig_rows[case.name]
        work = cnr_frame >= FLOOR_CNR
        ax.plot(case.t, nf, color="k", lw=1.8, label="noiseless fit (ref)")
        ax.plot(case.t, case.var_true, color="#888888", lw=1.0, ls=":",
                label="true 2nd moment")
        ax.plot(case.t, raw, ".", color=DICE_GOLD, ms=3, alpha=0.7,
                label="raw fit")
        ax.plot(case.t, cor, ".", color=DICE_BLUE, ms=3, label="corrected")
        # Mark where per-frame CNR drops below the working floor.
        if work.any() and not work.all():
            t_floor = case.t[work][-1]
            ax.axvline(t_floor, color="#bbbbbb", ls="--", lw=1.0)
        ax.set_title(f"{case.name}", fontsize=10)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.set_ylim(0, max(case.var_true.max() * 1.5, 0.05))
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axes[1, :]:
        ax.set_xlabel("time  (ns)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"variance  $\sigma^2(t)$  (um$^2$)")
    fig.suptitle(f"Correction on Marc's diffusion maps  (CNR$_0$ = "
                 f"{FIG_CNR0:.0f})", fontsize=12)
    fig.tight_layout()
    path = OUTPUT / "marc_diffmap_validation.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
