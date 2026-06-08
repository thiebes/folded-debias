"""Validate the folded-normal correction on the Solari disordered-NC-solid maps.

This is the poster's headline regime on real model transport: CH3NH3PbBr3
nanocrystal-solid diffusion maps from Solari et al. (Fig. 5), L_D/FWHM ~ 0.14.
The maps track position only, so `mc_experiments.disordered_nc_solid.nc_data`
imposes the population decay `N0 exp(-t/tau)` (paper lifetime tau ~ 7 ns) and
projects each 2-D frame to a 1-D profile.

First pass uses the Gaussian fit and the NC-solid-geometry calibration
(`calibration_nc_solid_gaussian.npz`, built at this pixel grid). The paper fits
a Voigt to each slice, so a Voigt pass is the natural refinement once this
validates the mechanics. As with the Mn maps we score the correction against
the NOISELESS fit (isolating the truncation correction) and report the shape
gap (noiseless fit vs true second moment) separately.

Run from the repository root::

    .venv/Scripts/python.exe scripts/nc_solid_validation.py

Outputs ``output/nc_solid_validation.{png,txt}``.
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

from mc_experiments.disordered_nc_solid.nc_data import load_all  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)
CAL_PATH = OUTPUT / "calibration_nc_solid_gaussian.npz"

DICE_BLUE = "#003f7f"
DICE_GOLD = "#f7941e"

CNR_GRID = (5.0, 10.0, 20.0, 50.0, 100.0)
FIG_CNR0 = 20.0
N_SEEDS = 15
FLOOR_CNR = 1.0


def fit_each_row(x: np.ndarray, profiles: np.ndarray) -> np.ndarray:
    out = np.full(profiles.shape[0], np.nan)
    for i in range(profiles.shape[0]):
        r = fit_gaussian(x, profiles[i])
        if getattr(r, "ok", True) and np.isfinite(r.variance):
            out[i] = r.variance
    return out


def validation_pass(case, trunc: TruncationModel, cnr0: float, n_seeds: int):
    peak0 = float(case.profiles[0].max())
    noise_sigma = peak0 / cnr0
    cnr_frame = case.profiles.max(axis=1) / noise_sigma
    n_frames = case.profiles.shape[0]
    raw = np.full((n_seeds, n_frames), np.nan)
    cor = np.full((n_seeds, n_frames), np.nan)
    for s in range(n_seeds):
        rng = np.random.default_rng(int(cnr0) * 1000 + s)
        noisy = case.profiles + rng.normal(0.0, noise_sigma, case.profiles.shape)
        var_hat = fit_each_row(case.x, noisy)
        raw[s] = var_hat
        cor[s] = corrected_variance("folded_normal", var_hat, cnr_frame, trunc)
    return np.nanmean(raw, axis=0), np.nanmean(cor, axis=0), cnr_frame


def _residual(recovered, truth, mask=None) -> float:
    d = np.abs(recovered - truth)
    if mask is not None:
        d = d[mask]
    return float(np.nanmean(d))


def main() -> None:
    if not CAL_PATH.exists():
        print(f"Calibration not found: {CAL_PATH.name}. Build it with "
              "scripts/build_calibration.py (tag nc_solid_gaussian) first.")
        return
    trunc = TruncationModel.from_calibration(CalibrationGrid.load(CAL_PATH))
    cases = load_all()
    if not cases:
        print("No NC-solid maps found on the pshare; nothing to validate.")
        return

    lines = [
        "=" * 70,
        "Disordered NC-solid validation (Solari et al.) -- folded-normal",
        "Gaussian fit; reference = noiseless fit; decay tau = "
        f"{cases[0].tau_ns:.0f} ns",
        "=" * 70,
        f"n_seeds per CNR_0: {N_SEEDS};  reliable regime: per-frame "
        f"CNR >= {FLOOR_CNR}",
        "",
    ]
    fig_rows = {}
    for case in cases:
        nf = fit_each_row(case.x, case.profiles)              # noiseless fit
        gap = _residual(nf, case.var_true)
        lines.append("-" * 70)
        lines.append(f"=== {case.name} ===   noiseless fit "
                     f"[{np.nanmin(nf):.4g}, {np.nanmax(nf):.4g}] um^2, "
                     f"shape gap {gap:.4g} um^2 "
                     f"(true 2nd-moment [{case.var_true.min():.4g}, "
                     f"{case.var_true.max():.4g}])")
        lines.append("  CNR_0 | min CNR | frames>=1 | mean|resid vs "
                     "noiseless fit| (raw / corrected)")
        for cnr0 in CNR_GRID:
            raw, cor, cnr_frame = validation_pass(case, trunc, cnr0, N_SEEDS)
            work = cnr_frame >= FLOOR_CNR
            r_raw = _residual(raw, nf, work)
            r_cor = _residual(cor, nf, work)
            lines.append(f"  {cnr0:5.0f} | {cnr_frame.min():7.2f} | "
                         f"{int(work.sum()):3d}/{cnr_frame.size} | "
                         f"{r_raw:9.4g}  /  {r_cor:9.4g}")
            if cnr0 == FIG_CNR0:
                fig_rows[case.name] = (cnr_frame, raw, cor, nf)
        lines.append("")

    report = "\n".join(lines)
    print(report)
    (OUTPUT / "nc_solid_validation.txt").write_text(report)

    # --- Figure: recovered vs reference variance per map at FIG_CNR0 ------
    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 3.6),
                             squeeze=False)
    for ax, case in zip(axes[0], cases):
        cnr_frame, raw, cor, nf = fig_rows[case.name]
        work = cnr_frame >= FLOOR_CNR
        ax.plot(case.t, nf, color="k", lw=1.7, label="noiseless fit (ref)")
        ax.plot(case.t, case.var_true, color="#888888", lw=1.0, ls=":",
                label="true 2nd moment")
        ax.plot(case.t, raw, ".", color=DICE_GOLD, ms=4, alpha=0.7,
                label="raw fit")
        ax.plot(case.t, cor, ".", color=DICE_BLUE, ms=4, label="corrected")
        if work.any() and not work.all():
            ax.axvline(case.t[work][-1], color="#bbbbbb", ls="--", lw=1.0)
        ax.set_title(f"{case.name}", fontsize=10)
        ax.set_xlabel("time  (ns)")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(True, alpha=0.2)
    axes[0][0].set_ylabel(r"variance  $\sigma^2(t)$  (um$^2$)")
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Correction on Solari NC-solid maps  (CNR$_0$ = "
                 f"{FIG_CNR0:.0f}, tau = {cases[0].tau_ns:.0f} ns)", fontsize=12)
    fig.tight_layout()
    path = OUTPUT / "nc_solid_validation.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved figure: {path.name}")


if __name__ == "__main__":
    main()
