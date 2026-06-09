"""Does the folded-normal correction recover alpha under non-Fickian transport?

The truncation correction is derived and calibrated on simple analytic profiles
whose MSD is linear in time (Fickian). This experiment feeds it the harder case:
emission profiles from the Saha reaction-diffusion model
(`reaction_diffusion_saha.py`), where excitons and free charges diffuse at
different rates and the local mix is set by density through the Saha equation.
The effective diffusivity then drifts as the population decays and spreads, so
the MSD is curved -- non-Fickian, with a tunable anomalous exponent alpha in
both directions -- and the observable is non-Gaussian (the n^2 free-charge term
sharpens the peak).

The test reuses the shipped pipeline. Marc's noise-free profiles replace the
analytic Gaussian series; the per-frame Gaussian width fit, the 2-D (CNR,
sigma^2) calibration, the folded-normal subtraction with its fixed-point
inversion, and the slope are all `weighting_wall` code.

Two issues are kept deliberately separate, because they have different causes
(the same split seen on the Mn maps):

1. **Estimator shape bias.** A Gaussian fit of a non-Gaussian (peak-sharpened)
   profile already distorts the width, and therefore alpha, before any noise.
   Quantified by comparing the noise-free Gaussian-fit alpha against the
   true second-moment alpha. This needs no calibration and runs cheaply.
2. **Truncation correction.** Whether folded-normal recovers the noise-free
   Gaussian-fit trajectory (and its alpha) under noise. The honest reference is
   the noise-free *fit* width, not the second moment.

Design choices, per coordination with the weighting-wall side:

* The calibration is the 2-D (CNR, sigma^2) joint lookup, geometry-matched to
  the Saha pixel grid, built once by `scripts/build_calibration.py` and loaded
  here. The 1-D D-free path is invalid here: the fitted width spreads by 1.6-3x
  over the window, far outside its small-spread premise.
* Only `folded_normal` (subtractive) with the fixed-point inversion is scored
  as the correction; `mas` is excluded (its gate currently zeroes the
  correction). `ols` and `inverse_variance` remain as the uncorrected baselines.
* Frames whose per-frame CNR falls below the inversion floor (~0.6) are flagged,
  not silently fit.
* The calibration is built from Gaussian profiles, so it is independent of the
  x-vs-x^2 emission choice; both `free_charge_emission_power` values are run
  through the one calibration and the spread reported.

Build the calibration first (do not run it against another large build)::

    .venv/Scripts/python.exe scripts/build_calibration.py \\
        --tag saha --model gaussian \\
        --n-pixels 401 --half-window 4.0 \\
        --sigma-min 0.15 --sigma-max 1.2 --n-sigma 100 \\
        --cnr-min 0.3 --cnr-max 150 --n-cnr 100 --n-runs 1000

Then run this experiment from the repository root::

    .venv/Scripts/python.exe mc_experiments/saha_reaction_diffusion/nonfickian_validation.py

Parameter magnitudes are anchored to the Cutrupi et al. (2026) scales (nu, rho
from Table S2; De/Df to literature 2D-perovskite exciton diffusivities). This is
a method stress-test on a physically grounded transport model, not a claim about
a specific material.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np

# Run from a source checkout without installing either package.
_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

import reaction_diffusion_saha as rd  # noqa: E402
from weighting_wall.calibration import CalibrationGrid  # noqa: E402
from weighting_wall.fit import fit_D  # noqa: E402
from weighting_wall.models import GaussianModel, VoigtModel  # noqa: E402
from weighting_wall.truncation import TruncationModel  # noqa: E402
from weighting_wall.weights import corrected_variance  # noqa: E402

# Physical pixel grid the Saha profiles are resampled onto, in um. Matched by
# the calibration build above (n_pixels / half_window identical). The absolute
# scale is free; pixels-per-sigma, not dx, is what the bias depends on. Here
# sigma0 ~ 0.3 um gives ~15 pixels/sigma, and var_max/sigma^2_max = 44 (no pinch).
HALF_WINDOW_UM = 4.0
N_PIXELS = 401
PIXEL_X = np.linspace(-HALF_WINDOW_UM, HALF_WINDOW_UM, N_PIXELS)
N_FRAMES = 12

CALIBRATION_PATH = os.path.join("output", "calibration_saha.npz")

# Correction-reliability floor. The diagnostic (diagnose_overcorrection.py)
# shows the 2-D var_bias matches the actual per-frame fit bias for per-frame
# CNR >~ 1.3, but over-estimates it (folded over-subtracts, flattening the late
# MSD and pulling alpha below the reference) in the band CNR ~ 0.6-1.3. So the
# correction is trusted only above ~1.3; frames below are flagged and dropped
# from the alpha fit. (Sharpens the "per-profile CNR floor ~1".)
CNR_FLOOR = 1.3

EVAL_CNR0 = 60.0          # initial CNR for the correction figures
N_RUNS = 400

# Estimator and instrument PSF, set from the CLI in main(). The Saha emission is
# near-Gaussian (beta ~ 1.93) and the simulation carries no instrument response,
# so the Voigt estimator only bites once a Lorentzian PSF is convolved in:
# PSF_GAMMA_UM (HWHM, um) broadens the emission and is also the Voigt fixed
# gamma. Default: Gaussian estimator, no PSF.
ESTIMATOR = GaussianModel()
PSF_GAMMA_UM = 0.0

# Base Saha parameters (paper-anchored). De/Df contrast drives the non-Fickian
# effect; A is swept to move alpha. Stability needs Df*dt/dx^2 < 0.5.
#
# rho is set at the low edge of the paper's Table S2 range (0.75e-3 to 3e-3
# /ns). At the higher fitted rho the bimolecular term rho*n^2 burns the dense
# core almost instantly, collapsing the emission peak ~25-1000x over the window
# so per-frame CNR cannot be sustained; at rho=1e-3 the decay is
# exciton-dominated (gentle exp(-nu t)), every frame stays above the inversion
# floor, and the non-Fickian signal is actually stronger (diffusion has time to
# act before the signal dies).
BASE = dict(L=24.0, nbins=960, De=0.005, Df=0.1, nu=0.25, rho=1e-3,
            tmax=3.0, dt=0.001, sigma0_init=0.35)
A_SWEEP = np.geomspace(30.0, 2000.0, 11)


@dataclass
class Regime:
    """One operating point: noise-free profiles and reference trajectories."""

    A: float
    emission_power: int
    times: np.ndarray          # frame times (ns)
    clean: np.ndarray          # (n_frames, n_pixels) noise-free emission (um grid)
    peak: np.ndarray           # per-frame noise-free peak (for CNR bookkeeping)
    fit_var: np.ndarray        # noise-free estimator-fit variance per frame (um^2)
    moment_var: np.ndarray     # noise-free second-moment variance per frame (um^2)
    alpha_fit: float           # alpha from the estimator-fit trajectory
    alpha_moment: float        # alpha from the second-moment trajectory
    width_ratio: float         # late / initial fitted width


def _alpha(times: np.ndarray, var: np.ndarray) -> float:
    """Anomalous exponent of MSD(t) = var(t) - var(0): slope of log-log fit."""
    msd = var - var[0]
    keep = (times > 0) & (msd > 0)
    if keep.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(times[keep]), np.log(msd[keep]), 1)[0])


def _second_moment_var(profile: np.ndarray) -> float:
    """Second central moment of a (non-negative) emission profile on PIXEL_X."""
    w = np.clip(profile, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.nan
    mean = float((PIXEL_X * w).sum() / s)
    return float(((PIXEL_X - mean) ** 2 * w).sum() / s)


def _psf_broaden(profile: np.ndarray, gamma: float) -> np.ndarray:
    """Convolve a profile with a Lorentzian instrument PSF of HWHM gamma (um).

    Real transport-microscopy profiles are a diffusing population convolved with
    the instrument response; the Saha simulation omits it, so adding it here is
    what makes the observable genuinely Voigt-shaped and the Voigt estimator the
    correct one. The kernel is normalised to conserve integrated intensity.
    """
    dx = float(PIXEL_X[1] - PIXEL_X[0])
    half = max(1, int(np.ceil(8.0 * gamma / dx)))
    offsets = np.arange(-half, half + 1) * dx
    kernel = (gamma / np.pi) / (offsets ** 2 + gamma ** 2)
    kernel /= kernel.sum()
    return np.convolve(profile, kernel, mode="same")


def build_regime(A: float, emission_power: int) -> Regime:
    """Run the Saha model at density coupling A and resample onto PIXEL_X (um).

    No per-regime width normalization: the profiles stay in physical um so the
    single geometry-matched calibration applies to every regime.
    """
    par = rd.Parameters(L=BASE["L"], nbins=BASE["nbins"], De=BASE["De"],
                        Df=BASE["Df"], nu=BASE["nu"], rho=BASE["rho"], A=A,
                        tsamp=BASE["tmax"] / 40.0, dt=BASE["dt"],
                        tmax=BASE["tmax"],
                        free_charge_emission_power=emission_power)
    xs = rd.x_axis(par)
    res = rd.simulate(par, np.exp(-xs ** 2 / (2.0 * BASE["sigma0_init"] ** 2)))

    idx = np.linspace(0, len(res.times) - 1, N_FRAMES).astype(int)
    times = res.times[idx]
    clean = np.array([np.interp(PIXEL_X, xs, res.emission[i], left=0.0, right=0.0)
                      for i in idx])
    if PSF_GAMMA_UM > 0.0:
        clean = np.array([_psf_broaden(row, PSF_GAMMA_UM) for row in clean])

    fit_var = np.array([ESTIMATOR.fit_variance(PIXEL_X, row).variance
                        for row in clean])
    moment_var = np.array([_second_moment_var(row) for row in clean])
    return Regime(
        A=A, emission_power=emission_power, times=times, clean=clean,
        peak=clean.max(axis=1), fit_var=fit_var, moment_var=moment_var,
        alpha_fit=_alpha(times, fit_var), alpha_moment=_alpha(times, moment_var),
        width_ratio=float(np.sqrt(fit_var.max() / fit_var[0])),
    )


def corrected_trajectory(regime: Regime, cnr0: float, model: TruncationModel,
                         n_runs: int = N_RUNS, seed: int = 0):
    """Mean folded-normal-corrected variance trajectory under noise at cnr0.

    Returns (mean_raw, mean_corrected, cnr_frame, below_floor). Mirrors
    `run_ensemble`: noise at the profile level, a per-frame Gaussian fit, then
    the scheme's per-frame correction averaged over realisations.
    """
    noise_sigma = regime.peak[0] / cnr0
    cnr_frame = regime.peak / noise_sigma
    rng = np.random.default_rng(seed)

    raw = np.full((n_runs, N_FRAMES), np.nan)
    cor = np.full((n_runs, N_FRAMES), np.nan)
    for r in range(n_runs):
        noisy = regime.clean + rng.normal(0.0, noise_sigma, size=regime.clean.shape)
        var_row = np.array([ESTIMATOR.fit_variance(PIXEL_X, noisy[f]).variance
                            for f in range(N_FRAMES)])
        raw[r] = var_row
        cor[r] = corrected_variance("folded_normal", var_row, cnr_frame, model)

    return (np.nanmean(raw, axis=0), np.nanmean(cor, axis=0),
            cnr_frame, cnr_frame < CNR_FLOOR)


def load_model(path: str = CALIBRATION_PATH) -> TruncationModel | None:
    """Load the 2-D Saha calibration, verifying its geometry matches PIXEL_X."""
    if not os.path.exists(path):
        return None
    calib = CalibrationGrid.load(path)
    if calib.pixel_x is None or calib.pixel_x.size != PIXEL_X.size or \
            not np.allclose(calib.pixel_x, PIXEL_X):
        raise ValueError(
            f"calibration geometry in {path} does not match PIXEL_X; "
            "rebuild with --n-pixels 401 --half-window 4.0")
    return TruncationModel.from_calibration(calib)


def shape_bias_report() -> dict[int, list[Regime]]:
    """The estimator shape-bias half: needs no calibration, runs cheaply.

    For each emission power, builds the A-sweep and reports the noise-free
    Gaussian-fit alpha against the true second-moment alpha -- the distortion a
    Gaussian fit of a non-Gaussian profile introduces before any noise.
    """
    out = {}
    for power in (1, 2):
        print(f"\nEstimator shape bias, free_charge_emission_power={power} "
              f"({'Marc C code' if power == 1 else 'paper eq. 9'})")
        print("      A    alpha_fit  alpha_moment  width_ratio")
        regimes = [build_regime(float(a), power) for a in A_SWEEP]
        for rg in regimes:
            print(f"  {rg.A:7.0f}    {rg.alpha_fit:7.3f}     "
                  f"{rg.alpha_moment:7.3f}      {rg.width_ratio:.2f}")
        out[power] = regimes
    return out


def correction_report(regimes_by_power: dict[int, list[Regime]],
                      model: TruncationModel, cnr0_list: list[float]) -> None:
    """The truncation-correction half: recover alpha from corrected trajectory."""
    for power, regimes in regimes_by_power.items():
        for ci, cnr0 in enumerate(cnr0_list):
            print(f"\nAlpha recovery at CNR_0={cnr0}, "
                  f"free_charge_emission_power={power}")
            print("      A   alpha_ref  alpha_raw(uncorr)  alpha_folded  "
                  "frames_below_floor")
            recovered = []
            for k, rg in enumerate(regimes):
                mean_raw, mean_cor, cnr_frame, below = corrected_trajectory(
                    rg, cnr0, model, seed=100 + k + 1000 * ci)
                keep = ~below
                ok = keep.sum() >= 2
                a_raw = _alpha(rg.times[keep], mean_raw[keep]) if ok else np.nan
                a_cor = _alpha(rg.times[keep], mean_cor[keep]) if ok else np.nan
                recovered.append((rg, mean_raw, mean_cor, cnr_frame, below, a_cor))
                print(f"  {rg.A:7.0f}    {rg.alpha_fit:6.3f}       {a_raw:6.3f}"
                      f"          {a_cor:6.3f}          {int(below.sum())}")
            # Figure for the lowest (most-stressed) CNR_0 only.
            if ci == 0:
                _plot_correction(power, recovered, cnr0)


def _plot_shape_bias(regimes_by_power) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed -- skipping figures]")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for power, c in ((1, "tab:green"), (2, "tab:purple")):
        regimes = regimes_by_power[power]
        amom = np.array([rg.alpha_moment for rg in regimes])
        afit = np.array([rg.alpha_fit for rg in regimes])
        order = np.argsort(amom)
        ax.plot(amom[order], afit[order], "o-", color=c,
                label=f"emission power {power}")
    lims = ax.get_xlim()
    ax.plot(lims, lims, ls=":", c="grey", lw=1, label="no shape bias")
    ax.set_xlabel(r"true second-moment $\alpha$")
    ax.set_ylabel(r"noise-free Gaussian-fit $\alpha$")
    ax.set_title("Estimator shape bias (before any noise)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", "nonfickian_shape_bias.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


def _plot_correction(power, recovered, cnr0) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Core figure: corrected variance trajectory vs the noise-free-fit
    # reference, for the lowest, middle, and highest-alpha regimes.
    alphas = np.array([r[0].alpha_fit for r in recovered])
    order = np.argsort(alphas)
    picks = [recovered[order[0]], recovered[order[len(order) // 2]],
             recovered[order[-1]]]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), sharey=False)
    for ax, (rg, mean_raw, mean_cor, cnr_frame, below, a_rec) in zip(axes, picks):
        t = rg.times
        ax.plot(t, rg.fit_var, "k-", lw=2, label="noise-free fit (reference)")
        ax.plot(t, mean_raw, "s--", color="tab:orange",
                label="uncorrected (raw)")
        ax.plot(t, mean_cor, "o-", color="tab:green",
                label="folded-normal corrected")
        if below.any():
            ax.plot(t[below], mean_cor[below], "x", color="tab:red",
                    ms=9, label=f"below CNR floor ({int(below.sum())})")
        ax.set_xlabel("time (ns)")
        ax.set_title(rf"$\alpha_{{ref}}$={rg.alpha_fit:.2f}, "
                     rf"$\alpha_{{rec}}$={a_rec:.2f}")
    axes[0].set_ylabel(r"variance (um$^2$)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(f"Corrected variance trajectory (emission power {power}, "
                 f"CNR$_0$={cnr0:.0f})")
    fig.tight_layout()
    out = os.path.join("output", f"nonfickian_trajectory_power{power}.png")
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out}")


def main() -> None:
    global ESTIMATOR, PSF_GAMMA_UM
    parser = argparse.ArgumentParser(
        description="Non-Fickian validation of the folded-normal correction.")
    parser.add_argument("--calibration", default=CALIBRATION_PATH,
                        help="path to the 2-D calibration .npz")
    parser.add_argument("--cnr0", default=str(EVAL_CNR0),
                        help="comma-separated initial CNR values")
    parser.add_argument("--estimator", choices=("gaussian", "voigt"),
                        default="gaussian", help="per-frame width estimator")
    parser.add_argument("--psf-gamma", type=float, default=0.0,
                        help="Lorentzian instrument-PSF HWHM in um; convolved "
                             "into the emission and used as the Voigt fixed "
                             "gamma. Must match the calibration's gamma.")
    args = parser.parse_args()

    PSF_GAMMA_UM = args.psf_gamma
    ESTIMATOR = (VoigtModel(gamma=args.psf_gamma) if args.estimator == "voigt"
                 else GaussianModel())
    cnr0_list = [float(c) for c in args.cnr0.split(",")]

    print("Non-Fickian validation of the folded-normal correction")
    print(f"  Saha model: De={BASE['De']}, Df={BASE['Df']} um^2/ns "
          f"(contrast {BASE['Df'] / BASE['De']:.0f}x), "
          f"nu={BASE['nu']}, rho={BASE['rho']} /ns")
    print(f"  grid: +/-{HALF_WINDOW_UM} um, {N_PIXELS} pixels, {N_FRAMES} frames")
    psf_note = f", PSF gamma={args.psf_gamma} um" if args.psf_gamma > 0 else ""
    print(f"  estimator: {args.estimator}{psf_note}")

    regimes_by_power = shape_bias_report()
    _plot_shape_bias(regimes_by_power)

    model = load_model(args.calibration)
    if model is None:
        print(f"\n[no calibration at {args.calibration}] -- shape-bias half "
              "done; build the calibration (see module docstring) to run the "
              "correction half.")
        return
    print(f"\nUsing calibration: {args.calibration}")
    correction_report(regimes_by_power, model, cnr0_list)


if __name__ == "__main__":
    main()
