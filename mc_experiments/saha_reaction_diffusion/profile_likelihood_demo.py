"""Profile-likelihood error bars for the De/Df diffusion fit.

The De/Df fit in Fit_Simulation_Diff_free_vs_exc_updated.py takes its
uncertainties from the L-BFGS-B hess_inv, which the optimizer builds for its own
search, not as a covariance. This gets the error by profile likelihood instead:
scan one diffusion constant, re-fit the other at each step, and read the 1-sigma
interval off where chi-squared rises by 1 above its minimum. It also writes out
the delta-chi-squared profile for each constant.

It rebuilds the forward model (the same reaction-diffusion PDE and the same
chi-squared against the experimental MSD), vectorised for speed. The one quantity
the saved fit files do not carry is the absolute initial density (the fit scales
the initial profile by Nexc/A and does not store A), so the initial amplitude is
recovered here by matching the run to the saved best-fit msd_sim.

The interval is conditional on the lifetime parameters (nu, rho, qe, qf) and the
amplitude held at their fitted values; the full uncertainty also has to fold in
how the lifetime fit propagates into De/Df.

Set DATA_DIR and CASE to the fit you want, then run.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import minimize, minimize_scalar

DATA_DIR = r"P:/Marc/shared_data/data_links/Diffusion_simulation"
CASE = "n_inf_N_exc_2.47e+13"

# Diffusion-run geometry, matching run_simulation in the fit code.
L, NBINS, SIGMA0 = 12.0, 101, 0.1
TMAX, TSAMP, DT = 10.0, 0.05, 2e-3      # tsamp/dt are coarser than the fit's; the MSD is dt-converged
DE_BOUNDS, DF_BOUNDS = (0.001, 0.12), (0.05, 0.7)


def msd_curve(De, Df, amp0, nu, rho, qe, qf):
    """Reaction-diffusion PDE -> second-moment MSD vs time.

    Saha fraction pointwise, x^2 free-charge recombination and emission, A = 1
    (the density scale is carried by amp0), non-negativity clip on the population.
    """
    y = np.linspace(-0.5 * L, 0.5 * L, NBINS)
    dx2 = (L / NBINS) ** 2
    n = amp0 * np.exp(-(y * y) / (2 * SIGMA0 ** 2))
    n[0] = n[-1] = 0.0
    nsteps, nsamp = int(TMAX / DT), int(TSAMP / DT)
    ce, cf = De * DT / dx2, Df * DT / dx2

    emission = []
    for step in range(nsteps + 1):
        ntilde = np.maximum(n, 0.0)
        x = np.where(ntilde > 0.0, 2.0 / (np.sqrt(4.0 * ntilde + 1.0) + 1.0), 1.0)
        x[0] = x[-1] = 1.0
        if step % nsamp == 0:
            emission.append(n * (qe * (1.0 - x) * nu + qf * x * x * rho * n))
        excitons, free = (1.0 - x) * n, x * n
        n[1:-1] = np.maximum(0.0, n[1:-1]
                             + ce * (excitons[2:] + excitons[:-2] - 2.0 * excitons[1:-1])
                             + cf * (free[2:] + free[:-2] - 2.0 * free[1:-1])
                             - DT * n[1:-1] * (nu * (1.0 - x[1:-1])
                                               + rho * n[1:-1] * x[1:-1] ** 2))

    emission = np.array(emission)
    y_centre = y + 0.5 * (L / NBINS)
    var = np.zeros(emission.shape[0])
    for t in range(emission.shape[0]):
        signal = emission[t].sum()
        if signal > 0:
            mean = (y_centre * emission[t]).sum() / signal
            var[t] = ((y_centre - mean) ** 2 * emission[t]).sum() / signal
    msd = np.maximum(var - var[0], 0.0)
    return np.arange(len(msd)) * TSAMP, msd


def calibrate_amplitude(de_fit, df_fit, params, time_ref, msd_ref):
    """Recover the initial-amplitude scale by matching the run to the saved msd_sim.

    The fit files store De/Df/nu/rho/qe/qf but not A, so the absolute density is
    unknown; this finds the amplitude whose noise-free MSD reproduces msd_sim.
    """
    def residual(amp0):
        time_sim, msd_sim = msd_curve(de_fit, df_fit, amp0, **params)
        return np.sum((np.interp(time_ref, time_sim, msd_sim) - msd_ref) ** 2)

    amp0 = min(np.logspace(-1, 5, 25), key=residual)
    for _ in range(3):
        amp0 = min(np.linspace(amp0 / 3, amp0 * 3, 15), key=residual)
    return amp0, residual(amp0)


def one_sigma_interval(grid, dchi2):
    """Where the profile crosses delta-chi-squared = 1, interpolated between points."""
    k = int(np.argmin(dchi2))
    lo, hi = grid[0], grid[-1]
    for i in range(k, 0, -1):
        if dchi2[i - 1] >= 1.0:
            f = (1.0 - dchi2[i]) / (dchi2[i - 1] - dchi2[i])
            lo = grid[i] + f * (grid[i - 1] - grid[i]); break
    for i in range(k, len(grid) - 1):
        if dchi2[i + 1] >= 1.0:
            f = (1.0 - dchi2[i]) / (dchi2[i + 1] - dchi2[i])
            hi = grid[i] + f * (grid[i + 1] - grid[i]); break
    return lo, grid[k], hi


def plot_profiles(de_grid, dchi2_de, de_interval, df_grid, dchi2_df, df_interval, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y_top = 6.0
    fig, (ax_de, ax_df) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, grid, dchi2, (lo, centre, hi), colour, label in (
        (ax_de, de_grid, dchi2_de, de_interval, "b", "De"),
        (ax_df, df_grid, dchi2_df, df_interval, "g", "Df"),
    ):
        ax.plot(grid, dchi2, colour, lw=1.8)
        ax.axhline(1.0, color="0.5", ls="--", lw=1)
        ax.axvspan(lo, hi, color=colour, alpha=0.15)

        # Zoom to where the curve lives, cap the wall above the crossing.
        shown = grid[dchi2 < y_top]
        margin = 0.25 * (shown.max() - shown.min())
        ax.set_xlim(shown.min() - margin, shown.max() + margin)
        ax.set_ylim(0, y_top)

        ax.text(ax.get_xlim()[1], 1.0, r" $\Delta\chi^2=1$ (1$\sigma$)",
                va="bottom", ha="right", fontsize=8, color="0.4")
        ax.text(0.04, 0.96, f"{label} = {centre:.4f}\n($-${centre-lo:.4f} / $+${hi-centre:.4f})",
                transform=ax.transAxes, va="top", fontsize=10)
        ax.set_xlabel(f"{label} (um$^2$/ns)")
        ax.set_ylabel(r"profile $\Delta\chi^2$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")


def main():
    params_file = json.load(open(os.path.join(DATA_DIR, CASE + ".json")))["fit_parameters"]
    saved = np.load(os.path.join(DATA_DIR, CASE + ".npy"), allow_pickle=True).item()
    time_exp = np.asarray(saved["time_exp_list"][0])
    msd_exp = np.asarray(saved["msd_exp_list"][0])
    msd_err = np.maximum(np.asarray(saved["msd_err_list"][0]), 1e-12)
    time_sim = np.asarray(saved["time_sim_list"][0])
    msd_sim = np.asarray(saved["msd_sim_list"][0])
    valid = np.ones(len(msd_sim), bool); valid[1:] = msd_sim[1:] > 1e-12   # drop trailing 0.0 sentinel
    time_sim, msd_sim = time_sim[valid], msd_sim[valid]

    de_fit, df_fit = params_file["De"], params_file["Df"]
    lifetime = {k: params_file[k] for k in ("nu", "rho", "qe", "qf")}

    print("Calibrating initial amplitude to the saved msd_sim ...")
    amp0, residual = calibrate_amplitude(de_fit, df_fit, lifetime, time_sim, msd_sim)
    print(f"  amp0 = {amp0:.4g}  (match residual {residual:.2e})")

    def chi2(De, Df):
        time_sim, msd_sim = msd_curve(De, Df, amp0, **lifetime)
        return float(np.sum(((np.interp(time_exp, time_sim, msd_sim) - msd_exp) / msd_err) ** 2))

    def chi2_profiled(De=None, Df=None):
        """chi-squared minimised over whichever constant is left free."""
        if Df is None:
            return minimize_scalar(lambda d: chi2(De, d), bounds=DF_BOUNDS, method="bounded").fun
        return minimize_scalar(lambda d: chi2(d, Df), bounds=DE_BOUNDS, method="bounded").fun

    print("Locating the joint minimum ...")
    opt = minimize(lambda p: chi2(p[0], p[1]), x0=[de_fit, df_fit],
                   method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6})
    de_hat, df_hat = opt.x
    chi2_min = opt.fun
    ndof = int(np.sum(np.asarray(saved["msd_err_list"][0]) > 0)) - 2

    # Dense, tight grids around the minimum: smooth curves and a well-resolved
    # Delta-chi^2 = 1 crossing. The other constant is re-fit at each step.
    print("Profiling De ...")
    de_grid = np.linspace(max(DE_BOUNDS[0], de_hat - 0.015), de_hat + 0.015, 61)
    dchi2_de = np.array([chi2_profiled(De=d) for d in de_grid]) - chi2_min
    print("Profiling Df ...")
    df_grid = np.linspace(max(DF_BOUNDS[0], df_hat - 0.03), df_hat + 0.03, 61)
    dchi2_df = np.array([chi2_profiled(Df=d) for d in df_grid]) - chi2_min

    de_interval = one_sigma_interval(de_grid, dchi2_de)
    df_interval = one_sigma_interval(df_grid, dchi2_df)
    de_lo, de_c, de_hi = de_interval
    df_lo, df_c, df_hi = df_interval

    print("\n" + "=" * 66)
    print(f"  {CASE}   (reduced chi^2 at min = {chi2_min / ndof:.2f})")
    print(f"  De = {de_c:.4f}  1-sigma [{de_lo:.4f}, {de_hi:.4f}]"
          f"  (-{de_c-de_lo:.4f} / +{de_hi-de_c:.4f})   hess_inv error: {params_file.get('De_err', np.nan):.4f}")
    print(f"  Df = {df_c:.4f}  1-sigma [{df_lo:.4f}, {df_hi:.4f}]"
          f"  (-{df_c-df_lo:.4f} / +{df_hi-df_c:.4f})   hess_inv error: {params_file.get('Df_err', np.nan):.4f}")
    print("=" * 66)

    out_path = os.path.join("output", "profile_likelihood_De_Df.png")
    os.makedirs("output", exist_ok=True)
    plot_profiles(de_grid, dchi2_de, de_interval, df_grid, dchi2_df, df_interval, out_path)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
