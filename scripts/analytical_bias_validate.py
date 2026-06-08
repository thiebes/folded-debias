"""Validate the closed-form width-estimator bias against the Monte Carlo.

For each estimator the script computes the analytical bias
`b = E[sigma_hat^2] - sigma_true^2` and the MC bias over a parameter grid, then
reports the post-correction residual. The stop criterion for the derivation is
that at per-frame CNR = 1.3 the corrected mean width is within 1% of truth.

Run from the repository root::

    .venv/Scripts/python.exe scripts/analytical_bias_validate.py [section]

with `section` one of: gaussian (default), voigt, gengauss, moment, all.

The MEAN is the target statistic; the median is printed only as
a tail-contamination diagnostic. Set WEIGHTING_WALL_WORKERS to parallelise the
MC oracle.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Stream output even when redirected to a file (Python block-buffers otherwise),
# so a long parallel run can be monitored as it goes.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighting_wall import MODELS, Experiment  # noqa: E402
from weighting_wall.montecarlo import run_ensemble  # noqa: E402
from weighting_wall.simulate import noise_sigma_for_cnr0  # noqa: E402


# --------------------------------------------------------------------------
# Analytical predictors
# --------------------------------------------------------------------------
def box_bias_gaussian(x: np.ndarray, sigma: float, amp: float,
                      noise_sigma: float) -> float:
    """Box (1971) second-order NLS bias of the fitted variance, free (v, A, c).

    Builds the discrete Jacobian and the relevant Hessian terms on the actual
    pixel grid and evaluates Box's bias formula exactly (no continuum
    approximation). Parameters are (v, A, c); the centre c decouples from v by
    parity but its inverse-information weight still enters the d-vector trace.

    Returns the additive variance bias E[sigma_hat^2] - sigma^2 to leading
    order in noise_sigma^2 (i.e. the 1/CNR^2 term). The fold and var_max
    saturation are separate low-CNR effects not included here.
    """
    v = sigma ** 2
    f = amp * np.exp(-0.5 * x ** 2 / v)            # noise-free profile, c = 0
    # Jacobian columns
    Fv = f * x ** 2 / (2.0 * v ** 2)
    FA = f / amp
    Fc = f * x / v
    # Hessian terms that survive the block-diagonal (v,A | c) inverse weighting
    Hvv = f * ((x ** 2 / (2.0 * v ** 2)) ** 2 - x ** 2 / v ** 3)
    HvA = (f / amp) * x ** 2 / (2.0 * v ** 2)
    Hcc = f * ((x / v) ** 2 - 1.0 / v)
    # H_AA = 0 (model linear in A)

    # (v, A) information block and its inverse; c is orthogonal (odd) so its
    # information is the scalar sum(Fc^2).
    Svv = float(np.sum(Fv * Fv))
    SvA = float(np.sum(Fv * FA))
    SAA = float(np.sum(FA * FA))
    Scc = float(np.sum(Fc * Fc))
    det = Svv * SAA - SvA ** 2
    Minv_vv = SAA / det
    Minv_vA = -SvA / det
    Minv_cc = 1.0 / Scc

    # d_i = tr[(F^T F)^-1 H_i] over the (v, A, c) parameters.
    d = Minv_vv * Hvv + 2.0 * Minv_vA * HvA + Minv_cc * Hcc

    # b(v_hat) = -(sigma_n^2/2) * sum_b (F^TF)^-1_{v,b} sum_i F_{i,b} d_i
    proj_v = float(np.sum(Fv * d))
    proj_A = float(np.sum(FA * d))
    return -(noise_sigma ** 2 / 2.0) * (Minv_vv * proj_v + Minv_vA * proj_A)


def box_bias_gaussian_continuum(dx: float, sigma: float, cnr: float) -> float:
    """Continuum closed form b = (2/sqrt(pi)) (dx sigma) / CNR^2.

    The hand-integrated limit of `box_bias_gaussian`; agrees with the discrete
    version once the profile is interior (W/sigma >~ 4).
    """
    return (2.0 / np.sqrt(np.pi)) * dx * sigma / cnr ** 2


def moment_bulk_bias(x: np.ndarray, sigma: float, amp: float,
                     noise_sigma: float) -> float:
    """Delta-method 'bulk' bias of the second-moment estimator, O(1/CNR^2).

    V = N2/T - (N1/T)^2 with N2 = sum w x^2, N1 = sum w x, T = sum w, w the
    background-subtracted profile. A second-order expansion of the ratio about
    the noise-free point gives

        E[V] - sigma^2 ~= (sigma_n^2 / S^2) [ sigma^2 N - 2 sum x^2 ],

    S = sum of the noise-free profile. Valid only where T stays well away from
    zero (high CNR); near the total-through-zero boundary the sample mean is
    dominated by small-T events and this bulk value no longer describes it.
    """
    f = amp * np.exp(-0.5 * x ** 2 / sigma ** 2)
    S = float(np.sum(f))
    N = x.size
    X2 = float(np.sum(x ** 2))
    return (noise_sigma ** 2 / S ** 2) * (sigma ** 2 * N - 2.0 * X2)


def moment_total_through_zero_fraction(x: np.ndarray, sigma: float, amp: float,
                                       noise_sigma: float,
                                       edge_frac: float = 0.1) -> float:
    """P(total background-subtracted signal < 0) -- the instability boundary.

    T = sum_i (eps_i - B_hat) has mean S and variance ~ N sigma_n^2 +
    N^2 Var(B_hat), B_hat the edge-average baseline. When T crosses zero the
    moment ratio is corrupted; this returns the Gaussian-tail probability of
    that event, the fraction of frames the estimator flags NaN / blows up on.
    """
    from scipy.stats import norm
    f = amp * np.exp(-0.5 * x ** 2 / sigma ** 2)
    S = float(np.sum(f))
    N = x.size
    edge = max(1, int(edge_frac * N))
    var_baseline = noise_sigma ** 2 / (2.0 * edge)   # Var(0.5(meanL+meanR))
    var_T = N * noise_sigma ** 2 + N ** 2 * var_baseline
    return float(norm.cdf(-S / np.sqrt(var_T)))


# --------------------------------------------------------------------------
# MC oracle
# --------------------------------------------------------------------------
def mc_bias(model_name: str, cnr: float, sigma: float = 1.0,
            half_window: float = 6.0, n_pixels: int = 241,
            n_runs: int = 20000, seed: int = 0, **model_kw):
    """Empirical bias of the fitted variance for an isolated frame at `cnr`.

    Uses a D = 0 experiment so every frame shares the same true width; frame 0
    has per-frame CNR exactly equal to `cnr` (full amplitude, width sigma).
    Returns (cnr_frame0, mean_bias, median_bias, se_mean).
    """
    model = MODELS[model_name](**model_kw) if model_kw else MODELS[model_name]()
    exp = Experiment(sigma0=sigma, D=0.0, half_window=half_window,
                     n_pixels=n_pixels, n_frames=1)
    ens = run_ensemble(exp, float(cnr), n_runs=n_runs, seed=seed, model=model)
    col = ens.variance[:, 0]
    col = col[np.isfinite(col)]
    v = sigma ** 2
    return (float(ens.cnr_frame[0]), float(col.mean()) - v,
            float(np.median(col)) - v, float(col.std(ddof=1) / np.sqrt(col.size)))


def gaussian_section(n_runs: int = 20000) -> None:
    print("=" * 78)
    print("Gaussian fit: Box high-CNR bias vs MC")
    print("Base geometry dx=0.05 (N=241, W=6), sigma=1; b in sigma0^2 units")
    print("=" * 78)
    sigma, W, N = 1.0, 6.0, 241
    x = np.linspace(-W, W, N)
    print(f"\n{'CNR':>6} {'b_MC_mean':>11} {'se':>9} {'b_MC_med':>11} "
          f"{'b_Box':>11} {'mean/Box':>9} {'med/Box':>9}")
    for cnr in (50.0, 30.0, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.3):
        _, bm, bd, se = mc_bias("gaussian", cnr, sigma, W, N, n_runs=n_runs)
        ns = noise_sigma_for_cnr0(
            Experiment(sigma0=sigma, D=0.0, half_window=W, n_pixels=N), cnr)
        bbox = box_bias_gaussian(x, sigma, 1.0, ns)
        print(f"{cnr:>6.1f} {bm:>11.3e} {se:>9.2e} {bd:>11.3e} {bbox:>11.3e} "
              f"{bm/bbox:>9.2f} {bd/bbox:>9.2f}")


def gaussian_geometry_sweep(n_runs: int = 20000) -> None:
    """Check the dx, W, sigma scaling of the Box coefficient at CNR = 8."""
    print("\n" + "=" * 78)
    print("Gaussian Box bias: geometry scaling at CNR = 8 (b vs b_Box)")
    print("=" * 78)
    cnr = 8.0
    print("\n-- pixel size dx (vary N at W = 6, sigma = 1) --")
    print(f"{'N':>6} {'dx':>8} {'b_MC':>11} {'se':>9} {'b_Box':>11} {'MC/Box':>8}")
    for N in (121, 161, 241, 321, 481):
        x = np.linspace(-6.0, 6.0, N)
        _, bm, _, se = mc_bias("gaussian", cnr, 1.0, 6.0, N, n_runs=n_runs)
        ns = noise_sigma_for_cnr0(
            Experiment(sigma0=1.0, D=0.0, half_window=6.0, n_pixels=N), cnr)
        bbox = box_bias_gaussian(x, 1.0, 1.0, ns)
        print(f"{N:>6} {x[1]-x[0]:>8.4f} {bm:>11.3e} {se:>9.2e} {bbox:>11.3e} "
              f"{bm/bbox:>8.2f}")
    print("\n-- window W (vary half_window at dx ~ 0.05, sigma = 1) --")
    print(f"{'W':>6} {'N':>6} {'b_MC':>11} {'se':>9} {'b_Box':>11} {'MC/Box':>8}")
    for W in (3.0, 4.0, 6.0, 8.0, 10.0):
        N = int(round(2 * W / 0.05)) + 1
        x = np.linspace(-W, W, N)
        _, bm, _, se = mc_bias("gaussian", cnr, 1.0, W, N, n_runs=n_runs)
        ns = noise_sigma_for_cnr0(
            Experiment(sigma0=1.0, D=0.0, half_window=W, n_pixels=N), cnr)
        bbox = box_bias_gaussian(x, 1.0, 1.0, ns)
        print(f"{W:>6.0f} {N:>6} {bm:>11.3e} {se:>9.2e} {bbox:>11.3e} "
              f"{bm/bbox:>8.2f}")
    print("\n-- width sigma (W = 6, N = 241, dx = 0.05) --")
    print(f"{'sigma':>6} {'b_MC':>11} {'se':>9} {'b_Box':>11} {'MC/Box':>8}")
    for sg in (0.5, 0.7, 1.0, 1.4, 2.0):
        x = np.linspace(-6.0, 6.0, 241)
        _, bm, _, se = mc_bias("gaussian", cnr, sg, 6.0, 241, n_runs=n_runs)
        ns = noise_sigma_for_cnr0(
            Experiment(sigma0=sg, D=0.0, half_window=6.0, n_pixels=241), cnr)
        bbox = box_bias_gaussian(x, sg, 1.0, ns)
        print(f"{sg:>6.2f} {bm:>11.3e} {se:>9.2e} {bbox:>11.3e} {bm/bbox:>8.2f}")


def moment_section(n_runs: int = 60000) -> None:
    print("=" * 78)
    print("Moment estimator: bulk delta-method bias vs MC, and T<0 boundary")
    print("Base geometry dx=0.05 (N=241, W=6), sigma=1; b in sigma0^2 units")
    print("=" * 78)
    sigma, W, N = 1.0, 6.0, 241
    x = np.linspace(-W, W, N)
    amp = 1.0
    print(f"\n{'CNR':>6} {'b_MC_mean':>11} {'se':>10} {'b_MC_med':>11} "
          f"{'b_bulk':>11} {'P(T<0)pred':>11} {'NaN_frac':>9}")
    for cnr in (50.0, 30.0, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.3):
        exp = Experiment(sigma0=sigma, D=0.0, half_window=W, n_pixels=N,
                         n_frames=1)
        ens = run_ensemble(exp, float(cnr), n_runs=n_runs, seed=0,
                           model=MODELS["moment"]())
        col = ens.variance[:, 0]
        finite = np.isfinite(col)
        nan_frac = 1.0 - finite.mean()
        cf = col[finite]
        bm = float(cf.mean()) - sigma ** 2
        bd = float(np.median(cf)) - sigma ** 2
        se = float(cf.std(ddof=1) / np.sqrt(cf.size))
        ns = noise_sigma_for_cnr0(exp, cnr)
        bulk = moment_bulk_bias(x, sigma, amp, ns)
        pz = moment_total_through_zero_fraction(x, sigma, amp, ns)
        print(f"{cnr:>6.1f} {bm:>11.3e} {se:>10.2e} {bd:>11.3e} {bulk:>11.3e} "
              f"{pz:>11.3e} {nan_frac:>9.3e}")


def main() -> None:
    section = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    if section in ("gaussian", "all"):
        gaussian_section()
    if section in ("gaussian_geom", "all"):
        gaussian_geometry_sweep()
    if section in ("moment", "all"):
        moment_section()


if __name__ == "__main__":
    main()
