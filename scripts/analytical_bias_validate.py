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


def box_bias_numeric(profile_v, v: float, amp: float, mean: float,
                     noise_sigma: float, x_scale: float = 1.0,
                     h_rel: float = 1e-4) -> float:
    """Box variance-bias for any profile parametrised by (v, amp, mean).

    `profile_v(v, amp, mean)` returns the noise-free profile sampled on a fixed
    pixel grid. The reported variance is `v` itself, so finite-differencing the
    profile w.r.t. theta = (v, amp, mean) and applying Box's multi-parameter
    formula returns the variance bias directly -- valid for any profile shape
    (Gaussian, Voigt, generalized-Gaussian) by MLE reparameterisation
    equivariance. The full 3x3 information inverse is used (no block-diagonal
    assumption), so cross-parameter curvature is captured exactly on the grid.
    """
    theta = np.array([v, amp, mean], float)
    h = np.array([h_rel * v, h_rel * amp, h_rel * x_scale])

    def f(th):
        return profile_v(th[0], th[1], th[2])

    p = theta.size
    n = f(theta).size
    F = np.empty((n, p))
    for k in range(p):
        tp = theta.copy(); tp[k] += h[k]
        tm = theta.copy(); tm[k] -= h[k]
        F[:, k] = (f(tp) - f(tm)) / (2.0 * h[k])
    H = np.empty((n, p, p))
    for j in range(p):
        for k in range(p):
            tpp = theta.copy(); tpp[j] += h[j]; tpp[k] += h[k]
            tpm = theta.copy(); tpm[j] += h[j]; tpm[k] -= h[k]
            tmp = theta.copy(); tmp[j] -= h[j]; tmp[k] += h[k]
            tmm = theta.copy(); tmm[j] -= h[j]; tmm[k] -= h[k]
            H[:, j, k] = (f(tpp) - f(tpm) - f(tmp) + f(tmm)) / (4.0 * h[j] * h[k])

    FtF = F.T @ F
    inv = np.linalg.inv(FtF)
    d = np.einsum("jk,ijk->i", inv, H)          # tr[(F^TF)^-1 H_i] per pixel
    bias_vec = -(noise_sigma ** 2 / 2.0) * (inv @ (F.T @ d))
    return float(bias_vec[0])                    # the v component


def box_bias_gaussian_continuum(dx: float, sigma: float, cnr: float) -> float:
    """Continuum closed form b = (2/sqrt(pi)) (dx sigma) / CNR^2.

    The hand-integrated limit of `box_bias_gaussian`; agrees with the discrete
    version once the profile is interior (W/sigma >~ 4).
    """
    return (2.0 / np.sqrt(np.pi)) * dx * sigma / cnr ** 2


def b_gaussian_closed(x: np.ndarray, sigma: float, amp: float,
                      noise_sigma: float, cnr: float, c4: float = 0.055) -> float:
    """Two-term Gaussian closed form: Box (1/CNR^2) + empirical C4/CNR^4.

    The leading term is the exact discrete Box bias; the second term is the
    next order in the asymptotic series, with C4 ~= 0.055 sigma0^2 pinned by MC
    at base geometry (its geometry scaling is under test).
    """
    return box_bias_gaussian(x, sigma, amp, noise_sigma) + c4 * sigma ** 2 / cnr ** 4


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


def _fit_variance_varmax(x, y, var_max_mult):
    """Gaussian variance fit with the upper bound set to var_max_mult * span^2.

    A local copy of fit.fit_variance with an adjustable var_max ceiling, to
    separate var_max-saturation from genuine higher-order NLS bias: if the
    low-CNR excess shrinks as the ceiling rises, the excess is the saturation
    tail; if it persists, it is the asymptotic series.
    """
    from scipy.optimize import curve_fit
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    span = float(x[-1] - x[0])
    var_min = (span / n) ** 2
    var_max = var_max_mult * span ** 2
    var0 = (span / 4.0) ** 2
    margin = 0.3 * span
    central = (x >= x[0] + margin) & (x <= x[-1] - margin)
    idx = np.where(central)[0]
    mu0 = float(x[idx][int(np.argmax(np.abs(y[idx])))]) if idx.size \
        else float(x[int(np.argmax(np.abs(y)))])
    amp_max = float(np.max(np.abs(y))) or 1.0

    def model(xx, mu, var, amp):
        return amp * np.exp(-0.5 * (xx - mu) ** 2 / var)
    try:
        popt, _ = curve_fit(model, x, y, p0=[mu0, var0, amp_max],
                            bounds=([x[0] + margin, var_min, 0.0],
                                    [x[-1] - margin, var_max, 2.0 * amp_max]),
                            maxfev=5000)
        return float(popt[1])
    except (RuntimeError, ValueError):
        return var0


def varmax_diagnostic_section(n_runs: int = 30000) -> None:
    """At CNR = 1.3, base geometry: low-CNR excess vs the var_max ceiling."""
    print("=" * 78)
    print("var_max saturation diagnostic at CNR = 1.3 (base geometry)")
    print("If excess (b_MC - b_Box) falls as var_max grows -> saturation tail")
    print("=" * 78)
    sigma, W, N, cnr = 1.0, 6.0, 241, 1.3
    x = np.linspace(-W, W, N)
    span = x[-1] - x[0]
    clean = np.exp(-0.5 * x ** 2 / sigma ** 2)
    ns = 1.0 / cnr
    b_box = box_bias_gaussian(x, sigma, 1.0, ns)
    seeds = np.random.SeedSequence(3).spawn(n_runs)
    print(f"\nb_Box = {b_box:.4e};  default var_max = span^2 = {span**2:.1f}")
    print(f"{'varmax_mult':>12} {'var_max':>10} {'b_MC':>11} {'excess':>11} "
          f"{'peg_frac':>9}")
    for mult in (1.0, 4.0, 16.0, 64.0):
        vh = np.empty(n_runs)
        for i in range(n_runs):
            rng = np.random.default_rng(seeds[i])
            y = clean + rng.normal(0.0, ns, size=N)
            vh[i] = _fit_variance_varmax(x, y, mult)
        b_mc = float(np.mean(vh)) - sigma ** 2
        peg = float(np.mean(vh > 0.9 * mult * span ** 2))
        print(f"{mult:>12.0f} {mult*span**2:>10.1f} {b_mc:>11.4e} "
              f"{b_mc-b_box:>11.4e} {peg:>9.2e}")


def correction_residual(cnr: float, sigma: float = 1.0, half_window: float = 6.0,
                        n_pixels: int = 241, n_runs: int = 40000, seed: int = 7,
                        which: str = "twoterm"):
    """Apply the closed-form correction to an MC ensemble and report residuals.

    Multiplicative, positivity-preserving correction (DERIVATION Sec 10):
    v_corr = v_hat / (1 + g), g = b_analytical / sigma0^2. Returns the corrected
    mean width and variance and their residuals from truth -- this is the actual
    stop-condition statistic (`which` selects 'box' or 'twoterm' b).
    """
    x = np.linspace(-half_window, half_window, n_pixels)
    exp = Experiment(sigma0=sigma, D=0.0, half_window=half_window,
                     n_pixels=n_pixels, n_frames=1)
    ns = noise_sigma_for_cnr0(exp, cnr)
    ens = run_ensemble(exp, float(cnr), n_runs=n_runs, seed=seed,
                       model=MODELS["gaussian"]())
    v = ens.variance[:, 0]
    v = v[np.isfinite(v)]
    if which == "box":
        b = box_bias_gaussian(x, sigma, 1.0, ns)
    else:
        b = b_gaussian_closed(x, sigma, 1.0, ns, cnr)
    g = b / sigma ** 2
    v_corr = v / (1.0 + g)
    mean_v = float(v_corr.mean())
    # Derived width = width of the corrected MEAN variance (the D-relevant
    # quantity); the variance form is the binding criterion (operator decision).
    derived_w = float(np.sqrt(max(mean_v, 0.0)))
    return {
        "cnr": float(ens.cnr_frame[0]), "b": b,
        "mean_v": mean_v, "v_resid": mean_v - sigma ** 2,
        "v_resid_pct": 100.0 * (mean_v - sigma ** 2) / sigma ** 2,
        "derived_w": derived_w, "dw_resid_pct": 100.0 * (derived_w - sigma) / sigma,
        "raw_v_pct": 100.0 * (float(v.mean()) - sigma ** 2) / sigma ** 2,
    }


def stop_condition_section(n_runs: int = 40000) -> None:
    print("=" * 78)
    print("Stop-condition test: |mean(var_corr) - var_true| <= 2% (variance form)")
    print("derived width = sqrt(mean v_corr); reported as a diagnostic")
    print("=" * 78)
    print(f"\n{'CNR':>6} {'corr':>8} {'raw_v%':>8} {'v_resid%':>9} "
          f"{'dwidth%':>8} {'PASS(2%)':>9}")
    for cnr in (8.0, 3.0, 2.0, 1.3):
        for which in ("box", "twoterm"):
            r = correction_residual(cnr, which=which, n_runs=n_runs)
            ok = "yes" if abs(r["v_resid_pct"]) <= 2.0 else "NO"
            print(f"{r['cnr']:>6.2f} {which:>8} {r['raw_v_pct']:>8.2f} "
                  f"{r['v_resid_pct']:>9.2f} {r['dw_resid_pct']:>8.3f} {ok:>9}")


def _geom_cell(sigma, W, N, n_runs):
    """Per geometry cell: Box-scaling check at CNR 8 and C4 extraction at 1.3.

    Returns (dx, dx*sigma, MC/Box at CNR 8, C4 = (b_MC - b_Box)*CNR^4 at 1.3).
    C4 is the empirical coefficient of the next-order 1/CNR^4 term; its scaling
    against dx*sigma and W is what we want to read off.
    """
    x = np.linspace(-W, W, N)
    dx = float(x[1] - x[0])
    exp = Experiment(sigma0=sigma, D=0.0, half_window=W, n_pixels=N)
    # high-CNR: Box scaling
    _, bm8, _, _ = mc_bias("gaussian", 8.0, sigma, W, N, n_runs=n_runs)
    box8 = box_bias_gaussian(x, sigma, 1.0, noise_sigma_for_cnr0(exp, 8.0))
    # low-CNR: C4
    _, bm13, _, _ = mc_bias("gaussian", 1.3, sigma, W, N, n_runs=n_runs)
    box13 = box_bias_gaussian(x, sigma, 1.0, noise_sigma_for_cnr0(exp, 1.3))
    c4 = (bm13 - box13) * 1.3 ** 4
    return dx, dx * sigma, bm8 / box8, c4


def gaussian_geometry_sweep(n_runs: int = 15000) -> None:
    """Pin the Box scaling (CNR 8) and the C4 coefficient (CNR 1.3) vs geometry."""
    print("\n" + "=" * 78)
    print("Gaussian geometry sweep: Box scaling (CNR8) + C4 = excess*CNR^4 (CNR1.3)")
    print("C4 in sigma0^2 units; watch whether C4 tracks dx*sigma, (dx*sigma)^2, or W")
    print("=" * 78)
    print("\n-- pixel size dx (vary N at W = 6, sigma = 1) --")
    print(f"{'N':>6} {'dx':>8} {'dx*sig':>8} {'MC/Box@8':>9} {'C4@1.3':>9} "
          f"{'C4/(dxsig)':>10}")
    for N in (121, 161, 241, 321, 481):
        dx, dxs, ratio, c4 = _geom_cell(1.0, 6.0, N, n_runs)
        print(f"{N:>6} {dx:>8.4f} {dxs:>8.4f} {ratio:>9.2f} {c4:>9.4f} "
              f"{c4/dxs:>10.3f}")
    print("\n-- window W (vary half_window at dx ~ 0.05, sigma = 1) --")
    print(f"{'W':>6} {'N':>6} {'dx':>8} {'MC/Box@8':>9} {'C4@1.3':>9}")
    for W in (3.0, 4.0, 6.0, 8.0, 10.0):
        N = int(round(2 * W / 0.05)) + 1
        dx, dxs, ratio, c4 = _geom_cell(1.0, W, N, n_runs)
        print(f"{W:>6.0f} {N:>6} {dx:>8.4f} {ratio:>9.2f} {c4:>9.4f}")
    print("\n-- width sigma (fixed absolute grid dx = 0.05, window = +/-6) --")
    print(f"{'sigma':>6} {'W/sig':>6} {'dx*sig':>8} {'MC/Box@8':>9} {'C4@1.3':>9} "
          f"{'C4/(dxsig)':>10}")
    for sg in (0.5, 0.7, 1.0, 1.4, 2.0):
        half_window = 6.0 / sg          # absolute window = +/-6, dx = 0.05
        dx, dxs, ratio, c4 = _geom_cell(sg, half_window, 241, n_runs)
        print(f"{sg:>6.2f} {6.0/sg:>6.1f} {dxs:>8.4f} {ratio:>9.2f} {c4:>9.4f} "
              f"{c4/dxs:>10.3f}")


def _profile_v_builder(model_name: str, x: np.ndarray, **kw):
    """Return profile_v(v, amp, mean) for the numeric Box predictor."""
    from weighting_wall.models import (_voigt_peaknorm, _gen_gaussian,
                                       _genstd_var_factor)
    if model_name == "voigt":
        gamma = kw.get("gamma", 0.3)
        return lambda v, a, m: _voigt_peaknorm(x, a, m, np.sqrt(v), gamma)
    if model_name == "generalized_gaussian":
        p = kw.get("shape", 2.0)
        fac = _genstd_var_factor(p)
        return lambda v, a, m: _gen_gaussian(x, a, m, np.sqrt(v / fac), p)
    return lambda v, a, m: a * np.exp(-0.5 * (x - m) ** 2 / v)


def shapes_section(n_runs: int = 25000, c4: float = 0.055) -> None:
    """MC-validate the numeric Box prediction and the CNR=1.3 variance criterion
    for Voigt (gamma) and generalized-Gaussian (p)."""
    print("=" * 78)
    print("Voigt / GenGauss: numeric Box vs MC, and CNR=1.3 variance criterion")
    print(f"Two-term b = box_numeric + {c4}/CNR^4; criterion |v_resid| <= 2%")
    print("=" * 78)
    sigma, W, N = 1.0, 6.0, 241
    x = np.linspace(-W, W, N)
    cases = [("voigt", {"gamma": 0.1}), ("voigt", {"gamma": 0.3}),
             ("voigt", {"gamma": 0.5}),
             ("generalized_gaussian", {"shape": 1.5}),
             ("generalized_gaussian", {"shape": 2.0}),
             ("generalized_gaussian", {"shape": 3.0})]
    print(f"\n{'model':>22} {'CNR':>5} {'b_MC':>11} {'b_Box':>11} {'MC/Box':>7} "
          f"{'raw_v%':>7} {'box_v%':>7} {'2term_v%':>8}")
    for name, kw in cases:
        profile_v = _profile_v_builder(name, x, **kw)
        exp = Experiment(sigma0=sigma, D=0.0, half_window=W, n_pixels=N,
                         n_frames=1)
        label = f"{name}{list(kw.values())[0]}"
        for cnr in (8.0, 1.3):
            ens = run_ensemble(exp, float(cnr), n_runs=n_runs, seed=11,
                               model=MODELS[name](**kw))
            col = ens.variance[:, 0]
            col = col[np.isfinite(col)]
            b_mc = float(col.mean()) - sigma ** 2
            ns = noise_sigma_for_cnr0(exp, cnr)
            b_box = box_bias_numeric(profile_v, sigma ** 2, 1.0, 0.0, ns,
                                     x_scale=sigma)
            b_2t = b_box + c4 * sigma ** 2 / cnr ** 4
            raw_v = 100.0 * b_mc / sigma ** 2
            box_v = 100.0 * (col / (1 + b_box / sigma ** 2)).mean() \
                - 100.0   # mean corrected variance residual %
            tt_v = 100.0 * (col / (1 + b_2t / sigma ** 2)).mean() - 100.0
            print(f"{label:>22} {cnr:>5.1f} {b_mc:>11.3e} {b_box:>11.3e} "
                  f"{b_mc/b_box if b_box else float('nan'):>7.2f} "
                  f"{raw_v:>7.2f} {box_v:>7.2f} {tt_v:>8.2f}")


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


def selftest_section() -> None:
    """Cheap, MC-free checks of the numeric Box predictor against the analytic
    Gaussian, and its values for Voigt / generalized-Gaussian profiles."""
    from weighting_wall.models import (_voigt_peaknorm, _gen_gaussian,
                                       _genstd_var_factor)
    print("=" * 78)
    print("Box predictor self-test (no MC): numeric vs analytic")
    print("=" * 78)
    sigma, W, N, cnr = 1.0, 6.0, 241, 8.0
    x = np.linspace(-W, W, N)
    v = sigma ** 2
    ns = 1.0 / cnr

    def gauss_v(vv, amp, mean):
        return amp * np.exp(-0.5 * (x - mean) ** 2 / vv)
    b_an = box_bias_gaussian(x, sigma, 1.0, ns)
    b_num = box_bias_numeric(gauss_v, v, 1.0, 0.0, ns, x_scale=sigma)
    print(f"\nGaussian:   analytic {b_an:.5e}  numeric {b_num:.5e}  "
          f"ratio {b_num/b_an:.4f}  (continuum {(2/np.sqrt(np.pi))*(x[1]-x[0]):.5e}/CNR^2)")

    for gamma in (0.1, 0.3, 0.5):
        def voigt_v(vv, amp, mean, g=gamma):
            return _voigt_peaknorm(x, amp, mean, np.sqrt(vv), g)
        b = box_bias_numeric(voigt_v, v, 1.0, 0.0, ns, x_scale=sigma)
        print(f"Voigt g={gamma}: numeric b = {b:.5e}  "
              f"(b*CNR^2 = {b*cnr**2:.5e} sigma0^2)")

    for p in (1.5, 2.0, 3.0):
        fac = _genstd_var_factor(p)
        def gen_v(vv, amp, mean, pp=p, ff=fac):
            return _gen_gaussian(x, amp, mean, np.sqrt(vv / ff), pp)
        b = box_bias_numeric(gen_v, v, 1.0, 0.0, ns, x_scale=sigma)
        print(f"GenGauss p={p}: numeric b = {b:.5e}  "
              f"(b*CNR^2 = {b*cnr**2:.5e} sigma0^2)")


def main() -> None:
    section = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    if section in ("selftest", "all"):
        selftest_section()
    if section in ("gaussian", "all"):
        gaussian_section()
    if section in ("gaussian_geom", "all"):
        gaussian_geometry_sweep()
    if section in ("varmax", "all"):
        varmax_diagnostic_section()
    if section in ("stop", "all"):
        stop_condition_section()
    if section in ("shapes", "all"):
        shapes_section()
    if section in ("moment", "all"):
        moment_section()


if __name__ == "__main__":
    main()
