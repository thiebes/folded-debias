"""Cramer-Rao lower bound on the diffusion coefficient.

The CRLB is a property of the measurement model alone -- it does not depend on
which estimator is used. It is the floor that *no* unbiased estimator of D can
beat: Var(D_hat) >= CRLB. It therefore answers the question the fresh-eyes
derivation cannot answer by itself -- "how close to optimal is the estimator?"
-- and separates the removable artefact (bias) from the irreducible cost
(variance).

Model (as simulated by ``weighting_wall.simulate``): at frame i the noise-free
profile is mu(x) = A_i * exp(-x^2 / (2 v_i)), with v_i = sigma_0^2 + 2 D t_i and
A_i the peak amplitude. Each pixel carries iid Gaussian noise of standard
deviation sigma_noise. For Gaussian noise with known sigma_noise the Fisher
information matrix is

    F_{ab} = (1 / sigma_noise^2) * sum_{pixels,frames} (d mu / d theta_a)
                                                       (d mu / d theta_b).

Parameters: theta = (sigma_0^2, D, A_1..A_N, c_1..c_N), where c_i is the
per-frame centre. sigma_0^2 and D are structural; A_i and c_i are per-frame
nuisances. Var(D_hat) >= [F^-1]_{DD}.

Run from the repository root::

    python experiments/crlb-precision/crlb.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from weighting_wall.montecarlo import run_ensemble  # noqa: E402
from weighting_wall.simulate import Experiment, noise_sigma_for_cnr0  # noqa: E402


def fisher_information(exp: Experiment, noise_sigma: float) -> np.ndarray:
    """Full Fisher information matrix for theta = (sigma_0^2, D, {A_i}, {c_i})."""
    x = exp.x
    t = exp.times
    v = exp.true_variance          # sigma_0^2 + 2 D t
    amp = exp.amplitude            # per-frame peak amplitude
    n = exp.n_frames
    nparam = 2 + 2 * n
    fim = np.zeros((nparam, nparam))

    for i in range(n):
        g = np.exp(-x ** 2 / (2.0 * v[i]))          # shape profile
        dv = amp[i] * g * x ** 2 / (2.0 * v[i] ** 2)  # d mu / d v_i
        cols = np.zeros((nparam, x.size))
        cols[0] = dv                 # d mu / d sigma_0^2
        cols[1] = dv * 2.0 * t[i]    # d mu / d D
        cols[2 + i] = g              # d mu / d A_i
        cols[2 + n + i] = amp[i] * g * x / v[i]      # d mu / d c_i
        fim += cols @ cols.T

    return fim / noise_sigma ** 2


def crlb_D(exp: Experiment, cnr0: float) -> float:
    """Var(D_hat) floor -- the (D,D) element of the inverse Fisher matrix."""
    noise_sigma = noise_sigma_for_cnr0(exp, cnr0)
    fim = fisher_information(exp, noise_sigma)
    return float(np.linalg.inv(fim)[1, 1])


def per_frame_width_crlb(exp: Experiment, noise_sigma: float) -> np.ndarray:
    """Var(sigma_hat^2) floor per frame, with the per-frame A and centre as
    nuisances. The variance estimate of a single isolated frame cannot beat
    this."""
    x = exp.x
    v = exp.true_variance
    amp = exp.amplitude
    out = np.empty(exp.n_frames)
    for i in range(exp.n_frames):
        g = np.exp(-x ** 2 / (2.0 * v[i]))
        dv = amp[i] * g * x ** 2 / (2.0 * v[i] ** 2)
        dA = g
        dc = amp[i] * g * x / v[i]
        jac = np.stack([dv, dA, dc])
        fim = jac @ jac.T / noise_sigma ** 2
        out[i] = float(np.linalg.inv(fim)[0, 0])
    return out


def _analytic_per_frame(exp: Experiment, noise_sigma: float) -> np.ndarray:
    """Continuum-limit per-frame width CRLB: 8/sqrt(pi) * sigma_n^2 dx v^1.5 / A^2.
    (A and centre as nuisances; the centre decouples by parity.)"""
    dx = float(exp.x[1] - exp.x[0])
    v = exp.true_variance
    amp = exp.amplitude
    return (8.0 / np.sqrt(np.pi)) * noise_sigma ** 2 * dx * v ** 1.5 / amp ** 2


def main() -> None:
    # Low-CNR, low-D regime from the fresh-eyes problem statement: L_D/FWHM_0 ~ 0.04.
    fwhm0 = 2.3548
    ld_target = 0.04 * fwhm0
    D = ld_target ** 2 / 2.0                       # L_D = sqrt(2 D tau), tau=1
    exp = Experiment(sigma0=1.0, D=D, tau=1.0, n_frames=12, t_max=3.0)
    print("Cramer-Rao bound on D")
    print(f"  D = {D:.5f},  L_D/FWHM_0 = {np.sqrt(2*D)/fwhm0:.3f},  "
          f"n_frames = {exp.n_frames}\n")

    print("  CNR_0   CRLB Var(D)    sqrt(CRLB)/D   meaning")
    for cnr0 in (3.0, 5.0, 8.0, 15.0, 30.0, 80.0):
        var_d = crlb_D(exp, cnr0)
        rel = np.sqrt(var_d) / D
        note = ("single shot cannot pin D" if rel > 1 else
                "single shot informative")
        print(f"  {cnr0:5.0f}   {var_d:11.3e}   {rel:11.2f}    {note}")

    # Internal check: the full-matrix D-CRLB must equal the CRLB-weighted
    # slope variance built from the per-frame width CRLBs.
    cnr0 = 5.0
    ns = noise_sigma_for_cnr0(exp, cnr0)
    pf = per_frame_width_crlb(exp, ns)
    w = 1.0 / pf
    t = exp.times
    tbar = np.sum(w * t) / np.sum(w)
    var_d_slope = 1.0 / (4.0 * np.sum(w * (t - tbar) ** 2))
    var_d_full = crlb_D(exp, cnr0)
    print(f"\n  consistency check (CNR_0=5):")
    print(f"    D-CRLB from full Fisher matrix      = {var_d_full:.4e}")
    print(f"    D-CRLB from per-frame widths + slope = {var_d_slope:.4e}")
    print(f"    (equal => the two-stage MSD pipeline loses no information)")

    ana = _analytic_per_frame(exp, ns)
    print(f"\n  per-frame width CRLB, exact vs continuum formula (CNR_0=5):")
    print(f"    frame   CNR_i   exact Var(sigma^2)   analytic   ratio")
    cnr_i = exp.amplitude / ns
    for i in (0, 3, 6, 9, 11):
        print(f"    {i:5d}  {cnr_i[i]:6.2f}   {pf[i]:15.3e}   {ana[i]:.3e}  "
              f"{pf[i]/ana[i]:.3f}")

    _validate(exp)


def _validate(exp: Experiment) -> None:
    """Check the CRLB against simulation: at high CNR the Gaussian least-squares
    fit is maximum-likelihood and attains the CRLB, so the empirical variance of
    the fitted width should match the per-frame bound."""
    print("\n  validation -- empirical Var(sigma_hat^2) vs CRLB at high CNR")
    cnr0 = 80.0
    ns = noise_sigma_for_cnr0(exp, cnr0)
    pf = per_frame_width_crlb(exp, ns)
    ens = run_ensemble(exp, cnr0, n_runs=2000, seed=0)
    print("    frame   CNR_i    empirical Var     CRLB        ratio")
    for i in (0, 3, 6, 9, 11):
        col = ens.variance[:, i]
        col = col[np.isfinite(col)]
        emp = float(np.var(col, ddof=1))
        cnr_i = exp.amplitude[i] / ns
        print(f"    {i:5d}  {cnr_i:6.2f}   {emp:13.3e}   {pf[i]:.3e}  "
              f"{emp / pf[i]:.2f}")
    print("    (ratios near 1 confirm the CRLB; >1 = the fitter is not yet")
    print("     fully efficient at this CNR, which is expected)")


if __name__ == "__main__":
    main()
