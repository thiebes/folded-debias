"""Quantify the two distortions between the carrier distribution and the fit.

The validation framework (docs/VALIDATION_DESIGN.md) names two distinct gaps on
the path from the recorded profile to the carrier distribution:

* the within-fit SHAPE gap -- a single-shape fit of the emission profile vs that
  profile's own second moment (a selection issue), and
* the EMISSION-TO-CARRIER gap -- the emission profile's second moment vs the
  carrier distribution's second moment, because emission is nonlinear in the
  carrier density n (E = qe nu (1-x) n + qf rho x^2 n^2).

Truth is the carrier distribution. This script measures all three alpha values
noise-free across the density-coupling sweep, using the carrier population map
the Saha port already returns, so both gaps are separated and sized:

* alpha_carrier  -- from the second moment of the population (carrier) map. TRUTH.
* alpha_emission -- from the second moment of the emission map (the stage-1
                    width-extraction target).
* alpha_fit      -- from a Gaussian fit of the emission map (a parametric
                    estimator).

It needs no noise, calibration, or Monte Carlo: it is a property of the
noise-free forward model. Run from the repository root::

    .venv/Scripts/python.exe mc_experiments/saha_reaction_diffusion/carrier_vs_emission.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

import reaction_diffusion_saha as rd  # noqa: E402
from weighting_wall.models import GaussianModel  # noqa: E402

# Match the validation harness's Saha parameters and sweep.
BASE = dict(L=24.0, nbins=960, De=0.005, Df=0.1, nu=0.25, rho=1e-3,
            tmax=3.0, dt=0.001, sigma0_init=0.35)
A_SWEEP = np.geomspace(30.0, 2000.0, 11)
N_FRAMES = 12
GAUSSIAN = GaussianModel()


def _second_moment(x: np.ndarray, profile: np.ndarray) -> float:
    """Second central moment of a non-negative profile on grid x."""
    w = np.clip(profile, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.nan
    mean = float((x * w).sum() / s)
    return float(((x - mean) ** 2 * w).sum() / s)


def _alpha(times: np.ndarray, var: np.ndarray) -> float:
    """Anomalous exponent: slope of log MSD vs log t, MSD(t) = var(t) - var(0)."""
    msd = var - var[0]
    keep = (times > 0) & (msd > 0)
    if keep.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(times[keep]), np.log(msd[keep]), 1)[0])


def alphas_for(A: float, emission_power: int) -> tuple[float, float, float]:
    """Return (alpha_carrier, alpha_emission, alpha_fit) for one regime.

    The second moments use the native, wide simulation grid so the tails are
    captured; the Gaussian fit uses the same grid.
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

    carrier = np.array([_second_moment(xs, res.population[i]) for i in idx])
    emission = np.array([_second_moment(xs, res.emission[i]) for i in idx])
    fit = np.array([GAUSSIAN.fit_variance(xs, res.emission[i]).variance
                    for i in idx])
    return _alpha(times, carrier), _alpha(times, emission), _alpha(times, fit)


def main() -> None:
    print("Carrier vs emission: the two distortions, noise-free")
    print(f"  Saha: De={BASE['De']}, Df={BASE['Df']} um^2/ns, "
          f"nu={BASE['nu']}, rho={BASE['rho']} /ns\n")
    for power in (1, 2):
        print(f"emission power {power}  "
              f"({'linear-x (old C)' if power == 1 else 'x^2 (paper eq. 9)'})")
        print("      A   a_carrier  a_emission  a_fit   emis-carr  fit-emis")
        for A in A_SWEEP:
            ac, ae, af = alphas_for(float(A), power)
            print(f"  {A:7.0f}    {ac:6.3f}     {ae:6.3f}   {af:6.3f}   "
                  f"{ae - ac:+.3f}     {af - ae:+.3f}")
        print()
    print("emis-carr = emission-to-carrier gap (needs the forward model to invert)")
    print("fit-emis  = within-fit shape gap (a selection issue)")


if __name__ == "__main__":
    main()
