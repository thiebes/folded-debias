"""Width estimators compared on the Saha model: moment vs Gaussian vs Voigt fit.

A controlled, repo-consistent comparison on Marc's Saha transport model. It puts
the three weighting-wall estimators on the SAME noise-free profiles and the SAME
zero-mean profile-noise realisations (the project's noise convention) and asks
how each behaves as a frame's CNR falls:

  * MomentModel        -- the intensity-weighted second moment. Linear in the
    data and admits negative variance, so it is approximately UNBIASED; its
    weakness is variance, which the x^2 weighting inflates at low CNR.
  * GaussianModel      -- a bounded Gaussian fit. Lower variance, but the width
    cannot go negative, so it is truncation-BIASED upward at low CNR (the bias
    folded-normal removes).
  * VoigtModel         -- the Prins-group estimator: a bounded fit of a
    diffusing Gaussian convolved with a fixed Lorentzian PSF. Matches the
    non-Gaussian (n^2-sharpened) emission better, so it carries the smallest
    shape offset, while sharing the bounded fit's low-CNR truncation bias.

The point is the trade-off, not a single villain: the moment trades bias for
variance; the bounded fits trade variance for a correctable bias; the Voigt fit
additionally minimises the noise-free shape offset. This is the evidence for
recommending a bounded Voigt fit plus the folded-normal correction over a raw
second moment.

Run: .venv/Scripts/python.exe mc_experiments/saha_reaction_diffusion/estimator_robustness.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from weighting_wall.models import GaussianModel, MomentModel, VoigtModel  # noqa: E402

import reaction_diffusion_saha as m  # noqa: E402

REALISATIONS = 400
SEED = 11


def main():
    rng = np.random.default_rng(SEED)

    par = m.Parameters(L=8.0, nbins=320, De=0.04, Df=0.26,
                       nu=0.06, rho=1e-4, A=10.0, qe=0.3, qf=0.9,
                       tmax=8.0, tsamp=0.08, dt=0.001)
    y = m.x_axis(par)
    initial = np.exp(-y ** 2 / (2.0 * 0.3 ** 2))
    clean = m.simulate(par, initial).emission
    times = m.sample_times(par)

    estimators = {
        "moment": MomentModel(),
        "Gaussian fit": GaussianModel(),
        "Voigt fit": VoigtModel(gamma=0.3),
    }

    # Noise-free MSD for each estimator (differ by profile-shape sensitivity).
    truth = {}
    for name, est in estimators.items():
        v = np.array([est.fit_variance(y, clean[j]).variance for j in range(len(times))])
        truth[name] = v - v[0]

    print("Noise-free MSD by estimator (the Saha emission is non-Gaussian):")
    print(f"{'t (ns)':>7}" + "".join(f"{n:>16}" for n in estimators))
    for tt in (2.0, 4.0, 6.0, 8.0):
        j = int(np.argmin(np.abs(times - tt)))
        print(f"{times[j]:>7.1f}" + "".join(f"{truth[n][j]:>16.3f}" for n in estimators))

    # Noisy comparison at a representative late frame. Zero-mean white noise at
    # the profile level (the project invariant), sigma set by the per-frame CNR.
    jrep = int(np.argmin(np.abs(times - 6.0)))
    frame = clean[jrep]
    peak = frame.max()
    truth_v = {n: est.fit_variance(y, frame).variance for n, est in estimators.items()}

    print(f"\nNoise robustness at t={times[jrep]:.1f} ns "
          f"(bias [scatter], relative to each estimator's noise-free value):")
    print(f"{'CNR':>6}" + "".join(f"{n:>20}" for n in estimators))

    rows = {n: [] for n in estimators}
    cnrs = []
    for cnr in (40, 20, 10, 5, 3, 2, 1.5):
        sigma = peak / cnr
        cnrs.append(cnr)
        cells = []
        for name, est in estimators.items():
            vals = np.empty(REALISATIONS)
            for r in range(REALISATIONS):
                noisy = frame + rng.normal(0.0, sigma, size=frame.shape)
                vals[r] = est.fit_variance(y, noisy).variance
            bias = (np.nanmean(vals) - truth_v[name]) / truth_v[name]
            scatter = np.nanstd(vals) / truth_v[name]
            rows[name].append((bias, scatter))
            cells.append(f"{bias:>+7.0%} [{scatter:>4.0%}]")
        print(f"{cnr:>6.1f}" + "".join(f"{c:>20}" for c in cells))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colours = {"moment": "r", "Gaussian fit": "g", "Voigt fit": "b"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in estimators:
        ax1.plot(times, truth[name], colours[name], lw=1.6, label=name)
    ax1.set_xlabel("time (ns)"); ax1.set_ylabel("MSD (um$^2$)")
    ax1.legend(fontsize=8, frameon=False)

    cnrs = np.array(cnrs)
    for name in estimators:
        b = np.array([r[0] for r in rows[name]])
        s = np.array([r[1] for r in rows[name]])
        ax2.errorbar(cnrs, b, yerr=s, fmt="o-", color=colours[name], ms=4,
                     lw=1, capsize=2, label=name)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel("per-frame CNR"); ax2.set_ylabel("width bias [+/- scatter]")
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join("output", "estimator_robustness.png")
    os.makedirs("output", exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
