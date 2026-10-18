"""Where between pure shot and pure white does a real detector sit?

`shot_noise_validation.py` showed the two limits: under constant (white) noise the
MSD-slope bias is huge (D inflated by factors of dozens at low CNR), while under
photon-limited shot noise (read floor = 0) it nearly vanishes, because the clean
wings constrain the width fit. Real photon-counting detectors are in between: a
read-noise / dark-count / background pedestal adds a constant-variance floor that
reintroduces wing noise and brings the bias back. The relevant axis is the read
floor.

This sweep holds the geometry fixed and raises the read floor `r` (constant noise
in count units, added in quadrature to the local shot variance) from 0 upward.
For each `r` it reports, summarised as mean |mean(D_hat)/D_true - 1| over the
CNR_0 sweep:

  OLS raw            -- the uncorrected bias, climbing from ~0 (pure shot) toward
                        the white level as the pedestal grows.
  folded, matched    -- the correction with a calibration built at the same read
                        floor (CORRECTNESS).
  folded, white cal  -- the shipped white calibration applied to this data
                        (ROBUSTNESS): it over-corrects at small r and becomes
                        correct only as the law approaches white.

The pure-white case is run as the large-r reference. Reading the read floor in
physical terms needs the detector's photon budget: the floor here is in the same
counts as the peak, which holds cnr0**2 counts at frame 0, so a floor of r counts
matters most for the dim, low-CNR frames where peak counts are fewest.

Run from the repository root (set FOLDED_DEBIAS_WORKERS to parallelise)::

    FOLDED_DEBIAS_WORKERS=0 .venv/Scripts/python.exe scripts/read_floor_sweep.py

Sweep sizes follow the SHOT_* environment variables (small defaults for a quick
serial smoke run; raise for the full run).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias.noise import ShotNoise, WhiteNoise  # noqa: E402
from folded_debias.simulate import Experiment  # noqa: E402
from folded_debias.validate import calibrate_d_free, evaluate  # noqa: E402

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "..", "output")

GEOM = dict(sigma0=1.0, tau=1.0, half_window=6.0, n_pixels=241,
            n_frames=12, t_max=3.0)
D_TRUE = 0.02                       # sigma_0^2 / tau; L_D / FWHM_0 ~ 0.085

N_CAL = int(os.environ.get("SHOT_NCAL", 120))
N_EVAL = int(os.environ.get("SHOT_NEVAL", 120))
N_EVAL_GRID = int(os.environ.get("SHOT_NGRID", 8))
CAL_CNR0 = np.geomspace(2.0, 80.0, 30)
EVAL_CNR0 = np.geomspace(3.0, 60.0, N_EVAL_GRID)

# Read floors in count units (0 = photon-limited). The peak holds cnr0**2 counts
# at frame 0, so these span negligible-vs-peak-shot up to pedestal-dominated.
READ_FLOORS = np.array([0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0])


def _mad(curve: np.ndarray) -> float:
    return float(np.nanmean(np.abs(curve - 1.0)))


def main() -> None:
    print("MSD-slope bias vs detector read floor (pure shot -> pure white)")
    print(f"  D_true = {D_TRUE} sigma_0^2/tau")
    print(f"  calibration: {CAL_CNR0.size} CNR_0 in [2, 80], {N_CAL} runs/cell")
    print(f"  evaluation:  {EVAL_CNR0.size} CNR_0 in [3, 60], {N_EVAL} runs/cell")
    print(f"  read floors (counts): {READ_FLOORS.tolist()}")
    print(f"  workers = {os.environ.get('FOLDED_DEBIAS_WORKERS', '(serial)')}")

    # The shipped default: a white-noise calibration. Used as the robustness
    # reference (applied to every read-floor case) and as the large-r limit.
    print("\nBuilding the default white calibration at D = 0 ...", flush=True)
    white_cal = calibrate_d_free(CAL_CNR0, n_runs=N_CAL, noise_model=WhiteNoise(),
                                 **GEOM)

    exp = Experiment(D=D_TRUE, **GEOM)
    ols, folded_matched, folded_white = [], [], []
    for r in READ_FLOORS:
        model = ShotNoise(read=float(r), poisson=True)
        print(f"\n[read = {r:5.1f}] building matched calibration ...", flush=True)
        cal_r = calibrate_d_free(CAL_CNR0, n_runs=N_CAL, noise_model=model, **GEOM)
        r_match = evaluate(exp, EVAL_CNR0, cal_r, n_runs=N_EVAL, noise_model=model)
        r_white = evaluate(exp, EVAL_CNR0, white_cal, n_runs=N_EVAL,
                           noise_model=model)
        ols.append(_mad(r_match["ols"].mean_bias))
        folded_matched.append(_mad(r_match["folded_normal"].mean_bias))
        folded_white.append(_mad(r_white["folded_normal"].mean_bias))
        print(f"  OLS raw {ols[-1]:.3f} | folded matched {folded_matched[-1]:.3f} "
              f"| folded white-cal {folded_white[-1]:.3f}")

    # The pure-white limit, for reference.
    print("\n[white limit] evaluating white data with the white calibration ...",
          flush=True)
    r_w = evaluate(exp, EVAL_CNR0, white_cal, n_runs=N_EVAL,
                   noise_model=WhiteNoise())
    ols_white = _mad(r_w["ols"].mean_bias)
    folded_white_white = _mad(r_w["folded_normal"].mean_bias)

    ols = np.array(ols)
    folded_matched = np.array(folded_matched)
    folded_white = np.array(folded_white)

    print("\n" + "=" * 72)
    print("mean |mean(D_hat)/D_true - 1| over the CNR_0 sweep, vs read floor")
    print("=" * 72)
    print(f"  {'read':>6} {'OLS raw':>10} {'folded matched':>16} "
          f"{'folded white-cal':>18}")
    for i, r in enumerate(READ_FLOORS):
        print(f"  {r:>6.1f} {ols[i]:>10.3f} {folded_matched[i]:>16.3f} "
              f"{folded_white[i]:>18.3f}")
    print(f"  {'white':>6} {ols_white:>10.3f} {'-':>16} "
          f"{folded_white_white:>18.3f}")

    # Figure: bias vs read floor, three curves, with the white limit marked.
    # Roles 1/2/4 from the poster palette (dark green / blue / rose).
    role1, role2, role4 = (0.058184, 0.273631, 0.191837), \
        (0.239379, 0.369361, 0.547850), (0.852116, 0.555525, 0.605505)
    x = READ_FLOORS.copy()
    x[0] = 0.4                          # show read = 0 on the log axis
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.loglog(x, ols, "o-", color=role4, lw=1.8, label="uncorrected (OLS)")
    ax.loglog(x, folded_white, "s-", color=role2, lw=1.8,
              label="folded, white calibration")
    ax.loglog(x, folded_matched, "o-", color=role1, lw=1.8,
              label="folded, matched calibration")
    ax.axhline(ols_white, color=role4, ls=":", lw=1.2)
    ax.text(x[-1], ols_white, " white limit", color=role4, fontsize=7,
            va="center")
    ax.set_xlabel("read floor (counts; 0 shown at left)")
    ax.set_ylabel(r"mean $|\langle\hat D\rangle/D_\mathrm{true}-1|$")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    path = os.path.join(OUTPUT, "read_floor_sweep.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"\nSaved {os.path.basename(path)}")


if __name__ == "__main__":
    main()
