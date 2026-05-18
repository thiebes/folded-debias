"""Task 1 (per the project brief): the clipping-vs-reflection diagnostic.

The correction-factor derivation forks on *how* the production fit enforces
the non-negativity of the width:

* **Clipping** -- the latent unconstrained variance estimate is censored at the
  lower bound, leaving a delta-mass (pile-up) there.  Route: Tobit /
  inverse-Mills-ratio censored regression.
* **Reflection** -- negative latent values are folded to positive ones; the
  distribution is strictly positive and right-skewed with no pile-up.  Route:
  folded-/half-normal.

The two routes give *different* truncation factors g(p), so this must be
settled before any derivation. This script runs a single-frame Monte Carlo at
a fixed true variance across a CNR sweep, fits each profile with the production
routine (``weighting_wall.fit.fit_variance``), and inspects the empirical PDF
of the fitted variance.

Run from the repository root::

    python scripts/diagnose_boundary.py

Writes ``output/boundary_diagnostic.png`` and prints a per-CNR verdict.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighting_wall.fit import _gaussian_var, fit_variance  # noqa: E402
from weighting_wall.simulate import gaussian  # noqa: E402

TRUE_SIGMA = 1.0                      # fixed true width
TRUE_VAR = TRUE_SIGMA ** 2
CNR_GRID = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0])
N_REPEAT = 6000
HALF_WINDOW = 6.0
N_PIXELS = 241
AMPLITUDE = 1.0


def _x_axis() -> np.ndarray:
    return np.linspace(-HALF_WINDOW * TRUE_SIGMA, HALF_WINDOW * TRUE_SIGMA, N_PIXELS)


def _fit_bounds(x: np.ndarray) -> tuple[float, float]:
    """The lower/upper variance bounds the production fit imposes."""
    span = float(x[-1] - x[0])
    return (span / x.size) ** 2, span ** 2


def diagnose() -> None:
    x = _x_axis()
    var_min, var_max = _fit_bounds(x)
    clean = gaussian(x, AMPLITUDE, 0.0, TRUE_SIGMA)
    rng = np.random.default_rng(20260517)

    print("Clipping-vs-reflection diagnostic")
    print(f"  true variance = {TRUE_VAR:.3f}; fit variance bounds = "
          f"[{var_min:.4f}, {var_max:.1f}]")
    print(f"  {N_REPEAT} fits per CNR level\n")
    print("   CNR   mean(var)  median   skew   P(var<true)  wall-mass   verdict")

    per_cnr_samples = []
    for cnr in CNR_GRID:
        noise = AMPLITUDE / cnr
        fitted = np.empty(N_REPEAT)
        for i in range(N_REPEAT):
            y = clean + rng.normal(0.0, noise, size=x.size)
            fitted[i] = fit_variance(x, y).variance
        per_cnr_samples.append(fitted)

        wall_mass = np.mean(fitted <= 1.05 * var_min)   # pile-up at lower bound
        ceil_mass = np.mean(fitted >= 0.95 * var_max)    # runaway pile-up at top
        d = fitted - fitted.mean()
        s2 = (d ** 2).mean()
        skew = (d ** 3).mean() / s2 ** 1.5 if s2 > 0 else 0.0
        below = np.mean(fitted < TRUE_VAR)

        if wall_mass > 0.02:
            verdict = f"CLIPPING (wall pile-up {wall_mass:.0%})"
        elif ceil_mass > 0.02:
            verdict = f"RUNAWAY ceiling pile-up {ceil_mass:.0%}"
        elif skew > 0.5:
            verdict = "reflection-like (right-skew, no pile-up)"
        else:
            verdict = "near-Gaussian"
        print(f"  {cnr:5.1f}  {fitted.mean():9.3f} {np.median(fitted):8.3f} "
              f"{skew:7.2f}  {below:9.2f}   {wall_mass:8.0%}   {verdict}")

    _verdict_summary(per_cnr_samples, var_min, var_max)
    _plot(per_cnr_samples, var_min)


def _verdict_summary(samples, var_min, var_max) -> None:
    low = samples[0]                                     # lowest-CNR level
    wall = np.mean(low <= 1.05 * var_min)
    ceil = np.mean(low >= 0.95 * var_max)
    print("\nOverall:")
    if wall > 0.02:
        print("  -> The fit CLIPS at the lower variance bound: a delta-mass")
        print("     accumulates there as CNR falls. Derivation route: Tobit /")
        print("     inverse-Mills-ratio censored regression (note the censoring")
        print("     point is the positive bound var_min, not exactly zero).")
    elif ceil > 0.02:
        print("  -> The fit produces a RUNAWAY pile-up at the upper bound.")
        print("     The heavy right tail dominates; treat the upper bound as the")
        print("     censoring point and report this regime explicitly.")
    else:
        print("  -> No boundary pile-up: the fitted-variance distribution is")
        print("     strictly interior and right-skewed. Derivation route:")
        print("     folded-/half-normal.")
    print("  Confirm against the six priority papers before deriving "
          "(see citations_audit.md).")


def _plot(samples, var_min) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed -- skipping figure]")
        return

    n = len(CNR_GRID)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.2), sharey=True)
    for ax, cnr, s in zip(axes, CNR_GRID, samples):
        s_clip = np.clip(s, 0, 4 * TRUE_VAR)
        ax.hist(s_clip, bins=60, color="tab:purple", alpha=0.8)
        ax.axvline(TRUE_VAR, color="k", ls="--", lw=1)
        ax.axvline(var_min, color="tab:red", ls=":", lw=1)
        ax.set_title(f"CNR = {cnr:g}")
        ax.set_xlabel("fitted variance")
    axes[0].set_ylabel("count")
    fig.suptitle("Fitted-variance PDF vs CNR  "
                 "(black = true, red = lower bound)")
    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", "boundary_diagnostic.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    diagnose()
