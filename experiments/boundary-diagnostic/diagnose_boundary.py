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
settled before any derivation -- and it must be settled *per profile model*,
because Gaussian, Voigt, and generalized-Gaussian fits may behave differently.

This script runs a single-frame Monte Carlo at a fixed true variance across a
CNR sweep, fits each profile with the chosen model, and inspects the empirical
PDF of the fitted variance.

Run from the repository root::

    python scripts/diagnose_boundary.py [gaussian|voigt|generalized_gaussian]

Writes ``output/boundary_diagnostic_<model>.png`` and prints a per-CNR verdict.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import MODELS  # noqa: E402
from folded_debias.simulate import Experiment  # noqa: E402

TRUE_SIGMA = 1.0
TRUE_VAR = TRUE_SIGMA ** 2
CNR_GRID = np.array([1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 30.0])
N_REPEAT = 6000
AMPLITUDE = 1.0


def _model_from_argv():
    name = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    if name not in MODELS:
        sys.exit(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return name, MODELS[name]()


def diagnose() -> None:
    name, model = _model_from_argv()
    # A one-frame, non-diffusing experiment fixes the true variance at TRUE_VAR.
    exp = Experiment(sigma0=TRUE_SIGMA, D=0.0, amplitude0=AMPLITUDE,
                     n_frames=1, t_max=0.0)
    x = exp.x
    clean = model.noise_free_series(exp)[0]
    peak = float(clean.max())
    span = float(x[-1] - x[0])
    var_floor = (span / x.size) ** 2
    rng = np.random.default_rng(20260517)

    print(f"Clipping-vs-reflection diagnostic -- model: {name}")
    print(f"  true variance = {TRUE_VAR:.3f}; {N_REPEAT} fits per CNR level\n")
    print("   CNR   mean(var)  median   skew   P(var<true)  wall-mass   verdict")

    per_cnr = []
    for cnr in CNR_GRID:
        noise = peak / cnr
        fitted = np.array([
            model.fit_variance(x, clean + rng.normal(0.0, noise, size=x.size)).variance
            for _ in range(N_REPEAT)
        ])
        per_cnr.append(fitted)

        wall = np.mean(fitted <= 4.0 * var_floor)     # pile-up near lower bound
        d = fitted - fitted.mean()
        s2 = (d ** 2).mean()
        skew = (d ** 3).mean() / s2 ** 1.5 if s2 > 0 else 0.0
        below = np.mean(fitted < TRUE_VAR)
        if wall > 0.02:
            verdict = f"CLIPPING (wall pile-up {wall:.0%})"
        elif skew > 0.5:
            verdict = "reflection-like (right-skew, no pile-up)"
        else:
            verdict = "near-Gaussian"
        print(f"  {cnr:5.1f}  {fitted.mean():9.3f} {np.median(fitted):8.3f} "
              f"{skew:7.2f}  {below:9.2f}   {wall:8.0%}   {verdict}")

    _summary(name, per_cnr, var_floor)
    _plot(name, per_cnr, var_floor)


def _summary(name, samples, var_floor) -> None:
    wall = np.mean(samples[0] <= 4.0 * var_floor)
    print(f"\nOverall ({name}):")
    if wall > 0.02:
        print("  -> CLIPPING: a delta-mass accumulates at the lower bound as")
        print("     CNR falls. Route: Tobit / inverse-Mills-ratio.")
    else:
        print("  -> REFLECTION: the fitted-variance distribution is strictly")
        print("     interior and right-skewed, no pile-up. Route: folded-normal.")
    print("  Run this for every profile model used on the target data.")


def _plot(name, samples, var_floor) -> None:
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
        ax.hist(np.clip(s, 0, 4 * TRUE_VAR), bins=60, color="tab:purple", alpha=0.8)
        ax.axvline(TRUE_VAR, color="k", ls="--", lw=1)
        ax.axvline(var_floor, color="tab:red", ls=":", lw=1)
        ax.set_title(f"CNR = {cnr:g}")
        ax.set_xlabel("fitted variance")
    axes[0].set_ylabel("count")
    fig.suptitle(f"Fitted-variance PDF vs CNR -- {name} model "
                 "(black = true, red = lower bound)")
    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    out = os.path.join("output", f"boundary_diagnostic_{name}.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    diagnose()
