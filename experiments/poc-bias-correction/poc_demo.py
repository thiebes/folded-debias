"""Proof-of-concept demo: the truncation bias and the folded-normal correction.

Run from the repository root::

    python scripts/poc_demo.py [gaussian|voigt|generalized_gaussian]

Produces ``output/poc_demo.png`` (if matplotlib is installed) with two panels:

* (a) the truncation fraction p and MSD skewness vs per-frame CNR -- evidence
      that the wall at zero distorts the width estimator, worst at low CNR;
* (b) the bias <D_est>/D_true vs CNR_0 for OLS, standard inverse-variance, and
      the derived folded-normal scheme (see docs/DERIVATION.md).

It also prints a summary table to stdout, so it is useful with or without a
plotting backend.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Allow running from a source checkout without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from folded_debias import MODELS, Experiment, evaluate, sweep_stats  # noqa: E402
from folded_debias.truncation import TruncationModel  # noqa: E402

# Nanocrystal-solid regime: short diffusion length against the spot size.
# D and tau are chosen so that L_D = sqrt(2 D tau) ~ 0.20 sigma0, i.e.
# L_D / FWHM_0 ~ 0.085 -- the short-diffusion-length corner that disordered
# perovskite nanocrystal solids occupy and where the truncation bias bites.
EXP = Experiment(sigma0=1.0, D=0.02, tau=1.0, n_frames=12, t_max=3.0)
CNR0_GRID = np.array([5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 90.0])
N_RUNS = 400


def _profile_model_from_argv():
    """Profile model selected by an optional command-line argument."""
    name = sys.argv[1] if len(sys.argv) > 1 else "gaussian"
    if name not in MODELS:
        sys.exit(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return name, MODELS[name]()


def main() -> None:
    model_name, pmodel = _profile_model_from_argv()
    fwhm = 2.3548 * EXP.sigma0
    ld = np.sqrt(2.0 * EXP.D * EXP.tau)
    print("weighting-wall POC demo")
    print(f"  profile model: {model_name}")
    print(f"  experiment: D={EXP.D}, tau={EXP.tau}, n_frames={EXP.n_frames}, "
          f"L_D/FWHM_0={ld / fwhm:.3f}")
    print(f"  CNR_0 sweep: {CNR0_GRID}")
    print(f"  {N_RUNS} Monte Carlo runs per CNR_0\n")

    # One calibration sweep, reused for the truncation model and panel (a).
    print("Calibrating truncation model ...")
    stats = sweep_stats(EXP, CNR0_GRID, n_runs=N_RUNS, seed=1, model=pmodel)
    model: TruncationModel = TruncationModel.from_stats(stats)

    # Independent evaluation sweep (different seed).
    print("Evaluating schemes ...")
    results = evaluate(EXP, CNR0_GRID, model, n_runs=N_RUNS,
                       seed=20000, profile_model=pmodel)

    # --- summary table ---------------------------------------------------
    print("\n  <D_est> / D_true   (1.000 = unbiased)")
    print("  CNR_0     OLS     inv-var   folded-normal")
    for i, c0 in enumerate(CNR0_GRID):
        print(f"  {c0:6.0f}  {results['ols'].mean_bias[i]:7.3f}  "
              f"{results['inverse_variance'].mean_bias[i]:7.3f}  "
              f"{results['folded_normal'].mean_bias[i]:11.3f}")

    ols = results["ols"].mean_bias
    iv = results["inverse_variance"].mean_bias
    fn = results["folded_normal"].mean_bias
    f = np.isfinite(ols) & np.isfinite(iv) & np.isfinite(fn)
    print(f"\n  mean |bias-1|  ols={np.nanmean(np.abs(ols[f] - 1)):.2f}  "
          f"inv-var={np.nanmean(np.abs(iv[f] - 1)):.3f}  "
          f"folded-normal={np.nanmean(np.abs(fn[f] - 1)):.3f}")
    print("  Reading: OLS is catastrophically biased at low CNR, and")
    print("  inverse-variance weighting leaves a residual it cannot remove.")
    print("  The folded-normal scheme subtracts the additive variance bias")
    print("  (see docs/DERIVATION.md) and cuts the mean residual bias well")
    print("  below inverse-variance; it can overcorrect in the deepest noise.")

    _plot(stats, results, model_name)


def _plot(stats, results, model_name) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[matplotlib not installed -- skipping figure; "
              "install with `pip install weighting-wall[plot]`]")
        return

    cnr = np.concatenate([s.cnr for s in stats])
    p = np.concatenate([s.trunc_fraction for s in stats])
    skew = np.concatenate([s.skew for s in stats])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.3))

    ax0.scatter(cnr, p, s=20, c="tab:red", alpha=0.75)
    ax0.set_xlabel("per-frame CNR")
    ax0.set_ylabel("truncation fraction $p$  =  P(MSD $\\leq$ 0)", color="tab:red")
    ax0.set_xscale("log")
    ax0.set_ylim(-0.02, 0.55)
    ax0.axhline(0.5, ls=":", c="grey", lw=1)
    ax0.tick_params(axis="y", labelcolor="tab:red")
    ax0b = ax0.twinx()
    ax0b.scatter(cnr, skew, s=16, c="tab:blue", alpha=0.5, marker="^")
    ax0b.set_ylabel("MSD sample skewness", color="tab:blue")
    ax0b.tick_params(axis="y", labelcolor="tab:blue")
    ax0.set_title(f"(a)  The wall distorts the width estimator ({model_name})")

    for scheme, color, label in [
        ("ols", "tab:gray", "OLS (unweighted)"),
        ("inverse_variance", "tab:orange", "inverse-variance (standard)"),
        ("folded_normal", "tab:green", "folded-normal (derived)"),
    ]:
        r = results[scheme]
        ax1.plot(r.cnr0, r.mean_bias, "o-", color=color, label=label)
    ax1.axhline(1.0, ls="--", c="k", lw=1, label="unbiased")
    ax1.set_xlabel("initial CNR$_0$")
    ax1.set_ylabel("$\\langle D_{est}\\rangle\\, /\\, D_{true}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")   # OLS spans ~40x; log keeps the corrected schemes legible
    ax1.set_title("(b)  Bias in $D$: only the folded-normal correction removes it")
    ax1.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs("output", exist_ok=True)
    suffix = "" if model_name == "gaussian" else f"_{model_name}"
    out = os.path.join("output", f"poc_demo{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    main()
