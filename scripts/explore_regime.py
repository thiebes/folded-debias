"""Parameter-space exploration: where is the weighting improvement clearest?

Sweeps the diffusion coefficient D and the initial CNR_0 and tabulates the
bias <D_est>/D_true for OLS, inverse-variance, and the folded-normal scheme.
The goal is to locate the regime -- the project hypothesis is small D and
small CNR -- where inverse-variance weighting is *itself* most biased, so that
the improvement offered by the folded-normal correction is clearly resolved.

Run from the repository root::

    python scripts/explore_regime.py

Prints, per D, a CNR_0 table and the (inverse-variance - folded-normal) gap,
then reports the cell with the largest inverse-variance bias and the cell with
largest weighting improvement.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from weighting_wall import Experiment, evaluate, sweep_stats  # noqa: E402
from weighting_wall.truncation import TruncationModel  # noqa: E402

# Small-D / small-CNR corner, swept finely.  L_D/FWHM_0 = sqrt(2 D tau)/2.3548.
D_GRID = np.array([0.004, 0.008, 0.015, 0.03, 0.06])
CNR0_GRID = np.array([5.0, 7.0, 10.0, 15.0, 22.0, 34.0])
N_RUNS = 250
TAU = 1.0


def main() -> None:
    print("weighting-wall parameter-space exploration")
    print(f"  D grid:    {D_GRID}")
    print(f"  CNR0 grid: {CNR0_GRID}")
    print(f"  {N_RUNS} Monte Carlo runs per cell\n")

    iv_grid = np.full((D_GRID.size, CNR0_GRID.size), np.nan)
    fn_grid = np.full((D_GRID.size, CNR0_GRID.size), np.nan)

    for di, D in enumerate(D_GRID):
        exp = Experiment(sigma0=1.0, D=float(D), tau=TAU, n_frames=12, t_max=3.0)
        ld_fwhm = np.sqrt(2 * D * TAU) / 2.3548
        stats = sweep_stats(exp, CNR0_GRID, n_runs=N_RUNS, seed=1 + di)
        model = TruncationModel.from_stats(stats)
        res = evaluate(exp, CNR0_GRID, model, n_runs=N_RUNS, seed=20000 + di)
        iv_grid[di] = res["inverse_variance"].mean_bias
        fn_grid[di] = res["folded_normal"].mean_bias

        print(f"--- D={D:.3f}  (L_D/FWHM_0={ld_fwhm:.3f}) ---")
        print("  CNR0   inv-var  folded-normal   gap(iv-fn)")
        for ci, c0 in enumerate(CNR0_GRID):
            iv, fn = iv_grid[di, ci], fn_grid[di, ci]
            print(f"  {c0:5.0f}  {iv:7.3f}  {fn:11.3f}   {iv - fn:+8.3f}")
        print()

    # Where is inverse-variance weighting itself most biased?
    iv_excess = np.abs(iv_grid - 1.0)
    di, ci = np.unravel_index(np.nanargmax(iv_excess), iv_excess.shape)
    print(f"Largest inverse-variance bias: D={D_GRID[di]:.3f}, "
          f"CNR0={CNR0_GRID[ci]:.0f}  ->  <D>/D_true = {iv_grid[di, ci]:.3f}")

    # Where does the folded-normal correction help most?
    gain = np.abs(iv_grid - 1.0) - np.abs(fn_grid - 1.0)
    di, ci = np.unravel_index(np.nanargmax(gain), gain.shape)
    print(f"Largest correction improvement: D={D_GRID[di]:.3f}, "
          f"CNR0={CNR0_GRID[ci]:.0f}  ->  inv-var {iv_grid[di, ci]:.3f} "
          f"vs folded-normal {fn_grid[di, ci]:.3f}")
    print("\nUse this cell's (D, CNR0) to set the demo so the improvement is "
          "clearly resolved.")


if __name__ == "__main__":
    main()
