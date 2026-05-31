"""2-D per-frame calibration of the variance-fit bias -- experimental.

The recommended D-free calibration is :func:`weighting_wall.calibrate_d_free`,
a 1-D lookup that simply runs the original calibration at ``D = 0``. It
captures the bias at ``sigma^2 = sigma_0^2``, which is the relevant region
for the low-D regime where the truncation bias bites (sigma^2(t) barely
changes across the measurement when D is small). Validated by Monte Carlo to
recover D across the full D range of interest at no accuracy cost.

The 2-D calibration in **this** module is a more general construction kept
for future work: it generates **single-frame** noisy profiles on a
(CNR, variance) grid, so the lookup is keyed on both the per-frame CNR and
the observed width. The intended use is **heterogeneous transport** where
the per-realisation sigma^2 spread is large enough that the 1-D-at-sigma_0^2
approximation breaks down.

**Status: experimental.** In the low-CNR cells (CNR < 1), the Monte Carlo
noise per cell dominates the per-cell bias estimate and the resulting lookup
is too noisy to use without substantial additional sample budget (n_runs
>~ 5000 per cell, plus post-hoc smoothing). The fixed-point inversion in
:meth:`TruncationModel.var_bias` compensates for the lookup-at-biased-sigma^2
issue but cannot rescue the underlying MC noise. See
``scripts/validate_d_free.py`` for a failing validation; for working D-free
calibration see ``scripts/validate_d_zero.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import GaussianModel
from .simulate import Experiment


@dataclass
class CalibrationGrid:
    """A 2-D lookup of ``var_bias`` and ``rel_variance`` over (CNR, variance).

    Built by :func:`calibrate_2d`. ``var_bias[i, j]`` is the empirical
    ``E[sigma_hat^2] - sigma^2_true`` from ``n_runs`` fits at
    ``cnr_grid[i]`` and ``variance_grid[j]``. ``rel_variance[i, j]`` is the
    sample variance of those fits -- the per-frame ``Var(sigma_hat^2)``.

    The grids are 1-D and ascending; bilinear interpolation between cells is
    handled by :class:`~weighting_wall.truncation.TruncationModel`.
    """

    cnr_grid: np.ndarray         # (n_cnr,), ascending
    variance_grid: np.ndarray    # (n_var,), ascending
    var_bias: np.ndarray         # (n_cnr, n_var)
    rel_variance: np.ndarray     # (n_cnr, n_var)
    n_runs: int
    model_name: str


def _single_frame_profile(model, pixel_x: np.ndarray, variance: float,
                          amplitude: float) -> np.ndarray:
    """A noise-free single-frame profile on the given pixel grid.

    Constructs a one-frame :class:`Experiment` whose `D=0`, `t_max=0`, and
    `sigma0 = sqrt(variance)`, so ``true_variance[0] = variance`` exactly, then
    asks the profile model for its noise-free series. The pixel grid is set by
    inverting :attr:`Experiment.x` so that the experiment's x array matches
    ``pixel_x``. ``pixel_x`` is assumed symmetric about zero.
    """
    sigma0 = float(np.sqrt(variance))
    x_max = float(np.max(np.abs(pixel_x)))
    half_window = x_max / sigma0
    exp = Experiment(
        sigma0=sigma0, D=0.0, amplitude0=float(amplitude), mean=0.0,
        half_window=half_window, n_pixels=int(pixel_x.size),
        n_frames=1, t_max=0.0,
    )
    return model.noise_free_series(exp)[0]


def calibrate_2d(model=None, *, pixel_x: np.ndarray | None = None,
                 variance_grid: np.ndarray | None = None,
                 cnr_grid: np.ndarray | None = None,
                 amplitude: float = 1.0, n_runs: int = 300,
                 seed: int = 0,
                 verbose: bool = False) -> CalibrationGrid:
    """Build the 2-D (CNR, variance) calibration of the fit bias.

    Parameters
    ----------
    model : profile model (see :mod:`weighting_wall.models`). Defaults to
        :class:`~weighting_wall.models.GaussianModel`.
    pixel_x : the spatial axis on which the fit will run. Must match the
        experimental grid for the calibration to apply. Defaults to the
        :class:`Experiment` standard grid.
    variance_grid : the true-variance values to span. Defaults to a generous
        range from ``0.6`` to ``3.5`` in units of ``sigma_0^2``.
    cnr_grid : the per-frame CNR values to span. Defaults to a log-spaced grid
        from ``0.5`` to ``120``.
    amplitude : the noise-free peak amplitude used in the calibration. Only its
        ratio to ``noise_sigma`` (= ``CNR``) affects the fit bias, so the choice
        is conventional; left at ``1.0`` to match the standard experiment.
    n_runs : Monte-Carlo runs per grid cell.
    seed : root random seed; each cell gets an independent stream.
    verbose : print per-cell progress.

    Returns
    -------
    :class:`CalibrationGrid`
        The 2-D ``var_bias`` and ``rel_variance`` lookups.
    """
    if model is None:
        model = GaussianModel()
    if pixel_x is None:
        pixel_x = Experiment().x
    if variance_grid is None:
        variance_grid = np.linspace(0.6, 3.5, 12)
    if cnr_grid is None:
        # Span deep noise: per-frame CNR drops below 1 in the low-CNR_0 regime
        # (CNR_0 = 5, t_max = 3*tau -> per-frame CNR -> ~0.25 by the last frame).
        # The grid extends down to 0.15 so those frames are not clipped.
        cnr_grid = np.logspace(np.log10(0.15), np.log10(120.0), 22)

    pixel_x = np.asarray(pixel_x, float)
    cnr_grid = np.asarray(cnr_grid, float)
    variance_grid = np.asarray(variance_grid, float)

    n_cnr = cnr_grid.size
    n_var = variance_grid.size
    var_bias = np.full((n_cnr, n_var), np.nan)
    rel_var = np.full((n_cnr, n_var), np.nan)

    # Independent RNG stream per cell, so the calibration is reproducible
    # whatever order the loops run in (or if they're later parallelised).
    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_cnr * n_var)

    for i, cnr in enumerate(cnr_grid):
        if verbose:
            print(f"  CNR {cnr:7.3f} ({i + 1}/{n_cnr})")
        noise_sigma = float(amplitude) / float(cnr)
        for j, variance in enumerate(variance_grid):
            clean = _single_frame_profile(model, pixel_x, variance, amplitude)
            rng = np.random.default_rng(child_seeds[i * n_var + j])
            fits = np.empty(n_runs)
            for r in range(n_runs):
                noise = rng.normal(0.0, noise_sigma, size=clean.shape)
                ff = model.fit_variance(pixel_x, clean + noise)
                fits[r] = ff.variance
            good = np.isfinite(fits)
            if good.sum() > 1:
                var_bias[i, j] = float(fits[good].mean() - variance)
                rel_var[i, j] = float(np.var(fits[good], ddof=1))

    return CalibrationGrid(cnr_grid=cnr_grid, variance_grid=variance_grid,
                           var_bias=var_bias, rel_variance=rel_var,
                           n_runs=int(n_runs), model_name=model.name)
