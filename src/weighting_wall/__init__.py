"""weighting-wall: CNR-aware weighting for MSD diffusion fits.

Fitted Gaussian widths cannot be negative, so the sampling distribution of a
fitted width is left-truncated at zero and right-skewed -- worst at low CNR.
That asymmetry leaks through the MSD fit and inflates the extracted diffusion
coefficient D.  Standard inverse-variance weighting assumes a symmetric error
distribution and so cannot remove the bias.

This package derives and validates a weighting scheme that takes the per-frame
contrast-to-noise ratio (from the ``fft-cnr`` package) as input and corrects
for the truncation as well as the heteroscedasticity.

Typical use::

    from weighting_wall import Experiment, calibrate, evaluate
    import numpy as np

    exp = Experiment(D=0.02)                 # short-L_D nanocrystal-solid regime
    cnr0 = np.array([2, 3, 5, 8, 12, 20, 40])
    model = calibrate(exp, cnr0)
    results = evaluate(exp, cnr0, model)     # bias curves per weighting scheme
"""
from .simulate import Experiment, cnr_per_frame, noisy_series
from .montecarlo import Ensemble, run_ensemble
from .truncation import FrameStats, TruncationModel, frame_stats
from .weights import SCHEMES, make_weights, truncation_factor
from .validate import SchemeResult, calibrate, evaluate, sweep_stats

__version__ = "0.1.0"

__all__ = [
    "Experiment", "cnr_per_frame", "noisy_series",
    "Ensemble", "run_ensemble",
    "FrameStats", "TruncationModel", "frame_stats",
    "SCHEMES", "make_weights", "truncation_factor",
    "SchemeResult", "calibrate", "evaluate", "sweep_stats",
]
