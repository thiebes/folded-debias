"""folded-debias: correcting the truncation-at-zero bias in MSD diffusion fits.

A fitted width cannot be negative, so at low CNR the sampling distribution of
the fitted variance is right-skewed and biased high. That bias leaks through
the MSD fit and inflates the extracted diffusion coefficient D. Standard
inverse-variance weighting corrects the unequal noise across frames but cannot
remove a mean bias from a slope fit.

This package derives the correction from a folded-normal model of the fitted
width (see ``docs/DERIVATION.md``): the fitted variance carries an additive
bias equal to the estimation-noise variance, so the fix is to subtract that
bias per frame, then weight by inverse variance. The per-frame CNR (from the
``fft-cnr`` package) is the input.

Typical use::

    from folded_debias import Experiment, calibrate, evaluate
    import numpy as np

    exp = Experiment(D=0.02)                 # short-L_D nanocrystal-solid regime
    cnr0 = np.array([5, 8, 13, 21, 34, 55, 90])
    model = calibrate(exp, cnr0)
    results = evaluate(exp, cnr0, model)     # bias curves: ols / inverse_variance / folded_normal
"""
from .simulate import (AnomalousExperiment, Experiment, MultiExpExperiment,
                       cnr_per_frame, noisy_series)
from .models import (GaussianModel, GeneralizedGaussianModel, MODELS,
                     MomentModel, UnboundedGaussianModel, VoigtModel)
from .montecarlo import Ensemble, run_ensemble
from .truncation import FrameStats, TruncationModel, frame_stats
from .weights import SCHEMES, corrected_variance, msd_and_weights
from .validate import (FrameDiagnostic, SchemeResult, calibrate,
                       calibrate_d_free, evaluate, per_frame_diagnostic,
                       sweep_stats)
from .calibration import CalibrationGrid, calibrate_cnr_sigma2

__version__ = "0.3.0"

__all__ = [
    "Experiment", "AnomalousExperiment", "MultiExpExperiment",
    "cnr_per_frame", "noisy_series",
    "GaussianModel", "VoigtModel", "GeneralizedGaussianModel", "MomentModel",
    "UnboundedGaussianModel", "MODELS",
    "Ensemble", "run_ensemble",
    "FrameStats", "TruncationModel", "frame_stats",
    "SCHEMES", "msd_and_weights", "corrected_variance",
    "SchemeResult", "calibrate", "calibrate_d_free", "evaluate", "sweep_stats",
    "FrameDiagnostic", "per_frame_diagnostic",
    "CalibrationGrid", "calibrate_cnr_sigma2",
]
