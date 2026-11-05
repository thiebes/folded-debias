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

    import numpy as np
    from folded_debias import Experiment, calibrate_d_free, evaluate
    from folded_debias.fit import fit_D
    from folded_debias.weights import msd_and_weights

    cnr0 = np.array([5, 8, 13, 21, 34, 55, 90])
    trunc = calibrate_d_free(cnr0)           # D-free lookup var_bias(CNR)

    # Monte Carlo bias curves <D_est>/D_ref for ols, inverse_variance,
    # folded_normal, and mas
    results = evaluate(Experiment(D=0.02), cnr0, trunc)

    # A measurement: fitted widths in units of sigma_0^2 and per-frame CNR
    msd, w = msd_and_weights("folded_normal", variance, cnr, trunc)
    w = np.where(cnr >= 1.3, w, 0.0)         # drop frames below the floor
    D = fit_D(times, msd, w).D

``folded_normal`` is the recommended scheme and :func:`calibrate_d_free` the
recommended calibration. The README states the conditions for real data: a
calibration that matches the data's geometry and units, and the per-frame CNR
floor.
"""
from .simulate import (AnomalousExperiment, Experiment, MultiExpExperiment,
                       cnr_per_frame, noisy_series)
from .models import (GaussianModel, GeneralizedGaussianModel, MODELS,
                     MomentModel, UnboundedGaussianModel, VoigtModel,
                     model_from_spec, spec_tag)
from .montecarlo import Ensemble, run_ensemble
from .truncation import FrameStats, TruncationModel, frame_stats
from .weights import (SCHEMES, calibration_D_se, corrected_variance,
                      msd_and_weights)
from .validate import (FrameDiagnostic, SchemeResult, calibrate,
                       calibrate_d_free, calibrate_fisher, evaluate,
                       per_frame_diagnostic, sweep_stats)
from .calibration import (CalibrationCells, CalibrationGrid, calibrate_cells,
                          calibrate_cnr_sigma2)
from .fisher import LeadingOrderTerms, leading_order_terms

__version__ = "0.4.0"

__all__ = [
    "Experiment", "AnomalousExperiment", "MultiExpExperiment",
    "cnr_per_frame", "noisy_series",
    "GaussianModel", "VoigtModel", "GeneralizedGaussianModel", "MomentModel",
    "UnboundedGaussianModel", "MODELS", "model_from_spec", "spec_tag",
    "Ensemble", "run_ensemble",
    "FrameStats", "TruncationModel", "frame_stats",
    "SCHEMES", "msd_and_weights", "corrected_variance", "calibration_D_se",
    "SchemeResult", "calibrate", "calibrate_d_free", "calibrate_fisher",
    "evaluate", "sweep_stats",
    "FrameDiagnostic", "per_frame_diagnostic",
    "CalibrationGrid", "calibrate_cnr_sigma2",
    "CalibrationCells", "calibrate_cells",
    "LeadingOrderTerms", "leading_order_terms",
]
