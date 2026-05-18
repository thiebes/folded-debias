"""Per-frame CNR estimation -- the input the whole weighting scheme rests on.

For Monte Carlo validation the per-frame CNR is known exactly (it is set by the
simulation; see :func:`weighting_wall.simulate.cnr_per_frame`).  For *real*
measurements it must be estimated from each spatial profile, with no repeated
frames and no separate background region available.  That is exactly what the
``fft-cnr`` package does: it separates slowly varying signal from rapid noise
in the spatial frequency domain and returns a CNR estimate per 1-D profile.

This module wraps ``fft-cnr`` behind a single function and degrades gracefully
when the package is not installed, so the simulation-only POC has no hard
dependency on it.
"""
from __future__ import annotations

import numpy as np

try:  # optional dependency -- only needed for real experimental data
    from fft_cnr import fft_cnr as _fft_cnr
    HAVE_FFT_CNR = True
except ImportError:  # pragma: no cover - exercised only without the package
    _fft_cnr = None
    HAVE_FFT_CNR = False


def per_frame_cnr(profiles: np.ndarray) -> np.ndarray:
    """Estimate the CNR of every row of ``profiles`` with ``fft-cnr``.

    Parameters
    ----------
    profiles : array, shape (n_frames, n_pixels)
        Stack of 1-D spatial profiles, one per time point.

    Returns
    -------
    cnr : array, shape (n_frames,)

    Raises
    ------
    ImportError
        If the optional ``fft-cnr`` dependency is not installed.  Install it
        with ``pip install weighting-wall[cnr]``.
    """
    if not HAVE_FFT_CNR:
        raise ImportError(
            "fft-cnr is required for CNR estimation from real data; "
            "install it with `pip install weighting-wall[cnr]`."
        )
    profiles = np.atleast_2d(np.asarray(profiles, float))
    return np.array([float(_fft_cnr(row).cnr) for row in profiles])
