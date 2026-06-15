"""Load the Solari et al. disordered-NC-solid diffusion maps (read-only from P:).

Source: Solari et al., "Exciton Transport in Disordered Perovskite Nanocrystal
Solids" (CH3NH3PbBr3 NC films, C8/C12/C16 ligands). Marc Melendez's numerical
simulations behind Fig. 5 live on the pshare; we read them, never write there.

Each `diffmap_<tag>.dat` is 20 frames of 250x250 exciton-concentration maps over
a 1500x1500 nm^2 area (6 nm/pixel), 1 ns/frame. The maps track exciton position
only -- there is NO recombination, and the concentration is per-frame
normalized. To model a real transient-PL measurement we impose the population
decay envelope `N0 * exp(-t/tau)` (the paper's thin-film exciton lifetime is
tau ~ 6-8 ns overall, with per-sample TPLM tau_1/e ~ 6 / 14 / 18 ns for
C16 / C12 / C8). The decay is a per-frame scalar, so it sets the amplitude --
hence the per-frame contrast-to-noise -- without changing the spatial variance.

The pipeline fits a 1-D width, so we project the 2-D map to 1-D by summing over
y. The paper itself fits each time slice to a Voigt profile; the spatial second
moment we report here is the transport ground truth (sigma0 ~ 200 nm spot,
MSD growth ~2000-5000 nm^2 over the run, L_D/FWHM ~ 0.14 -- the headline
nanocrystal-solid regime).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

PSHARE = Path("P:/Marc/shared_data/disordered_perovskite_nanocrystals_solids")
L_UM = 1.5            # box length (um); 1500 nm
NBINS = 250
NFRAMES = 20
DT_NS = 1.0
CASES = ("small", "medium", "large")

# Representative thin-film exciton lifetime (ns), from Solari et al. A single
# value validates the mechanics; per-sample tau (C16/C12/C8 ~ 6/14/18 ns) is the
# refinement once small/medium/large are mapped to ligands.
DEFAULT_TAU_NS = 7.0


@dataclass
class NCCase:
    """One disordered-NC-solid map: geometry, decay-weighted 1-D profiles, truth."""
    name: str
    x: np.ndarray          # spatial axis (centred), um
    t: np.ndarray          # time axis, ns
    profiles: np.ndarray   # (n_frames, n_x) 1-D profiles after the decay envelope
    var_true: np.ndarray   # (n_frames,) spatial variance, um^2
    tau_ns: float


def _denoise_row(x: np.ndarray, row: np.ndarray) -> tuple[np.ndarray, float]:
    """Smooth Gaussian density underlying a finite-walker histogram row.

    Marc's maps are histograms of ~50,000 discrete excitons, so each bin carries
    Poisson (shot-like) sampling noise from the finite walker count. That scatter
    is neither the smooth physical density a continuum measurement would image nor
    the measurement noise we control -- and read through ``max`` it biases the
    per-frame contrast-to-noise high (the peak appears to *rise* with spreading).
    Fitting a Gaussian (centre, width) to the noiseless row recovers the smooth
    density and removes the sampling noise; the reconstruction keeps the row's
    integral, so the population stays conserved and the peak falls as 1/sigma with
    spreading, as diffusion requires. Returns ``(smooth_row, sigma2)``.
    """
    tot = float(row.sum())
    p = row / tot
    c0 = float((p * x).sum())
    s0 = float(np.sqrt(((x - c0) ** 2 * p).sum()))

    def model(xx, c, s):
        e = np.exp(-0.5 * ((xx - c) / s) ** 2)
        return e / e.sum()

    try:
        (c, s), _ = curve_fit(model, x, p, p0=[c0, s0], maxfev=20000)
    except (RuntimeError, ValueError):
        c, s = c0, s0
    return model(x, c, s) * tot, float(s) ** 2


def load_case(tag: str, tau_ns: float = DEFAULT_TAU_NS, n0: float = 1.0,
              denoise: bool = False) -> NCCase:
    """Load one map by tag (``small`` / ``medium`` / ``large``), with decay applied.

    With ``denoise=True`` each frame's projection is replaced by its smooth
    Gaussian reconstruction (see :func:`_denoise_row`) before the decay envelope,
    so the clean signal is the continuum density and the only noise in the Monte
    Carlo is the measurement noise added downstream. ``var_true`` is then the
    de-noised Gaussian width; otherwise it is the raw projection's second moment.
    """
    cube = np.loadtxt(PSHARE / f"diffmap_{tag}.dat").reshape(NFRAMES, NBINS, NBINS)
    x = np.linspace(-L_UM / 2.0, L_UM / 2.0, NBINS)
    t = np.arange(NFRAMES) * DT_NS
    prof = cube.sum(axis=1)                              # project y -> (t, x)
    if denoise:
        smooth = np.empty_like(prof)
        var = np.empty(NFRAMES)
        for f in range(NFRAMES):
            smooth[f], var[f] = _denoise_row(x, prof[f])
        prof = smooth
    prof = prof * (n0 * np.exp(-t / tau_ns))[:, None]    # impose decay envelope
    if not denoise:
        tot = prof.sum(axis=1)
        xbar = (prof * x).sum(1) / tot
        var = (prof * (x[None, :] - xbar[:, None]) ** 2).sum(1) / tot
    return NCCase(name=tag, x=x, t=t, profiles=prof, var_true=var, tau_ns=tau_ns)


def load_all(tau_ns: float = DEFAULT_TAU_NS, denoise: bool = False) -> list[NCCase]:
    """Load every map present on the pshare, in small/medium/large order."""
    return [load_case(tag, tau_ns=tau_ns, denoise=denoise) for tag in CASES
            if (PSHARE / f"diffmap_{tag}.dat").exists()]
