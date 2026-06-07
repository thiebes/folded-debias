"""Reaction-diffusion model for excitons in Mn-doped 2D perovskites.

Refactor of Marc Melendez's `diffusion_map.py` (provided 2026-05-30) to use
`par.X` everywhere instead of free globals. Mathematical content is unchanged;
the original is the implementation of the model derived in the SI of:

  Magdaleno, Kshirsagar, Melendez et al., "Role of Exciton Diffusion in the
  Efficiency of Mn Dopant Emission in Two-Dimensional Perovskites",
  ACS Nanosci. Au 5, 29-36 (2025).

Three populations: free excitons (Gaussian diffusion + recombination +
trapping at two channels), Mn-trapped (channel 1), defect-trapped (channel 2).
Free-exciton spatial profile is the analytic Gaussian solution at every time
slice; trapped populations are integrated with explicit Euler and inherit the
local free-exciton density. Optional pulsed-mode adds the steady-state
trapped-population background that accumulates from repeated laser pulses.

The `diffusion(par)` function returns a (n_t, n_bins) array of PL intensities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    """Physical and numerical parameters for the reaction-diffusion model.

    Lengths in um, times in ns, rates in 1/ns. Defaults match the 0.51% Mn
    sample in Magdaleno et al. 2025 (Table S2; main-text Section 5.2).
    """

    L: float = 5.0          # simulation box length (um)
    nbins: int = 250        # spatial grid points
    N0: float = 1.0         # initial exciton population
    sigma0: float = 0.1     # initial spatial std (um)
    D0: float = 0.025       # free-exciton diffusivity (um^2/ns)
    lambda1: float = 0.35   # trapping rate at Mn sites (1/ns)
    lambda2: float = 0.02   # trapping rate at defect sites (1/ns)
    nu0: float = 0.47       # free-exciton recombination rate (1/ns)
    nu1: float = 0.008      # Mn-trapped recombination rate (1/ns)
    nu2: float = 0.1        # defect-trapped recombination rate (1/ns)
    q0: float = 0.25        # detection-weighted QY for free excitons
    q1: float = 0.37 * 0.74 # detection-weighted QY for Mn traps (q_Mn * y_Mn)
    q2: float = 1.0         # detection-weighted QY for defect traps
    tmax: float = 25.0      # simulation time (ns)
    tp: float = 25.0        # laser pulse period (ns)
    dt: float = 0.1         # numerical time step (ns)
    pulsed: bool = True     # include steady-state pulsed background


def x_axis(par: Parameters) -> np.ndarray:
    """Spatial pixel centres for the diffusion map."""
    dx = par.L / par.nbins
    return -0.5 * par.L + (np.arange(par.nbins) + 0.5) * dx


def times(par: Parameters) -> np.ndarray:
    """Time stamps for each row of the diffusion map."""
    nt = int(par.tmax / par.dt)
    return np.arange(nt) * par.dt


def diffusion(par: Parameters) -> np.ndarray:
    """Build the (n_t, n_bins) diffusion map of total PL intensity.

    Free excitons follow the analytic Gaussian solution at each time;
    trapped populations are integrated with explicit Euler. With
    `par.pulsed = True`, the steady-state pulsed background of trapped
    populations is added.
    """
    dx = par.L / par.nbins
    nt = int(par.tmax / par.dt)
    diffmap = np.zeros((nt, par.nbins))
    n1 = np.zeros(par.nbins)
    n2 = np.zeros(par.nbins)
    for j in range(nt):
        t = j * par.dt
        var_t = par.sigma0 ** 2 + 2 * par.D0 * t
        amp = 1.0 / np.sqrt(2 * np.pi * var_t)
        decay = np.exp(-(par.nu0 + par.lambda1 + par.lambda2) * t)
        for i in range(par.nbins):
            x = -0.5 * par.L + (i + 0.5) * dx
            n0 = par.N0 * amp * decay * np.exp(-x * x / (2 * var_t))
            n1[i] = n1[i] + par.dt * (par.lambda1 * n0 - par.nu1 * n1[i])
            n2[i] = n2[i] + par.dt * (par.lambda2 * n0 - par.nu2 * n2[i])
            diffmap[j, i] = (par.q0 * par.nu0 * n0
                             + par.q1 * par.nu1 * n1[i]
                             + par.q2 * par.nu2 * n2[i])

    if par.pulsed:
        n1b = n1 * np.exp(-par.nu1 * (par.tp - par.tmax)) / (
            1 - np.exp(-par.nu1 * par.tp))
        n2b = n2 * np.exp(-par.nu2 * (par.tp - par.tmax)) / (
            1 - np.exp(-par.nu2 * par.tp))
        for j in range(nt):
            t = j * par.dt
            diffmap[j] += (par.q1 * par.nu1 * n1b * np.exp(-par.nu1 * t)
                           + par.q2 * par.nu2 * n2b * np.exp(-par.nu2 * t))

    return diffmap


def photoluminescence(par: Parameters, diffmap: np.ndarray) -> np.ndarray:
    """Spatially integrated PL vs time."""
    dx = par.L / par.nbins
    return np.sum(dx * diffmap, axis=1)


def variance(par: Parameters, diffmap: np.ndarray) -> np.ndarray:
    """Spatial second central moment of the PL profile at each time."""
    dx = par.L / par.nbins
    x = x_axis(par)
    var = np.zeros(diffmap.shape[0])
    for t in range(diffmap.shape[0]):
        pl = float(diffmap[t].sum())
        if pl <= 0:
            var[t] = np.nan
            continue
        meanx = float(np.sum(x * diffmap[t]) / pl)
        var[t] = float(np.sum((x - meanx) ** 2 * diffmap[t]) / pl)
    return var
