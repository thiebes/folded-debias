"""Saha-controlled reaction-diffusion model for excitons and free charges.

Python port of Marc Melendez's reaction-diffusion model (corrected version
2026-06-08), the spatial extension of the zero-dimensional rate model in:

  Cutrupi, Melendez Schofield, Utrera-Melero et al., "Determining the
  Free-Carrier Fraction in 2D Perovskites using Power Dependent
  Photoluminescence", arXiv:2604.07262 (2026).

This model adds space to that point-like rate equation. A single charge-pair
population n(y, t) diffuses, and at every point the Saha equation sets the
fraction x in the free-charge state versus (1 - x) bound as excitons. Excitons
and free charges diffuse at different rates (De, Df), so the local transport is
a density-weighted mix of the two that drifts as the population decays and
spreads -- the non-Fickian behaviour the weighting correction is meant to
handle. The integrated PDE is

  dn/dt = De d2((1-x) n)/dy2 + Df d2(x n)/dy2 - nu (1-x) n - rho x^2 n^2,

with the free-charge recombination loss bimolecular in the free population
x n, hence x^2 n^2. Emission is likewise (x n)^2 for the free-charge channel.

The math mirrors Marc's corrected code. One deliberate, documented departure:
the Saha fraction uses the algebraically identical but numerically stable form
x = 2 / (sqrt(4 ntilde + 1) + 1), which avoids the 0/0 the original guards with
an explicit branch and goes smoothly to x = 1 as n -> 0. The
`free_charge_emission_power` switch defaults to the correct x^2 and is retained
only to reproduce the earlier buggy linear-x emission for comparison.

Data flow matches the sibling `marc_mn_trapping/diffusion_map.py`: `simulate`
returns the spreading emission map keyed on time, ready to feed the per-frame
width fit -> MSD -> D pipeline. Lengths in um, times in ns, rates in 1/ns,
following the sibling module and the paper's natural units; n and A share
whatever density unit the initial profile is given in.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class Parameters:
    """Physical and numerical parameters for the reaction-diffusion model.

    The diffusion constants De and Df are the genuinely new inputs: the
    published rate model has no spatial term, so they are not in the paper.
    The decay rates nu, rho, the quantum yields, and the Saha prefactor A
    derive from the paper's Tables S1 and S2 for the chosen RP layer number.
    """

    L: float = 5.0      # simulation box length (um)
    nbins: int = 250    # spatial grid points
    De: float = 0.0     # exciton diffusivity (um^2/ns)
    Df: float = 0.0     # free-charge diffusivity (um^2/ns)
    nu: float = 0.25    # exciton (first-order) recombination rate (1/ns)
    rho: float = 3e-3   # free-charge (bimolecular) recombination rate (1/ns)
    A: float = 1.0      # Saha prefactor (sets ntilde = n / A)
    qe: float = 1.0     # quantum yield for exciton emission
    qf: float = 1.0     # quantum yield for free-charge emission
    tmax: float = 25.0  # simulation time (ns)
    tsamp: float = 0.5  # sampling interval for output frames (ns)
    dt: float = 0.01    # integration time step (ns)

    # Capture the t=0 frame as well, so the MSD baseline (msd = var - var[0])
    # is the true initial width. The C starts output at the first sampling
    # step, omitting t=0; this is the one behavioural deviation from it.
    include_initial: bool = True

    # Power of x on the free-charge emission term. 2 is correct (Marc's
    # corrected code, 2026-06-08, and paper eq. 9: bimolecular emission scales
    # as (x n)^2). 1 reproduces the earlier buggy linear-x output, retained
    # only for before/after comparison.
    free_charge_emission_power: int = 2


@dataclass
class Result:
    """Output of a run: time stamps and the maps recorded at each frame."""

    times: np.ndarray          # (n_t,) frame times (ns)
    emission: np.ndarray       # (n_t, nbins) photoluminescence intensity
    population: np.ndarray     # (n_t, nbins) charge-pair population n
    free_fraction: np.ndarray  # (n_t, nbins) Saha free-charge fraction x


def x_axis(par: Parameters) -> np.ndarray:
    """Spatial pixel centres for the diffusion map."""
    dx = par.L / par.nbins
    return -0.5 * par.L + (np.arange(par.nbins) + 0.5) * dx


def sample_times(par: Parameters) -> np.ndarray:
    """Frame time stamps, matching the frames `simulate` records."""
    nsteps = int(par.tmax / par.dt)
    sampstep = int(par.tsamp / par.dt)
    ts = [0.0] if par.include_initial else []
    ts.extend(step * par.dt for step in range(1, nsteps) if step % sampstep == 0)
    return np.asarray(ts, dtype=float)


def free_fraction(ntilde: np.ndarray) -> np.ndarray:
    """Saha free-charge fraction x as a function of ntilde = n / A.

    Solving x^2 / (1 - x) = 1 / ntilde gives
    x = (sqrt(4 ntilde + 1) - 1) / (2 ntilde). The equivalent form
    x = 2 / (sqrt(4 ntilde + 1) + 1) (rationalised numerator) is used because
    it has no 0/0 at ntilde = 0 and tends to 1 as the density vanishes, which
    is the correct dilute limit (everything dissociates into free charges).
    Non-positive ntilde -- which an explicit Euler overshoot can produce -- is
    clamped to x = 1, matching the C's guard.
    """
    ntilde = np.asarray(ntilde, dtype=float)
    # Clamp the radicand before sqrt so the discarded (ntilde <= 0) branch
    # does not raise on the negative populations an Euler overshoot can yield.
    radicand = np.maximum(4.0 * ntilde + 1.0, 0.0)
    return np.where(ntilde > 0.0, 2.0 / (np.sqrt(radicand) + 1.0), 1.0)


def _emission(par: Parameters, n: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Photoluminescence intensity from population n and free fraction x.

    Excitons recombine first-order (rate nu), emitting linearly in n; free
    charges recombine bimolecularly (rate rho), emitting as (x n)^2. The free-
    charge power of x is x^2 in Marc's corrected code (and paper eq. 9); the
    power is left selectable only for comparison -- see `Parameters`.
    """
    xf = x if par.free_charge_emission_power == 1 else x ** par.free_charge_emission_power
    return n * (par.qe * (1.0 - x) * par.nu + par.qf * xf * par.rho * n) * par.dt


def _prepare_initial(par: Parameters, initial: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Set absorbing boundaries and rescale the observed profile to population.

    The experiment provides an emission profile, but emission is a nonlinear
    function of population (the n^2 free-charge term), so the initial
    population that reproduces an observed intensity is not the intensity
    itself. The C inverts a local approximation: with k = observed /
    ((qe nu + qf rho) dt), the population is n = k + sqrt(k). Edges are
    absorbing (population zero, fully dissociated).
    """
    n = np.array(initial, dtype=float).copy()
    if n.shape != (par.nbins,):
        raise ValueError(
            f"initial profile has shape {n.shape}, expected ({par.nbins},)"
        )

    denom = (par.qe * par.nu + par.qf * par.rho) * par.dt
    if denom == 0.0:
        raise ValueError(
            "qe*nu + qf*rho must be nonzero to invert emission to population; "
            "got both recombination channels at zero."
        )

    n[0] = n[-1] = 0.0  # absorbing boundaries
    x = np.ones_like(n)  # edges: x = 1

    interior = slice(1, -1)
    ni = n[interior]
    observed = ni != 0.0
    k = np.where(observed, ni / denom, 0.0)
    ni = np.where(observed, k + np.sqrt(k), 0.0)
    n[interior] = ni
    # Bins with no observed emission stay fully dissociated (x = 1).
    x[interior] = np.where(observed, free_fraction(ni / par.A), 1.0)
    return n, x


def simulate(par: Parameters, initial: np.ndarray) -> Result:
    """Integrate the reaction-diffusion PDE and record the emission map.

    `initial` is the observed emission profile (length `nbins`) used to seed
    the population. Integration is forward-Euler in time with centred second
    differences in space; the explicit scheme is stable only when
    D dt / dx^2 < 1/2, checked and warned about up front. Boundaries are
    absorbing and held fixed throughout. Returns a `Result` whose rows are the
    emission, population, and free-fraction profiles at each sampled time.
    """
    dx = par.L / par.nbins
    dx2 = dx * dx
    if par.De * par.dt / dx2 > 0.5 or par.Df * par.dt / dx2 > 0.5:
        warnings.warn(
            "Unstable forward-Euler scheme: D dt / dx^2 > 1/2. "
            "Reduce dt or coarsen the grid.",
            stacklevel=2,
        )

    n, x = _prepare_initial(par, initial)

    nsteps = int(par.tmax / par.dt)
    sampstep = int(par.tsamp / par.dt)

    times: list[float] = []
    emission: list[np.ndarray] = []
    population: list[np.ndarray] = []
    fraction: list[np.ndarray] = []

    def record(t: float) -> None:
        times.append(t)
        emission.append(_emission(par, n, x))
        population.append(n.copy())
        fraction.append(x.copy())

    if par.include_initial:
        record(0.0)

    ce = par.De * par.dt / dx2
    cf = par.Df * par.dt / dx2
    for step in range(1, nsteps):
        # Diffusion acts on the two species separately: excitons carry weight
        # (1 - x) n with constant De, free charges carry x n with Df.
        ex = (1.0 - x) * n
        fr = x * n
        lap_ex = ex[2:] + ex[:-2] - 2.0 * ex[1:-1]
        lap_fr = fr[2:] + fr[:-2] - 2.0 * fr[1:-1]
        # Free-charge recombination is bimolecular and needs BOTH carriers
        # free, so it scales as (x n)^2 -- the x^2 here, not x. (Marc's
        # corrected code, 2026-06-08; the earlier linear-x form was a bug that
        # over-recombined and collapsed the late-time signal.)
        react = n[1:-1] * (par.nu * (1.0 - x[1:-1])
                           + par.rho * n[1:-1] * x[1:-1] ** 2)
        n[1:-1] = n[1:-1] + ce * lap_ex + cf * lap_fr - par.dt * react
        # Recompute the Saha fraction from the updated interior population.
        x[1:-1] = free_fraction(n[1:-1] / par.A)

        if step % sampstep == 0:
            record(step * par.dt)

    return Result(
        times=np.asarray(times, dtype=float),
        emission=np.asarray(emission, dtype=float),
        population=np.asarray(population, dtype=float),
        free_fraction=np.asarray(fraction, dtype=float),
    )


def diffusion(par: Parameters, initial: np.ndarray) -> np.ndarray:
    """Convenience wrapper: the (n_t, nbins) emission map alone.

    Mirrors `marc_mn_trapping.diffusion_map.diffusion` so the emission map can
    be consumed the same way; use `simulate` when the population or
    free-fraction maps are also needed.
    """
    return simulate(par, initial).emission


def photoluminescence(par: Parameters, diffmap: np.ndarray) -> np.ndarray:
    """Spatially integrated PL vs time."""
    dx = par.L / par.nbins
    return np.sum(dx * diffmap, axis=1)


def variance(par: Parameters, diffmap: np.ndarray) -> np.ndarray:
    """Spatial second central moment of the PL profile at each time.

    The true (noise-free) width trajectory. NaN where a frame carries no
    signal, so it propagates rather than poisoning a mean.
    """
    x = x_axis(par)
    var = np.full(diffmap.shape[0], np.nan)
    for t in range(diffmap.shape[0]):
        pl = float(diffmap[t].sum())
        if pl <= 0:
            continue
        meanx = float(np.sum(x * diffmap[t]) / pl)
        var[t] = float(np.sum((x - meanx) ** 2 * diffmap[t]) / pl)
    return var


if __name__ == "__main__":
    # Numerical smoke test of the integrator, not physical data: a Gaussian
    # initial profile is the standard analytic check for a diffusion solver.
    # A real run must be seeded with an observed profile (from Marc / Carlos)
    # and physical De, Df. This only confirms the scheme runs and spreads.
    par = Parameters(De=0.02, Df=0.05, tsamp=1.0, dt=0.002)
    xs = x_axis(par)
    sigma0 = 0.3
    gaussian = np.exp(-xs ** 2 / (2.0 * sigma0 ** 2))
    result = simulate(par, gaussian)
    var = variance(par, result.emission)
    print(f"frames recorded: {len(result.times)}")
    print(f"first / last frame time (ns): {result.times[0]:.3f} / {result.times[-1]:.3f}")
    print(f"emission-profile variance at t=0:  {var[0]:.5f} um^2")
    print(f"emission-profile variance at t_max: {var[-1]:.5f} um^2")
    print(f"MSD over the run (var - var[0]):    {var[-1] - var[0]:.5f} um^2")
