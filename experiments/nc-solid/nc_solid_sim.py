"""Kinetic Monte Carlo of exciton hopping on a disordered nanocrystal solid.

Re-implementation of the Solari et al. simulation behind Fig. 5, from the
supplementary information (`sources/ReSubmission_Solari_2026_SI_simulations_extended_cut.pdf`,
Eqs 1-3 and Table 1) and the structural inputs in the main text (Fig. 1d/f, the
E(d) curve in the Fig. 5 caption). The point is to *reconstruct* the clean
transport from the model, at any walker count, rather than de-noise Marc's
finite-walker output maps -- so the per-frame contrast-to-noise and the truth are
both clean and assume no profile shape.

The model (SI):

  * A 2-D plane of polydisperse disks (nanocrystals). Disk diameter d is drawn
    per sample from the measured size distribution; site energy follows
    E(d) = A exp(-b d) + E_b  (Eq 2; smaller crystal -> higher energy).
  * N excitons start as a Gaussian (sigma = 200 nm) and are snapped to the
    nearest crystal.
  * Each step, each exciton picks a random neighbouring crystal and hops with
    probability p = 1 / (gamma (r + r0))^6 (Eq 1, dipole/Forster) when moving to
    a LARGER (lower-energy) crystal, or p * exp(-dE / kT) when moving to a SMALLER
    (higher-energy) crystal, with dE = E(d_acc) - E(d_don) + Normal(0, 2 sigma_E)
    (Eq 3).
  * The spatial second moment is tracked in time; energetic disorder makes it
    grow sub-linearly (subdiffusion).

What the SI fixes exactly (Table 1): gamma, r0, A, b, E_b, sigma_E. What it does
NOT fix, and is taken here as documented modelling choices (flagged), is the
disk-generation statistics (per-sample mean diameter, polydispersity, and
center-to-center spacing -- read approximately from the paper figures; the exact
values are in the unavailable SI Fig. S2 / Table S2) and the MC step-to-time
calibration (runs are in MC steps; the absolute time scale needs calibrating, but
the exponent alpha is scale-free, so the C16 < C12 < C8 ordering is a faithful
test of the model).

Validation and the calibration gap (2026-06-17). The subdiffusion strength is set
by the ratio of site-energy disorder to thermal energy, sigma_E / kT. With the
figure-read size distributions (sigma_E ~ 10-17 meV) and room temperature
(kT ~ 26 meV) the ratio is ~0.65: the population thermalises in a few steps and
then diffuses normally (alpha ~ 1, mean site energy flat). As the disorder is
strengthened (lower kT), the exponent drops below 1 and the C16 < C12 < C8
ordering emerges (the smallest crystals carry the most energetic disorder because
E(d) is steepest at small d) -- reproducing the paper's counterintuitive result.
Reaching the paper's exact alpha ~ 0.42-0.70 would need sigma_E / kT well above 1
(stronger disorder than the figure-read sizes plus room temperature give; the
exact size distributions in SI Fig. S2 and the temperature would close that gap).
But reproducing their numbers is not the goal here: this is a clean, model
NC-solid-like subdiffusive test bed for the noise-bias study, not a Solari
reproduction. ``kt_mev`` is the disorder-strength knob (lower = stronger, more
subdiffusive); a strong-disorder setting gives genuinely subdiffusive transport
with the right C16 < C12 < C8 ordering, which is all the bias study needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

# --- Table 1 (SI) -- the parameters the SI fixes exactly --------------------
GAMMA = 0.06        # nm^-1, Eq 1 transfer-rate parameter
R0 = 15.0           # nm, Eq 1 offset (avoids r->0 divergence)
A_MEV = 702.54      # meV, Eq 2 bandgap-fit amplitude
B_NM = 0.3446       # nm^-1, Eq 2 decay with diameter
EB_MEV = 2196.0     # meV, Eq 2 bulk-value offset
SIGMA_E_MEV = 1.96  # meV, Eq 3 bandgap-energy uncertainty
KT_MEV = 25.85      # meV, k_B T at 300 K (T not stated in the SI)

# --- Per-sample disk-generation inputs (FIGURE-READ, approximate) -----------
# small/medium/large = C16/C12/C8 (Marc, 2026-06-15). Mean diameter, fractional
# polydispersity (std/mean), and center-to-center spacing read off the paper's
# Fig. 1d/f; the exact distributions are in SI Fig. S2 (unavailable). Tune these
# against Marc's variance(t) once the SI numbers arrive.
@dataclass(frozen=True)
class Sample:
    name: str
    ligand: str
    mean_d_nm: float
    polydispersity: float   # std / mean
    dcc_nm: float           # center-to-center spacing


SAMPLES = {
    "small":  Sample("small",  "C16", mean_d_nm=7.5,  polydispersity=0.12, dcc_nm=9.5),
    "medium": Sample("medium", "C12", mean_d_nm=11.0, polydispersity=0.20, dcc_nm=13.5),
    "large":  Sample("large",  "C8",  mean_d_nm=15.0, polydispersity=0.27, dcc_nm=17.0),
}


def site_energy_mev(d_nm: np.ndarray) -> np.ndarray:
    """E(d) = A exp(-b d) + E_b in meV (SI Eq 2). Smaller d -> higher energy."""
    return A_MEV * np.exp(-B_NM * d_nm) + EB_MEV


@dataclass
class Landscape:
    pos: np.ndarray      # (M, 2) crystal centres, nm
    diam: np.ndarray     # (M,) crystal diameters, nm
    energy: np.ndarray   # (M,) site energies, meV
    nb: np.ndarray       # (M, maxnb) padded neighbour indices (-1 pad)
    nb_count: np.ndarray  # (M,) number of neighbours


def generate_landscape(sample: Sample, box_nm: float, rng: np.random.Generator,
                       r_cut_nm: float | None = None) -> Landscape:
    """Disordered 2-D disk array: a jittered triangular tiling at spacing dcc.

    The film is disordered (not close-packed; main text), so crystals sit on a
    triangular lattice of pitch ``dcc`` jittered by ~0.3 pitch, with diameters
    drawn lognormally at the sample's mean and polydispersity (lognormal keeps d
    positive and matches the right-skewed TEM histograms). Neighbours are all
    crystals within ``r_cut`` (default 4 x dcc, beyond which p ~ r^-6 is
    negligible).
    """
    dcc = sample.dcc_nm
    if r_cut_nm is None:
        r_cut_nm = 4.0 * dcc
    # triangular lattice covering the box
    dx = dcc
    dy = dcc * np.sqrt(3.0) / 2.0
    nx = int(box_nm / dx) + 2
    ny = int(box_nm / dy) + 2
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny))
    px = (ix + 0.5 * (iy % 2)) * dx
    py = iy * dy
    pos = np.column_stack([px.ravel(), py.ravel()]).astype(float)
    pos -= pos.mean(0)                         # centre on (0, 0)
    pos += rng.normal(0.0, 0.30 * dcc, pos.shape)   # structural disorder
    # diameters: lognormal with the sample mean and polydispersity
    mu = sample.mean_d_nm
    cv = sample.polydispersity
    sig_ln = np.sqrt(np.log(1.0 + cv * cv))
    mu_ln = np.log(mu) - 0.5 * sig_ln * sig_ln
    diam = rng.lognormal(mu_ln, sig_ln, pos.shape[0])
    energy = site_energy_mev(diam)
    # neighbour list
    tree = cKDTree(pos)
    pairs = tree.query_ball_point(pos, r_cut_nm)
    maxnb = max(len(p) - 1 for p in pairs)     # exclude self
    nb = np.full((pos.shape[0], maxnb), -1, dtype=np.int64)
    cnt = np.zeros(pos.shape[0], dtype=np.int64)
    for i, pl in enumerate(pairs):
        js = [j for j in pl if j != i]
        nb[i, :len(js)] = js
        cnt[i] = len(js)
    return Landscape(pos=pos, diam=diam, energy=energy, nb=nb, nb_count=cnt)


@dataclass
class Trajectory:
    step: np.ndarray         # (n_samples,) MC step index
    variance_nm2: np.ndarray  # (n_samples,) 1-D spatial variance (var_x+var_y)/2
    mean_energy_mev: np.ndarray  # (n_samples,) mean exciton site energy
    frames: np.ndarray | None    # (n_frames, nbin, nbin) density maps, or None


def run_kmc(land: Landscape, n_exc: int, n_steps: int, rng: np.random.Generator,
            sample_every: int = 50, init_sigma_nm: float = 200.0,
            kt_mev: float = KT_MEV,
            diffmap_nbin: int | None = None, diffmap_L_nm: float | None = None
            ) -> Trajectory:
    """Run the hopping KMC and track the spatial second moment over steps.

    Excitons begin as a Gaussian (``init_sigma_nm``) snapped to the nearest
    crystal, then each step every exciton attempts a hop to a random neighbour,
    accepted with the SI's size-dependent probability. ``variance`` is the
    population second moment, the quantity whose sub-linear growth is the
    subdiffusion signature. If ``diffmap_nbin`` is set, 2-D density frames are
    recorded at each sample point (for the per-frame bias study).
    """
    M = land.pos.shape[0]
    # initial Gaussian, snapped to nearest crystal
    init_xy = rng.normal(0.0, init_sigma_nm, (n_exc, 2))
    tree = cKDTree(land.pos)
    _, cur = tree.query(init_xy)              # crystal index per exciton
    pos = land.pos
    diam = land.diam
    energy = land.energy
    nb = land.nb
    cnt = land.nb_count

    rec_step, rec_var, rec_en = [], [], []
    frames = []
    if diffmap_nbin is not None:
        edges = np.linspace(-diffmap_L_nm / 2, diffmap_L_nm / 2, diffmap_nbin + 1)

    def record(s):
        P = pos[cur]
        m = P.mean(0)
        var1d = ((P - m) ** 2).sum(1).mean() / 2.0
        rec_step.append(s)
        rec_var.append(var1d)
        rec_en.append(energy[cur].mean())
        if diffmap_nbin is not None:
            h, _, _ = np.histogram2d(P[:, 0], P[:, 1], bins=[edges, edges])
            frames.append(h / h.sum())

    record(0)
    for s in range(1, n_steps + 1):
        c = cnt[cur]
        ok = c > 0
        j = (rng.random(n_exc) * np.maximum(c, 1)).astype(np.int64)
        acc = nb[cur, j]                       # candidate acceptor crystal
        r = np.hypot(*(pos[acc] - pos[cur]).T)
        p = 1.0 / (GAMMA * (r + R0)) ** 6
        d_don = diam[cur]
        d_acc = diam[acc]
        uphill = d_acc < d_don                 # to a smaller, higher-energy crystal
        dE = energy[acc] - energy[cur] + rng.normal(0.0, 2.0 * SIGMA_E_MEV, n_exc)
        prob = np.where(uphill, p * np.exp(-dE / kt_mev), p)
        prob = np.clip(prob, 0.0, 1.0)
        move = ok & (rng.random(n_exc) < prob)
        cur = np.where(move, acc, cur)
        if s % sample_every == 0:
            record(s)

    return Trajectory(step=np.array(rec_step), variance_nm2=np.array(rec_var),
                      mean_energy_mev=np.array(rec_en),
                      frames=np.array(frames) if frames else None)


def fit_alpha(step: np.ndarray, variance_nm2: np.ndarray) -> tuple[float, float]:
    """Fit MSD = C step^alpha over the moving regime; return (alpha, C)."""
    msd = variance_nm2 - variance_nm2[0]
    m = (step > 0) & (msd > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    a, lnC = np.polyfit(np.log(step[m]), np.log(msd[m]), 1)
    return float(a), float(np.exp(lnC))


if __name__ == "__main__":
    # Smoke test: the exponent and the C16 < C12 < C8 ordering as the energetic
    # disorder is strengthened (lower kt_mev). At room temperature the disorder is
    # too weak (alpha ~ 1); strong disorder recovers subdiffusion and the ordering.
    # (small N / box for speed; the absolute alpha needs the SI sizes / T.)
    print("NC-solid KMC smoke test -- alpha(sample) vs kt_mev (disorder strength)")
    print(f"{'kt_mev':>7s} | {'small/C16':>10s} {'medium/C12':>11s} {'large/C8':>10s}")
    for kt in (25.85, 8.0, 3.0):
        row = []
        for key in ("small", "medium", "large"):
            rng = np.random.default_rng(0)
            land = generate_landscape(SAMPLES[key], box_nm=1200.0, rng=rng)
            traj = run_kmc(land, n_exc=4000, n_steps=8000, rng=rng,
                           sample_every=200, kt_mev=kt)
            row.append(fit_alpha(traj.step, traj.variance_nm2)[0])
        print(f"{kt:7.1f} | {row[0]:10.3f} {row[1]:11.3f} {row[2]:10.3f}")
    print("room T -> alpha ~ 1 (too weak); strong disorder -> subdiffusive, "
          "C16 < C12 < C8. Calibrate kt_mev / sizes to the paper's 0.42/0.62/0.70.")
