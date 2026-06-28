#%%
# Lifetime fit with a trapped slow population.
# Free carriers n trap at rate lam; trapped carriers m decay slowly (gamma << nu)
# and emit with relative yield qtrap. The qtrap*gamma*m term is the long tail.

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import glob
import json
import os
from scipy.optimize import minimize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

@dataclass
class parameters:
    L: float
    nbins: int
    nu: float
    rho: float
    A: float
    qe: float
    qf: float
    De: float
    Df: float
    tmax: float
    tsamp: float
    dt: float
    initial: np.ndarray


def diffusion(par: parameters):
    nsteps = int(par.tmax / par.dt)
    nsamp = int(par.tsamp / par.dt)
    dx = (1.0 * par.L) / par.nbins
    n = par.initial.copy()
    nn = np.zeros(par.nbins)
    x = np.ones(par.nbins)
    x[0] = 1
    x[par.nbins - 1] = 1
    j = 0
    nframes = int(par.tmax / par.tsamp) + 1

    diffmap = np.zeros(shape=(nframes, par.nbins))

    for t in range(nsteps):
        for i in range(1, par.nbins - 1):
            ntilde = n[i] / par.A
            if ntilde > 0:
                x[i] = (np.sqrt(4 * ntilde + 1) - 1) / (2 * ntilde)
            else:
                x[i] = 1

        for i in range(1, par.nbins - 1):
            exciton_diff = (1 - x[i + 1]) * n[i + 1] + (1 - x[i - 1]) * n[i - 1] - 2 * (1 - x[i]) * n[i]
            freecharge_diff = x[i + 1] * n[i + 1] + x[i - 1] * n[i - 1] - 2 * x[i] * n[i]

            nn[i] = (n[i]
                     + (par.De * par.dt / (dx * dx)) * exciton_diff
                     + (par.Df * par.dt / (dx * dx)) * freecharge_diff
                     - par.dt * n[i] * (par.nu * (1 - x[i]) + par.rho * n[i] * x[i] * x[i]))

            nn[i] = max(0, nn[i])

        for i in range(1, par.nbins - 1):
            n[i] = nn[i]

        if np.any(np.isnan(n)) or np.any(np.isinf(n)):
            break

        if t % nsamp == 0:
            for i in range(1, par.nbins - 1):
                diffmap[j, i] = n[i] * (par.qe * (1 - x[i]) * par.nu
                                        + par.qf * x[i] * x[i] * par.rho * n[i]) * par.dt
            j += 1

    return diffmap

def variance(par: parameters, diffmap: np.ndarray):
    dy = (1.0 * par.L) / par.nbins
    y = np.linspace(-0.5 * par.L, 0.5 * par.L, par.nbins)
    y_center = y + 0.5 * dy
    var = np.zeros(diffmap.shape[0])

    for t in range(diffmap.shape[0]):
        PL = np.sum(diffmap[t])
        if PL == 0:
            var[t] = 0
            continue
        meany = np.sum(y_center * diffmap[t]) / PL
        var[t] = np.sum((y_center - meany) ** 2 * diffmap[t]) / PL

    var_initial = var[0]
    var_corrected = var - var_initial
    var_corrected[0] = 0
    var_corrected = np.maximum(var_corrected, 0)

    return var_corrected, var_initial


def cut_negative_msd_start(time, msd, msd_err):

    mask = msd >= 0
    if not np.any(mask):
        return time, msd, msd_err

    first_pos = np.argmax(mask)
    return time[first_pos:], msd[first_pos:], msd_err[first_pos:]

# %%

SAMPLE = "n_inf"
json_path = r"/Users/antonellacutrupi/Desktop/Diffusion_free_exciton_simulation/Samples_Eb_nu_rho_mu.json"
folder = "/Users/antonellacutrupi/Desktop/Diffusion_free_exciton_simulation/fluence/n_inf/"

with open(json_path, "r") as f:
    samples_json = json.load(f)

s = samples_json[SAMPLE]
Eb_meV = s["Eb"] #meV
nu_0 = s["nu"]
rho_0 = s["rho"]
mu_r = s["mu"]

print("=" * 60)
print(f"  sample: {SAMPLE}")
print(f"  Eb       : {s['Eb']} meV")
print(f"  nu       : {nu_0}")
print(f"  rho      : {rho_0}")
print(f"  mu       : {mu_r} m0")
print("=" * 60)

h_planck = 6.62607015e-34
c_light = 2.99792458e8
laser_wavelength_nm = 375
E_foton = h_planck * c_light / (laser_wavelength_nm * 1e-9)

m0 = 9.10938356e-31
kb = 1.380649e-23
h = 6.62607015e-34
T = 300
eV_to_J = 1.602176634e-19
mu = mu_r

def calc_A_3D(q0, kB, m0, mu, Eb_meV, T):
    Eb_J = Eb_meV * 1e-3 * q0

    prefactor = (2 * np.pi * mu * m0 * kB * T / h**2) ** (3/2)

    A_m3 = prefactor * np.exp(-Eb_J / (kB * T))

    return A_m3 * 1e-6  # m^-3 -> cm^-3


files = sorted(glob.glob(os.path.join(folder, "*_uJcm2.npy")))

fluence_list = []
Nexc_list = []
time_exp_list = []
msd_exp_list = []
msd_err_list = []
pl_exp_list = []

for fpath in files:
    fname = os.path.basename(fpath)
    fluence = float(fname.split("_")[0])
    data = np.load(fpath, allow_pickle=True)

    if isinstance(data.item(), dict):
        data = data.item()
        pl_exp = data['Integrated_amplitudes']
        #pl_exp = data['Amplitudes']
        msd = data["MSD_values(um2)"]
        time = data["Time (ns)"]
        msd_err = data["MSD_error (um2)"]
    else:
        msd = data[0]
        time = data[1]
        msd_err = np.zeros_like(msd)

    fluence_J = fluence * 1e-6
    phy = fluence_J / E_foton

    alpha = 2e5
    d = 1/alpha
    f = 1 - np.exp(-1)
    N_exc = phy * f /d

    fluence_list.append(fluence)
    Nexc_list.append(N_exc)
    time_exp_list.append(time)
    pl_exp_list.append(pl_exp)
    msd_exp_list.append(msd)
    msd_err_list.append(msd_err)

    print(f"  Fluence: {fluence:>6.3f} uJ/cm2  -->  Nexc = {N_exc:.3e}")

print("=" * 60)

A = calc_A_3D(eV_to_J,kb, m0, mu, Eb_meV, T)
#%%

print("\n" + "=" * 60)
print("LIFETIME FIT (free carriers + trapped slow population)")
print("=" * 60)

def run_lifetime_simple(qratio, nu, rho, n0, A,
                        lam=0.0, gamma=0.0, qtrap=0.0,
                        dt=1e-11, tmax=20e-9,
                        max_steps=2_000_000, verbose=False):
    n = n0          # free carriers
    m = 0.0         # trapped
    t = 0.0
    time_list = []
    PL_list = []
    step = 0

    while t < tmax and step < max_steps:
        ntilde = n / A
        if ntilde > 0:
            x = 2 / (np.sqrt(4 * ntilde + 1) + 1)
        else:
            x = 1.0

        # free channels + slow trap term
        PL_val = qratio * nu * (1 - x) * n + rho * x * x * n * n + qtrap * gamma * m

        # exciton, bimolecular, trapping
        # (Meitner-Auger -C*n^3 omitted: high-density early times only, not the tail)
        dn = (-nu * (1 - x) * n
              - rho * x * x * n * n
              - lam * n)
        # trapping in, slow recombination out
        dm = lam * n - gamma * m

        # step on the faster population
        rate_n = abs(dn / n) if n > 0 else 0.0
        rate_m = abs(dm / m) if m > 0 else 0.0
        rate   = max(rate_n, rate_m)
        dt_eff = min(dt, 0.01 / rate) if rate > 0 else dt

        time_list.append(t)
        PL_list.append(PL_val)

        n += dt_eff * dn
        m += dt_eff * dm
        n  = max(n, 0)
        m  = max(m, 0)
        t += dt_eff
        step += 1

    time = np.array(time_list)
    PL   = np.array(PL_list)
    if PL.size > 0 and PL[0] > 0:
        PL /= PL[0]

    return time, PL

from scipy.optimize import differential_evolution

def loss_single(params, Nexc, time_exp, pl_exp):
    qratio, log_nu, log_rho, log_lam, log_gamma, qtrap = params
    nu    = 10 ** log_nu
    rho   = 10 ** log_rho
    lam   = 10 ** log_lam
    gamma = 10 ** log_gamma

    t_exp = time_exp * 1e-9
    mask  = (t_exp <= 20e-9) & (pl_exp > 0)
    t = t_exp[mask]
    y = pl_exp[mask].copy()

    t_sim, ysim = run_lifetime_simple(qratio, nu, rho, Nexc, A,
                                      lam=lam, gamma=gamma, qtrap=qtrap,
                                      dt=1e-11, tmax=20e-9)
    ysim = np.interp(t, t_sim, ysim)

    y    = y / np.max(y)
    ysim = ysim / np.max(ysim)

    # log residual so the tail drives the fit
    return np.mean((np.log10(ysim + 1e-12) - np.log10(y + 1e-12)) ** 2)

bounds = [
    (0.0, 100.0),    # qratio
    (3.0,   12.0),   # log10 nu
    (-18.0, -8.0),   # log10 rho
    (3.0,    9.0),   # log10 lam     trapping rate
    (3.0,    8.0),   # log10 gamma   slow trapped decay (lower = slower tail)
    (0.0,  100.0),   # qtrap         trap emission ratio (> 0 for a tail)
]

fit_results = []

for i, (Nexc, time_exp, pl_exp) in enumerate(zip(Nexc_list, time_exp_list, pl_exp_list)):
    print(f"\nFitting Nexc={Nexc:.3e} ...")
    result = differential_evolution(
        loss_single,
        bounds=bounds,
        args=(Nexc, time_exp, pl_exp),
        seed=50,
        maxiter=1000,
        tol=1e-7,
        popsize=15,
        mutation=(0.5, 1),
        recombination=0.7,
        polish=True,
        disp=False,
    )
    qratio_i = result.x[0]
    nu_i     = 10 ** result.x[1]
    rho_i    = 10 ** result.x[2]
    lam_i    = 10 ** result.x[3]
    gamma_i  = 10 ** result.x[4]
    qtrap_i  = result.x[5]
    fit_results.append((qratio_i, nu_i, rho_i, lam_i, gamma_i, qtrap_i))
    print(f"  qratio = {qratio_i:.4e}")
    print(f"  nu     = {nu_i:.3e}")
    print(f"  rho    = {rho_i:.3e}")
    print(f"  lam    = {lam_i:.3e}")
    print(f"  gamma  = {gamma_i:.3e}")
    print(f"  qtrap  = {qtrap_i:.4e}")
    print(f"  loss   = {result.fun:.4e}")


#%%
# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
colors_trpl = ['blue', 'green', 'purple']

for i, (Nexc, time_exp, pl_exp) in enumerate(zip(Nexc_list, time_exp_list, pl_exp_list)):
    color = colors_trpl[i % len(colors_trpl)]
    qratio_i, nu_i, rho_i, lam_i, gamma_i, qtrap_i = fit_results[i]

    mask_view = time_exp <= 20.0
    ax.errorbar(
        time_exp[mask_view], pl_exp[mask_view] / (np.max(pl_exp[mask_view]) + 1e-30),
        fmt='o', ms=5, elinewidth=1, capsize=3,
        color=color, alpha=0.8,
        label=f'Exp Nexc={Nexc:.2e}'
    )

    time_sim, pl_sim = run_lifetime_simple(qratio_i, nu_i, rho_i, Nexc, A,
                                           lam=lam_i, gamma=gamma_i, qtrap=qtrap_i,
                                           dt=1e-11, tmax=20e-9)
    ax.plot(
        time_sim * 1e9, pl_sim,
        '-', color=color, linewidth=2,
        label=f'Fit Nexc={Nexc:.2e}'
    )

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('PL (norm)', fontsize=12)
ax.set_title('Fit with trapped slow population', fontsize=12)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_ylim(3e-3, 1.1)

plt.tight_layout()
plt.show()
# %%
