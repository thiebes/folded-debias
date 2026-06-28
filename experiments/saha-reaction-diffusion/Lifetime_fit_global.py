#%%
# Global version of the lifetime fit: all fluences fit at once with one shared
# set of rates, instead of one fit per fluence.
#
# Fitting each fluence on its own leaves nu and rho underdetermined - a single
# decay can't pin every rate, so they come out different for each fluence, which
# can't be right for one material. Fitting the curves together with nu, rho, lam,
# gamma, qtrap (and qratio) shared fixes that: the only thing that changes
# between curves is the known Nexc, so you get one nu/rho to put in the diffusion
# sim.
#
# SHARE_QRATIO - qratio is an emission ratio, so I share it too by default (the
#   overall brightness is already taken out when each curve is normalised). Set
#   it False to let it float per fluence.
# FIT_AUGER - adds the Meitner-Auger term -C*n^3 (shared). Off by default; turn
#   it on for the high-density data where the extra channel shows up.

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import glob
import json
import os
from scipy.optimize import differential_evolution

SHARE_QRATIO = True
FIT_AUGER    = False

# %%

SAMPLE = "n_inf"
json_path = r"/Users/antonellacutrupi/Desktop/Diffusion_free_exciton_simulation/Samples_Eb_nu_rho_mu.json"
folder = "/Users/antonellacutrupi/Desktop/Diffusion_free_exciton_simulation/fluence/n_inf/"

with open(json_path, "r") as f:
    samples_json = json.load(f)

s = samples_json[SAMPLE]
Eb_meV = s["Eb"]
nu_0 = s["nu"]
rho_0 = s["rho"]
mu_r = s["mu"]

print("=" * 60)
print(f"  sample: {SAMPLE}")
print(f"  Eb       : {s['Eb']} meV")
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
pl_exp_list = []

for fpath in files:
    fname = os.path.basename(fpath)
    fluence = float(fname.split("_")[0])
    data = np.load(fpath, allow_pickle=True)

    if isinstance(data.item(), dict):
        data = data.item()
        pl_exp = data['Integrated_amplitudes']
        time = data["Time (ns)"]
    else:
        time = data[1]
        pl_exp = data[0]

    fluence_J = fluence * 1e-6
    phy = fluence_J / E_foton
    alpha = 2e5
    d = 1 / alpha
    f = 1 - np.exp(-1)
    N_exc = phy * f / d

    fluence_list.append(fluence)
    Nexc_list.append(N_exc)
    time_exp_list.append(time)
    pl_exp_list.append(pl_exp)

    print(f"  Fluence: {fluence:>6.3f} uJ/cm2  -->  Nexc = {N_exc:.3e}")

print("=" * 60)

A = calc_A_3D(eV_to_J, kb, m0, mu, Eb_meV, T)
nfluence = len(Nexc_list)

#%%

def run_lifetime_simple(qratio, nu, rho, n0, A,
                        lam=0.0, gamma=0.0, qtrap=0.0, Cauger=0.0,
                        dt=1e-11, tmax=20e-9, max_steps=2_000_000):
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

        # exciton, bimolecular, Meitner-Auger, trapping
        dn = (-nu * (1 - x) * n
              - rho * x * x * n * n
              - Cauger * n * n * n
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


def unpack(params):
    nu    = 10 ** params[0]
    rho   = 10 ** params[1]
    lam   = 10 ** params[2]
    gamma = 10 ** params[3]
    qtrap = params[4]
    i = 5
    if FIT_AUGER:
        Cauger = 10 ** params[i]; i += 1
    else:
        Cauger = 0.0
    if SHARE_QRATIO:
        qratios = [params[i]] * nfluence; i += 1
    else:
        qratios = list(params[i:i + nfluence]); i += nfluence
    return nu, rho, lam, gamma, qtrap, Cauger, qratios


def loss_global(params):
    nu, rho, lam, gamma, qtrap, Cauger, qratios = unpack(params)
    total = 0.0
    for k in range(nfluence):
        t_exp = time_exp_list[k] * 1e-9
        pl_exp = pl_exp_list[k]
        mask = (t_exp <= 20e-9) & (pl_exp > 0)
        t = t_exp[mask]
        y = pl_exp[mask].astype(float)

        t_sim, ysim = run_lifetime_simple(qratios[k], nu, rho, Nexc_list[k], A,
                                          lam=lam, gamma=gamma, qtrap=qtrap, Cauger=Cauger,
                                          dt=1e-11, tmax=20e-9)
        ysim = np.interp(t, t_sim, ysim)

        y    = y / np.max(y)
        ysim = ysim / np.max(ysim)
        # log residual so the tail drives the fit, summed over all fluences
        total += np.mean((np.log10(ysim + 1e-12) - np.log10(y + 1e-12)) ** 2)
    return total


# bounds in the same order unpack() reads them
bounds = [
    (3.0,   12.0),   # log10 nu
    (-18.0, -8.0),   # log10 rho
    (3.0,    9.0),   # log10 lam     trapping rate
    (3.0,    8.0),   # log10 gamma   slow trapped decay (lower = slower tail)
    (0.0,  100.0),   # qtrap         trap emission ratio
]
if FIT_AUGER:
    bounds.append((-32.0, -26.0))   # log10 Cauger  Meitner-Auger, around 1e-28
if SHARE_QRATIO:
    bounds.append((0.0, 100.0))     # qratio shared
else:
    bounds += [(0.0, 100.0)] * nfluence

print("\n" + "=" * 60)
print(f"GLOBAL FIT  (share_qratio={SHARE_QRATIO}, fit_auger={FIT_AUGER})")
print("=" * 60)

result = differential_evolution(
    loss_global,
    bounds=bounds,
    seed=50,
    maxiter=1000,
    tol=1e-7,
    popsize=15,
    mutation=(0.5, 1),
    recombination=0.7,
    polish=True,
    disp=True,
)

nu, rho, lam, gamma, qtrap, Cauger, qratios = unpack(result.x)
print(f"\n  nu     = {nu:.3e}")
print(f"  rho    = {rho:.3e}")
print(f"  lam    = {lam:.3e}")
print(f"  gamma  = {gamma:.3e}")
print(f"  qtrap  = {qtrap:.4e}")
if FIT_AUGER:
    print(f"  Cauger = {Cauger:.3e}")
for k in range(nfluence):
    print(f"  qratio[{k}] (Nexc={Nexc_list[k]:.2e}) = {qratios[k]:.4e}")
print(f"  loss   = {result.fun:.4e}")

#%%
# --- plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
colors_trpl = ['blue', 'green', 'purple']

for k in range(nfluence):
    color = colors_trpl[k % len(colors_trpl)]
    time_exp = time_exp_list[k]
    pl_exp = pl_exp_list[k]

    mask_view = time_exp <= 20.0
    ax.errorbar(
        time_exp[mask_view], pl_exp[mask_view] / (np.max(pl_exp[mask_view]) + 1e-30),
        fmt='o', ms=5, color=color, alpha=0.8,
        label=f'Exp Nexc={Nexc_list[k]:.2e}'
    )

    time_sim, pl_sim = run_lifetime_simple(qratios[k], nu, rho, Nexc_list[k], A,
                                           lam=lam, gamma=gamma, qtrap=qtrap, Cauger=Cauger,
                                           dt=1e-11, tmax=20e-9)
    ax.plot(time_sim * 1e9, pl_sim, '-', color=color, linewidth=2,
            label=f'Fit Nexc={Nexc_list[k]:.2e}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time (ns)', fontsize=12)
ax.set_ylabel('PL (norm)', fontsize=12)
ax.set_title('Global fit, shared rates', fontsize=12)
ax.legend(fontsize=9, framealpha=0.9)
ax.set_ylim(3e-3, 1.1)

plt.tight_layout()
plt.show()
# %%
