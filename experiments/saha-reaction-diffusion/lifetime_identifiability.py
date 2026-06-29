"""Identifiability analysis of the global multi-fluence lifetime fit.

Point --data at a folder holding a sample-parameter JSON (Eb, mu per sample) and
one ``*_uJcm2.npy`` per fluence (each a dict with ``Time (ns)`` and
``Integrated_amplitudes``). The script:

  fit       -- global fit of all fluences with one shared rate set
               (nu, rho, lam, gamma, qtrap), only Nexc differing.
  profile   -- deterministic profile likelihood of each rate (fix one, refit the
               rest; SSR vs the fixed rate), with ~1sigma confidence intervals.
               A flat profile means the rate is not determined by the data.
  tradeoff  -- demonstrates the Saha rho<->nu degeneracy: refit nu while
               profiling rho; nu tracks nu_eff - rho*A (the fit holds nu + rho*A
               fixed) until rho*A reaches nu and the fit breaks.

The model mirrors Lifetime_fit_global.py. A Meitner-Auger term was tested
separately and is not needed for the n_inf data (its coefficient runs to zero),
so it is omitted here. qratio is shared across fluences.

Example::

    python lifetime_identifiability.py \
      --data "P:/Marc/shared_data/data_links/Diffusion_simulation_Antonella/TPLM" \
      --sample n_inf --out output/n_inf_lifetime --mode all
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# fit-space parameter vector: [log10 nu, log10 rho, log10 lam, log10 gamma,
#                              qtrap, qratio]
NAMES = ["nu", "rho", "lam", "gamma", "qtrap", "qratio"]
BOUNDS = [(3, 12), (-18, -8), (3, 9), (3, 11), (0, 100), (0, 100)]

H = 6.62607015e-34; KB = 1.380649e-23; M0 = 9.10938356e-31
T = 300.0; Q0 = 1.602176634e-19
LASER_NM = 375.0
ALPHA = 2e5            # absorption coefficient (1/m), for Nexc from fluence


def load_data(data_dir, sample, laser_nm=LASER_NM):
    """Return (A, Nexc_list, time_list, pl_list) for the sample's fluences."""
    js = glob.glob(os.path.join(data_dir, "*.json"))
    s = None
    for j in js:
        d = json.load(open(j))
        if isinstance(d, dict) and sample in d:
            s = d[sample]; break
    if s is None:
        raise SystemExit(f"sample {sample!r} not found in any JSON in {data_dir}")
    Eb_meV, mu_r = s["Eb"], s["mu"]
    A = ((2*np.pi*mu_r*M0*KB*T/H**2)**1.5
         * np.exp(-(Eb_meV*1e-3*Q0)/(KB*T))) * 1e-6      # cm^-3
    E_photon = H * 2.99792458e8 / (laser_nm*1e-9)
    Nexc, times, pls = [], [], []
    for f in sorted(glob.glob(os.path.join(data_dir, "*_uJcm2.npy"))):
        dd = np.load(f, allow_pickle=True).item()
        fl = float(os.path.basename(f).split("_")[0])
        Nexc.append((fl*1e-6/E_photon) * (1-np.exp(-1)) / (1/ALPHA))
        times.append(np.asarray(dd["Time (ns)"], float))
        pls.append(np.asarray(dd["Integrated_amplitudes"], float))
    if not Nexc:
        raise SystemExit(f"no *_uJcm2.npy fluence files in {data_dir}")
    return A, Nexc, times, pls


def run_lifetime(A, qratio, nu, rho, n0, lam, gamma, qtrap,
                 dt=1e-11, tmax=20e-9, max_steps=2_000_000):
    """Free + trapped population PL decay; normalised to PL(0)."""
    n, m, t, step = n0, 0.0, 0.0, 0
    ts, pl = [], []
    while t < tmax and step < max_steps:
        ntilde = n / A
        x = 2/(np.sqrt(4*ntilde+1)+1) if ntilde > 0 else 1.0
        pl.append(qratio*nu*(1-x)*n + rho*x*x*n*n + qtrap*gamma*m)
        dn = -nu*(1-x)*n - rho*x*x*n*n - lam*n
        dm = lam*n - gamma*m
        rn = abs(dn/n) if n > 0 else 0.0
        rm = abs(dm/m) if m > 0 else 0.0
        rate = max(rn, rm)
        dt_eff = min(dt, 0.01/rate) if rate > 0 else dt
        ts.append(t); n += dt_eff*dn; m += dt_eff*dm
        n = max(n, 0); m = max(m, 0); t += dt_eff; step += 1
    ts = np.array(ts); pl = np.array(pl)
    if pl.size and pl[0] > 0:
        pl /= pl[0]
    return ts, pl


def ssr(p, data):
    """Total sum of squared log-PL residuals over all fluences; and point count."""
    A, Nexc, times, pls = data
    nu, rho, lam, gamma, qtrap, qratio = (10**p[0], 10**p[1], 10**p[2],
                                          10**p[3], p[4], p[5])
    tot, N = 0.0, 0
    for n0, te, ple in zip(Nexc, times, pls):
        ts = te*1e-9; mask = (ts <= 20e-9) & (ple > 0)
        t, y = ts[mask], ple[mask]
        tsim, ysim = run_lifetime(A, qratio, nu, rho, n0, lam, gamma, qtrap)
        ysim = np.interp(t, tsim, ysim)
        y = y/np.max(y); ysim = ysim/np.max(ysim)
        r = np.log10(ysim+1e-12) - np.log10(y+1e-12)
        tot += np.sum(r**2); N += r.size
    return tot, N


# --------------------------------------------------------------------------
def global_fit(data, seed=50, maxiter=1000):
    res = differential_evolution(lambda p: ssr(p, data)[0], BOUNDS, seed=seed,
                                 maxiter=maxiter, tol=1e-7, popsize=15,
                                 mutation=(0.5, 1), recombination=0.7,
                                 polish=True, disp=False)
    return res.x


def _refit_fixing(idx, value, pbest, data):
    free = [j for j in range(6) if j != idx]
    x0 = [pbest[j] for j in free]; bnds = [BOUNDS[j] for j in free]
    def obj(xf):
        p = pbest.copy(); p[idx] = value
        for k, j in enumerate(free):
            p[j] = xf[k]
        return ssr(p, data)[0]
    r = minimize(obj, x0, method="L-BFGS-B", bounds=bnds,
                 options={"maxiter": 200, "ftol": 1e-11})
    return r.fun, r.x, free


def profile(data, pbest, out, prefix, npts=21):
    """Profile each rate; mark the ~1sigma SSR threshold; save a figure."""
    SSR_min, N = ssr(pbest, data)
    s2 = SSR_min / (N - 6)
    thr = SSR_min + s2
    print(f"SSR_min={SSR_min:.4e} N={N} s^2={s2:.3e}  (1sigma: deltaSSR=s^2)")
    grids = {0: np.linspace(7.0, 10.0, npts), 2: np.linspace(7.0, 9.0, npts),
             3: np.linspace(5.0, 11.0, npts+4), 1: np.linspace(-18, -9, npts+4)}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, idx in zip(axes.ravel(), [0, 2, 3, 1]):
        grid = grids[idx]
        prof = np.array([_refit_fixing(idx, v, pbest, data)[0] for v in grid])
        ax.plot(grid, prof, "o-"); ax.axhline(thr, color="g", ls=":",
                                               label="SSR_min + s^2")
        ax.axvline(pbest[idx], color="r", ls="--", lw=1, label="best fit")
        ax.set_xlabel("log10 " + NAMES[idx]); ax.set_ylabel("profiled SSR")
        rng = np.nanmax(prof) - np.nanmin(prof)
        verdict = "determined" if rng > 10*s2 else "DEGENERATE (flat)"
        ax.set_title(f"{NAMES[idx]}: {verdict}", fontsize=10); ax.legend(fontsize=8)
        print(f"  {NAMES[idx]:6s} dSSR over scan = {rng:.3e}  -> {verdict}")
    fig.tight_layout()
    path = os.path.join(out, f"{prefix}_profile_ci.png"); fig.savefig(path, dpi=115)
    print("saved", path)


def tradeoff(data, pbest, out, prefix, npts=25):
    """Profile rho, recording the refit nu, to show nu = nu_eff - rho*A."""
    A = data[0]
    nu_eff = 10**pbest[0]
    grid = np.linspace(-11, -8, npts)
    ssr_v, nu_v = [], []
    for lr in grid:
        fun, x, free = _refit_fixing(1, lr, pbest, data)
        ssr_v.append(fun); nu_v.append(10**x[free.index(0)])
    ssr_v, nu_v = np.array(ssr_v), np.array(nu_v)
    rho = 10**grid; cross = np.log10(nu_eff/A)
    print(f"rho*A = nu_eff at rho = {nu_eff/A:.3e} (log10 {cross:.2f})")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(grid, ssr_v, "o-"); ax[0].axvline(cross, color="g", ls=":",
                                                  label="rho*A = nu_eff")
    ax[0].set_xlabel("log10 rho"); ax[0].set_ylabel("profiled SSR")
    ax[0].set_title("fit quality flat until rho*A reaches nu"); ax[0].legend(fontsize=8)
    ax[1].plot(grid, nu_v, "o-", label="refit nu")
    ax[1].plot(grid, np.clip(nu_eff-rho*A, 0, None), "r--", label="nu_eff - rho*A")
    ax[1].axvline(cross, color="g", ls=":")
    ax[1].set_xlabel("log10 rho"); ax[1].set_ylabel("nu (/s)")
    ax[1].set_title("nu drops to keep nu + rho*A fixed (the degeneracy)")
    ax[1].legend(fontsize=8); fig.tight_layout()
    path = os.path.join(out, f"{prefix}_rho_tradeoff.png"); fig.savefig(path, dpi=115)
    print("saved", path)


def plot_fit(data, pbest, out, prefix):
    A, Nexc, times, pls = data
    nu, rho, lam, gamma, qtrap, qratio = (10**pbest[0], 10**pbest[1], 10**pbest[2],
                                          10**pbest[3], pbest[4], pbest[5])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["blue", "green", "purple"]
    for i, (n0, te, ple) in enumerate(zip(Nexc, times, pls)):
        c = colors[i % len(colors)]; mv = te <= 20.0
        ax.plot(te[mv], ple[mv]/(np.max(ple[mv])+1e-30), "o", ms=4, color=c,
                label=f"Exp Nexc={n0:.2e}")
        ts, ys = run_lifetime(A, qratio, nu, rho, n0, lam, gamma, qtrap)
        ax.plot(ts*1e9, ys, "-", color=c, lw=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-3, 1.1)
    ax.set_xlabel("Time (ns)"); ax.set_ylabel("PL (norm)")
    ax.set_title("Global fit, shared rates"); ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, f"{prefix}_global_fit.png"); fig.savefig(path, dpi=115)
    print("saved", path)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="TPLM folder (JSON + *_uJcm2.npy)")
    p.add_argument("--sample", default="n_inf")
    p.add_argument("--out", default="output/lifetime_identifiability")
    p.add_argument("--mode", default="all",
                   choices=["all", "fit", "profile", "tradeoff"])
    p.add_argument("--seed", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_data(args.data, args.sample)
    print(f"A = {data[0]:.3e} cm^-3;  {len(data[1])} fluences "
          f"(Nexc {min(data[1]):.2e}..{max(data[1]):.2e})")

    print("global fit (seeded differential_evolution)...")
    pbest = global_fit(data, seed=args.seed)
    nu, rho, lam, gamma, qtrap, qratio = (10**pbest[0], 10**pbest[1], 10**pbest[2],
                                          10**pbest[3], pbest[4], pbest[5])
    print(f"  nu={nu:.3e} rho={rho:.3e} lam={lam:.3e} gamma={gamma:.3e} "
          f"qtrap={qtrap:.2f} qratio={qratio:.2f}")

    prefix = f"{args.sample}_lifetime"
    if args.mode in ("all", "fit"):
        plot_fit(data, pbest, args.out, prefix)
    if args.mode in ("all", "profile"):
        profile(data, pbest, args.out, prefix)
    if args.mode in ("all", "tradeoff"):
        tradeoff(data, pbest, args.out, prefix)


if __name__ == "__main__":
    main()
