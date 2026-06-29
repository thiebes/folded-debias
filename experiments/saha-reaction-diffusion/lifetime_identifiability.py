"""Identifiability analysis of the global multi-fluence lifetime fit.

Point --data at a folder holding a sample-parameter JSON (Eb, mu per sample) and
one ``*_uJcm2.npy`` per fluence (each a dict with ``Time (ns)`` and
``Integrated_amplitudes``). The script:

  fit       -- global fit of all fluences with one shared rate set
               (nu, rho, lam, gamma, qtrap), only Nexc differing.
  profile   -- deterministic profile likelihood of each rate (fix one, refit the
               rest; SSR vs the fixed rate), with an ~1sigma threshold. A flat
               profile means the rate is not determined by the data.
  tradeoff  -- the Saha rho<->nu degeneracy: refit nu while profiling rho. nu
               tracks nu_eff - rho*A (the fit holds nu + rho*A fixed) and lam,
               gamma do not move, until rho*A reaches nu and the fit breaks.
  saha      -- the exact Saha identity (1-x) = x^2 n/A makes the exciton rate
               nu(1-x) and the bimolecular rate rho*x^2*n proportional at all
               densities (constant ratio nu/(rho*A)), so the fit sees only their
               sum nu + rho*A. This is why rho is degenerate with nu.

The model mirrors Lifetime_fit_global.py. A Meitner-Auger term was tested
separately and is not needed for the n_inf data (its coefficient runs to zero),
so it is omitted here. qratio is shared across fluences.

Examples::

    python lifetime_identifiability.py --data "<TPLM folder>" --sample n_inf \
        --out output/n_inf_lifetime --mode all
    # regenerate the figures from the cached best fit, skipping the optimiser:
    python lifetime_identifiability.py --data "<TPLM folder>" --no-fit
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

plt.rcParams.update({"axes.grid": False})
CB = ["#0072B2", "#009E73", "#CC79A7"]   # colorblind-safe: blue, green, magenta

NAMES = ["nu", "rho", "lam", "gamma", "qtrap", "qratio"]
BOUNDS = [(3, 12), (-18, -8), (3, 9), (3, 11), (0, 100), (0, 100)]
# cached global best fit (for --no-fit figure regeneration)
CACHED_BEST = np.array([np.log10(6.58e8), np.log10(5.95e-14), np.log10(4.06e8),
                        np.log10(1.05e8), 35.96, 98.41])

H = 6.62607015e-34; KB = 1.380649e-23; M0 = 9.10938356e-31
T = 300.0; Q0 = 1.602176634e-19
LASER_NM = 375.0
ALPHA = 2e5            # absorption coefficient (1/m), for Nexc from fluence


def load_data(data_dir, sample, laser_nm=LASER_NM):
    """Return (A, Nexc_list, time_list, pl_list) for the sample's fluences."""
    s = None
    for j in glob.glob(os.path.join(data_dir, "*.json")):
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


def plot_fit(data, pbest, out, prefix):
    A, Nexc, times, pls = data
    nu, rho, lam, gamma, qtrap, qratio = (10**pbest[0], 10**pbest[1], 10**pbest[2],
                                          10**pbest[3], pbest[4], pbest[5])
    fig, ax = plt.subplots(figsize=(6, 5.2), constrained_layout=True)
    order = np.argsort(Nexc)
    for i in order:
        c = CB[list(order).index(i) % 3]
        te, ple, n0 = times[i], pls[i], Nexc[i]
        mv = te <= 20.0
        ax.plot(te[mv], ple[mv]/(np.max(ple[mv])+1e-30), "o", ms=4, mfc="none",
                color=c, label=f"Nexc = {n0:.1e}")
        ts, ys = run_lifetime(A, qratio, nu, rho, n0, lam, gamma, qtrap)
        ax.plot(ts*1e9, ys, "-", color=c, lw=1)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(3e-3, 1.1)
    ax.set_xlabel("time (ns)"); ax.set_ylabel("PL (normalised)")
    ax.tick_params(direction="in", which="both")
    ax.legend(fontsize=8, title="points: data   line: global fit")
    p = os.path.join(out, f"{prefix}_global_fit.png"); fig.savefig(p, dpi=120)
    print("saved", p)


def profile(data, pbest, out, prefix, npts=19):
    SSR_min, N = ssr(pbest, data); s2 = SSR_min/(N-6)
    print(f"SSR_min={SSR_min:.4e} N={N} s^2={s2:.3e}")
    specs = [(0, "nu  (s^-1)", np.linspace(7, 10, npts)),
             (2, "lam  (s^-1)", np.linspace(7, 9, npts)),
             (3, "gamma  (s^-1)", np.linspace(5, 11, npts+6)),
             (1, "rho  (cm^3 s^-1)", np.linspace(-18, -9, npts))]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (idx, lab, grid) in zip(axes.ravel(), specs):
        prof = np.array([_refit_fixing(idx, v, pbest, data)[0] for v in grid]) - SSR_min
        ax.plot(grid, prof, "o-", color=CB[0])
        ax.axhline(s2, color="0.4", ls=":", label="s^2 (~1 sigma)")
        ax.axvline(pbest[idx], color=CB[2], ls="--", lw=1)
        ax.set_xlabel("log10 " + lab); ax.set_ylabel("SSR - SSR_min")
        ax.set_ylim(-0.05*max(8*s2, 1), max(8*s2, 1.3*np.nanpercentile(prof, 35)+1e-9))
        ax.tick_params(direction="in"); ax.legend(fontsize=8)
        rng = float(np.nanmax(prof))
        print(f"  {NAMES[idx]:6s} max(SSR-SSR_min) over scan = {rng:.3e}"
              f"  -> {'determined' if rng > 10*s2 else 'degenerate (flat)'}")
    p = os.path.join(out, f"{prefix}_profile_ci.png"); fig.savefig(p, dpi=120)
    print("saved", p)


def tradeoff(data, pbest, out, prefix, npts=25):
    A = data[0]; nu_eff = 10**pbest[0]
    grid = np.linspace(-11, -8, npts)
    ssr_v, nu_v, lam_v, gam_v = [], [], [], []
    for lr in grid:
        fun, x, free = _refit_fixing(1, lr, pbest, data)
        ssr_v.append(fun); nu_v.append(10**x[free.index(0)])
        lam_v.append(10**x[free.index(2)]); gam_v.append(10**x[free.index(3)])
    ssr_v, nu_v, lam_v, gam_v = map(np.array, (ssr_v, nu_v, lam_v, gam_v))
    rho = 10**grid; cross = np.log10(nu_eff/A)
    flat = grid < cross
    print(f"  over the flat region: lam spread "
          f"{100*(lam_v[flat].max()/lam_v[flat].min()-1):.2f}%, "
          f"gamma spread {100*(gam_v[flat].max()/gam_v[flat].min()-1):.2f}%")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    ax[0].plot(grid, ssr_v, "o-", color=CB[0]); ax[0].axvline(cross, color="0.4", ls=":")
    ax[0].set_xlabel("log10 rho  (cm^3 s^-1)"); ax[0].set_ylabel("profiled SSR")
    ax[0].tick_params(direction="in")
    ax[1].plot(grid, nu_v, "o", color=CB[0], label="refit nu")
    ax[1].plot(grid, np.clip(nu_eff-rho*A, 0, None), "--", color="0.2",
               label="nu_eff - rho*A")
    ax[1].plot(grid, lam_v, "s", ms=3, color=CB[1], label="lam (refit)")
    ax[1].plot(grid, gam_v, "^", ms=3, color=CB[2], label="gamma (refit)")
    ax[1].axvline(cross, color="0.4", ls=":")
    ax[1].set_xlabel("log10 rho  (cm^3 s^-1)"); ax[1].set_ylabel("rate (s^-1)")
    ax[1].tick_params(direction="in"); ax[1].legend(fontsize=8)
    p = os.path.join(out, f"{prefix}_rho_tradeoff.png"); fig.savefig(p, dpi=120)
    print("saved", p)


def saha(data, pbest, out, prefix):
    A, Nexc = data[0], data[1]
    nu = 10**pbest[0]; rho = 1e-10
    n = np.logspace(15, 21, 400); x = 2/(np.sqrt(4*n/A+1)+1)
    exc = nu*(1-x); bim = rho*x*x*n
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax[0].loglog(n, exc, color=CB[0], lw=2, label="exciton  nu(1-x)")
    ax[0].loglog(n, bim, color=CB[1], lw=2,
                 label="bimolecular  rho x^2 n  (rho=1e-10)")
    ax[0].axvline(A, color="0.4", ls=":", label="n = A")
    ax[1].semilogx(n, exc/bim, color="0.2", lw=2)
    ax[1].axhline(nu/(rho*A), color=CB[2], ls="--", label="nu / (rho A)")
    ax[1].axvline(A, color="0.4", ls=":"); ax[1].set_ylim(0, 2*nu/(rho*A))
    for a in ax:
        for nx in Nexc:
            a.axvline(nx, color="0.8", lw=0.8)
        a.tick_params(direction="in", which="both"); a.set_xlabel("density n  (cm^-3)")
    ax[0].set_ylabel("per-carrier rate  (s^-1)"); ax[0].legend(fontsize=8, loc="lower right")
    ax[1].set_ylabel("exciton rate / bimolecular rate"); ax[1].legend(fontsize=8)
    p = os.path.join(out, f"{prefix}_saha_collapse.png"); fig.savefig(p, dpi=120)
    print("saved", p)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="TPLM folder (JSON + *_uJcm2.npy)")
    p.add_argument("--sample", default="n_inf")
    p.add_argument("--out", default="output/lifetime_identifiability")
    p.add_argument("--mode", default="all",
                   choices=["all", "fit", "profile", "tradeoff", "saha"])
    p.add_argument("--no-fit", action="store_true",
                   help="skip the optimiser; use the cached best fit "
                        "(for regenerating figures)")
    p.add_argument("--seed", type=int, default=50)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = load_data(args.data, args.sample)
    print(f"A = {data[0]:.3e} cm^-3;  {len(data[1])} fluences")

    if args.no_fit:
        pbest = CACHED_BEST.copy()
        print("using cached best fit")
    else:
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
    if args.mode in ("all", "saha"):
        saha(data, pbest, args.out, prefix)


if __name__ == "__main__":
    main()
