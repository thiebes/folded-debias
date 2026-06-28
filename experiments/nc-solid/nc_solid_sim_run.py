"""Generate a clean, high-walker-count NC-solid map from the re-implemented model.

Runs the Solari hopping KMC (`nc_solid_sim` (this folder)) at
a high walker count to produce a smooth, low-sampling-noise, subdiffusive density
map -- the clean ground truth that replaces de-noising Marc's finite-walker
output. The map carries no shape assumption and its sampling noise is far below
Marc's 50k-walker maps, so the per-frame contrast-to-noise reads off the true
peak.

The energetic disorder is set by ``kt_mev`` (lower = stronger). With the
figure-read size distributions the achievable exponent floors near 0.64; matching
the paper's 0.42-0.70 needs the SI Fig. S2 size distributions and the temperature
(see `nc_solid_sim.py` and Email 9). This run uses the best-achievable strong-
disorder setting to demonstrate a clean subdiffusive map end to end.

Run from the repository root::

    .venv/Scripts/python.exe scripts/nc_solid_sim_run.py [sample] [kt_mev] [N]

Writes ``output/nc_solid_sim_<sample>.npz`` (frames, variance, step, meta).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import nc_solid_sim as sim  # noqa: E402

OUTPUT = Path(__file__).resolve().parents[1] / "output"
OUTPUT.mkdir(exist_ok=True)

# best-achievable strong-disorder kt per sample (from the calibration scan;
# floors near alpha ~ 0.64, pending the SI Fig. S2 sizes / temperature)
KT_BY_SAMPLE = {"small": 2.0, "medium": 3.0, "large": 0.7}


def main() -> None:
    sample = sys.argv[1] if len(sys.argv) > 1 else "small"
    kt = float(sys.argv[2]) if len(sys.argv) > 2 else KT_BY_SAMPLE[sample]
    n_exc = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000
    box_nm, L_nm, nbin, n_steps, n_frames = 1500.0, 1500.0, 250, 16000, 20

    rng = np.random.default_rng(2025)
    land = sim.generate_landscape(sim.SAMPLES[sample], box_nm=box_nm, rng=rng)
    traj = sim.run_kmc(land, n_exc=n_exc, n_steps=n_steps, rng=rng,
                       sample_every=n_steps // n_frames, kt_mev=kt,
                       diffmap_nbin=nbin, diffmap_L_nm=L_nm)
    alpha, _ = sim.fit_alpha(traj.step, traj.variance_nm2)

    # cleanliness: intrinsic 1-D peak shot-CNR of the final frame (sqrt of the
    # peak's walker count), versus Marc's ~50k-walker maps.
    last = traj.frames[-1]
    prof = last.sum(axis=1)                      # project to 1-D
    peak_counts = prof.max() * n_exc             # frame normalised -> counts
    intrinsic_cnr = float(np.sqrt(peak_counts))

    out = OUTPUT / f"nc_solid_sim_{sample}.npz"
    np.savez(out, frames=traj.frames.astype(np.float32), step=traj.step,
             variance_nm2=traj.variance_nm2, mean_energy_mev=traj.mean_energy_mev,
             sample=sample, ligand=sim.SAMPLES[sample].ligand, kt_mev=kt,
             n_exc=n_exc, alpha=alpha, box_nm=box_nm, nbin=nbin)
    print(f"sample={sample} ({sim.SAMPLES[sample].ligand})  kt={kt} meV  "
          f"N={n_exc}  M={land.pos.shape[0]} crystals")
    print(f"  alpha={alpha:.3f}  MSD: {traj.variance_nm2[0]:.0f} -> "
          f"{traj.variance_nm2[-1]:.0f} nm^2")
    print(f"  intrinsic 1-D peak CNR = {intrinsic_cnr:.0f}  "
          f"(Marc's 50k maps ~ 24; higher N -> cleaner)")
    print(f"  saved {out.name} ({traj.frames.shape})")


if __name__ == "__main__":
    main()
