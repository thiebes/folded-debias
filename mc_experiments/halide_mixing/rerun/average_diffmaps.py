#!/usr/bin/env python3
"""Average the per-instance halide re-run outputs into one clean map.

Each instance (a separate random realisation) writes a per-frame-normalised
``diffmap.x_<frac>.dat`` and a particle ``variance.x_<frac>.dat``. Averaging K
instances gives the disorder-averaged density with K-fold more effective
statistics, so the sampling noise falls as 1/sqrt(K * N_inst) and no shape-fitting
de-noise is needed. The diffmaps are accumulated one at a time (a 500x500x601
running sum is ~1.2 GB in float64; the per-instance text file is read and then
released), so memory stays bounded regardless of K.

Usage:
    python3 average_diffmaps.py <frac> <inst_dir> [<inst_dir> ...]
Writes (in the current directory):
    diffmap.x_<frac>.avg.npy    -- (n_frames, nx, nx) float32 clean map
    variance.x_<frac>.avg.dat   -- averaged (time, variance) carrier truth

Note: reads ``diffmap_nx`` from each instance's parameter file to reshape; if the
diffmap holds a different number of frame blocks than variance.dat has rows
(seen as 605 vs 601 in the shared data), it reshapes by nx*nx and reports the
frame count so the mismatch can be reconciled before downstream use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def read_nx(param_path: Path) -> int:
    for line in param_path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "diffmap_nx":
            return int(parts[1])
    raise ValueError(f"diffmap_nx not found in {param_path}")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit("usage: average_diffmaps.py <frac> <inst_dir> [<inst_dir> ...]")
    frac = sys.argv[1]
    inst_dirs = [Path(p) for p in sys.argv[2:]]

    dsum: np.ndarray | None = None
    vsum: np.ndarray | None = None
    nx = None
    n = 0
    for d in inst_dirs:
        dpath = d / f"diffmap.x_{frac}.dat"
        vpath = d / f"variance.x_{frac}.dat"
        ppath = d / f"parameters.x_{frac}.dat"
        if not (dpath.exists() and vpath.exists()):
            print(f"skip {d}: missing diffmap or variance")
            continue
        if nx is None:
            nx = read_nx(ppath)
        flat = np.fromfile(str(dpath), sep=" ")        # whitespace text -> 1-D
        v = np.loadtxt(vpath)
        dsum = flat if dsum is None else dsum + flat
        vsum = v if vsum is None else vsum + v
        n += 1
        print(f"added {d}  (n={n}, {flat.size} values)")
        del flat

    if n == 0:
        sys.exit("no instances found")
    dsum /= n
    vsum /= n

    n_frames = dsum.size // (nx * nx)
    if dsum.size != n_frames * nx * nx:
        print(f"WARNING: {dsum.size} values not divisible by nx*nx={nx*nx}; "
              "check the diffmap layout before use")
    cube = dsum[: n_frames * nx * nx].reshape(n_frames, nx, nx).astype(np.float32)
    np.save(f"diffmap.x_{frac}.avg.npy", cube)
    np.savetxt(f"variance.x_{frac}.avg.dat", vsum)
    print(f"averaged {n} instances -> diffmap.x_{frac}.avg.npy "
          f"(frames={n_frames}, nx={nx}; variance rows={vsum.shape[0]})")


if __name__ == "__main__":
    main()
