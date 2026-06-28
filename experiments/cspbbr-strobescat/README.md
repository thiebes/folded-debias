# cspbbr-strobescat -- CsPbBr3 single-crystal diffusion, real-data correction example

**Status:** work-in-progress; adjacent-project (Upasana's data, the
Antonella/Carlos diffusion paper), not the poster's real-data panel -- that is
`experiments/wse2-strobescat/`. Kept here as an end-to-end real-data application
of the correction. Data on the pshare; see `DATASETS.md`.

End-to-end application of the truncation-bias correction to a real measurement:
CsPbBr3 single-crystal transient-PL-microscopy diffusion (Upasana's data; the
Antonella/Carlos diffusion paper). The late-time MSD inflates as the per-frame
CNR collapses; the correction removes that bias before the D fit.

The noise is white (constant-floor), not shot-limited -- a raster-scanned APD
measurement whose photon-transfer fit comes out floor-dominated -- so the
calibration uses the gaussian model. See `docs/calibration_runs.md`.

## All fluences at once (recommended)

`run_all.py` does the whole thing across every fluence subfolder: reduces each,
derives its geometry from the fresh reduction, builds one calibration per unique
pixel grid (shared across fluences on that grid, since the gaussian lookup is
fluence-independent), applies the correction, and writes a D-vs-fluence summary
(`cspbbr_D_vs_fluence.csv` and `.png`). The heavy calibration honours `--workers`.

```bash
python experiments/cspbbr-strobescat/run_all.py \
  --sample-root "P:/Upasana/CsPbBr_SingleCrystal/Sample_1" \
  --out-dir output/cspbbr --workers 0 --n-runs 8000
```

`--stages` resumes a partial run (e.g. `--stages apply,summary` after the
calibrations are built). The run machine must be able to read the raw data at
`--sample-root`.

## Single fluence (manual)

1. **Reduce** raw acquisition CSVs to per-frame profiles, fitted variance, and
   CNR (mirrors the lab's reduction; per-frame fit and CNR are this project's):

   ```bash
   python experiments/cspbbr-strobescat/reduce.py \
     --data "P:/Upasana/CsPbBr_SingleCrystal/Sample_1/13.47 uJ_Fluence_19_May_2026" \
     --n-files 3 --max-var 6.25 --out output/cspbbr_13p47.npz
   ```

2. **Calibrate** the `(CNR, sigma^2)` bias lookup for this geometry (heavy; run
   on the multi-core box -- see `docs/cspbbr_calibration_run.md`):

   ```bash
   PYTHON=.venv-linux/bin/python bash scripts/run_cspbbr_calibration.sh
   ```

3. **Apply** the correction and compare naive vs corrected D:

   ```bash
   python experiments/cspbbr-strobescat/apply_correction.py \
     --reduced output/cspbbr_13p47.npz \
     --calibration output/calibration_cspbbr_gaussian.npz
   ```

## Consistency requirements

The bias lookup is conditional on the measurement geometry, so the reduction and
the calibration must agree on:

- the **pixel grid** (`--n-pixels`, `--half-window` in the calibration must match
  the reduced profiles' grid; `apply_correction.py` warns if they disagree), and
- the **fit variance cap** (`reduce.py --max-var` must equal the calibration's
  `--max-var`).

Raw data is not committed (it is the collaborators'); point `--data` at the
acquisition folder on the share.
