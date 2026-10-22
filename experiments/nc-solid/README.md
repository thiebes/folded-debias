# nc-solid

## Purpose

The realistic-transport result: on a Gaussian-width bed built from Marc
Meléndez's Solari et al. disordered nanocrystal-solid exciton-hopping maps, a
standard frame-by-frame Gaussian-width MSD fit inflates the apparent
subdiffusive exponent alpha toward the ballistic ceiling as the per-frame CNR
falls, and the folded-normal correction recovers the nominal exponent.

Quantitative thresholds on this bed are soft (the maps' finite-walker sampling
noise), so the result is stated qualitatively; the precise thresholds come from
the synthetic beds.

## Contents

- `nc_data.py` -- loader for the NC-solid maps (reads the pshare; `denoise=True`
  reconstructs the smooth Gaussian per frame before measurement noise).
- `nc_solid_sim.py` -- the Solari hopping-KMC simulator.
- `nc_solid_sim_run.py` -- runs the KMC simulator (`nc_solid_sim`).
- `nc_solid_alpha_vs_cnr.py` -- naive vs corrected alpha against CNR0 (the core
  compute; reused by the figure script).
- `nc_solid_poster_figure.py` -- the figure (naive vs corrected alpha,
  ballistic-divergence inset). Reuses `nc_solid_alpha_vs_cnr` and `nc_data`.
- `nc_solid_validation.py` -- scores D and alpha against the carrier moment.
- `run_nc_solid_calibration.sh` -- the high-statistics calibration runner.
- `calibration_nc_solid_gaussian.npz` -- the committed high-statistics
  (8000 runs/cell, D=0 white-noise) var_bias calibration the figure uses (output/
  is gitignored, so the version-controlled copy lives here).
- `nc_solid_poster_figure.png` / `.txt` -- the committed figure and per-CNR0 table.
- `nc_solid_poster_figure_caption.md` -- the figure caption.

## Data

NC-solid maps on the pshare (`P:/Marc/shared_data/disordered_perovskite_nanocrystals_solids`);
see `DATASETS.md`. No local copy. The committed calibration `.npz` is the only
large derived artifact kept in-repo.

## Run

```bash
# regenerate the figure (needs the calibration in output/ first):
cp experiments/nc-solid/calibration_nc_solid_gaussian.npz output/
FOLDED_DEBIAS_WORKERS=0 .venv/Scripts/python.exe experiments/nc-solid/nc_solid_poster_figure.py
```

Rebuild the calibration with `run_nc_solid_calibration.sh` on a many-core Linux
or WSL machine (usage in the script header). Other scripts run the same way from
the repo root.

## Key result + figure

Naive alpha reads ~0.09 high at CNR0 = 30 and climbs to the ballistic ceiling
(alpha = 2) below CNR0 ~ 8; the folded-normal correction recovers the nominal
alpha ~ 0.88 across the measured regime (residual ~0.01 at CNR0 = 30). The bias
only raises alpha, so it can mask subdiffusion but cannot manufacture it. Figure:
`nc_solid_poster_figure.png`; table `nc_solid_poster_figure.txt`.

## Depends on

`folded_debias` (`GaussianModel`, `noise.WhiteNoise`, `truncation`, `validate`,
`weights`). Local sibling modules `nc_data.py`, `nc_solid_sim.py`,
`nc_solid_alpha_vs_cnr.py`. Optional `curved_text` (the figure's curved label;
`pip install -e ../curved-text`).

## See also

`nc_solid_bias_note.md` (technical note), `docs/VALIDATION_DESIGN.md`,
`experiments/subdiffusive-alpha/` (exponent recovery on a synthetic bed).
