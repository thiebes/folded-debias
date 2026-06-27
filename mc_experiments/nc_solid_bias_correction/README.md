# NC-solid noise bias and its correction (bundle)

Self-contained snapshot of the contrast-to-noise (CNR) bias in the measured MSD
exponent of disordered nanocrystal-solid exciton transport, and the folded-normal
correction. June 2026.

The script and docs here are **snapshot copies**; the canonical, maintained
versions live in the repo root (see the cross-references below). The figure
outputs and the high-statistics calibration are kept here because the repo's
`output/` is gitignored, so this folder is where they are version-controlled.

## The result

A standard frame-by-frame Gaussian-width MSD fit overestimates the diffusion
exponent alpha as the per-frame CNR falls: the bounded width fit is truncated at
zero, right-skewed at low CNR, and lifts the late-time MSD. On a realistic-width
Gaussian bed built from the Solari et al. exciton-hopping maps (C16 sample), the
apparent exponent reads about 0.09 too high at CNR_0 = 30 and climbs to the
ballistic ceiling (alpha = 2) below CNR_0 ~ 8. The folded-normal correction
subtracts a calibrated per-frame variance bias var_bias(CNR) before the slope fit;
it recovers the nominal exponent (alpha ~ 0.88) across the realistic measurement
regime, leaving a small residual (~0.01 at CNR_0 = 30). The bias only raises
alpha, so it masks subdiffusion but never manufactures it.

## Contents

- `nc_solid_poster_figure.png` / `.txt` -- the figure (naive vs corrected alpha
  against CNR_0, with a low-CNR ballistic-divergence inset) and its per-CNR_0
  table. **Not duplicated elsewhere** (output/ is gitignored).
- `calibration_nc_solid_gaussian.npz` -- the high-statistics (8000 runs/cell),
  D = 0 white-noise var_bias(CNR, sigma^2) calibration the correction uses, on a
  250-pixel grid matched to the bed (CNR 1-250, sigma^2 to 0.12). **Not
  duplicated elsewhere** (the working copy in output/ is gitignored; rebuilt on a
  many-core machine).
- `nc_solid_poster_figure.py` -- snapshot of `scripts/nc_solid_poster_figure.py`.
- `run_nc_solid_calibration.sh` -- snapshot of
  `scripts/run_nc_solid_calibration.sh`.
- `nc_solid_calibration_rerun.md` -- snapshot of
  `docs/nc_solid_calibration_rerun.md`.
- `nc_solid_bias_note.md` -- snapshot of `docs/nc_solid_bias_note.md` (technical
  note + figure caption).
- `alpha_fit_weighting_survey.md` -- snapshot of
  `docs/alpha_fit_weighting_survey.md` (how the field fits the MSD/alpha --
  unweighted least squares, frame trimming over weighting -- and the direction
  distinction: this truncation bias raises alpha; the single-particle-tracking
  localization bias lowers it).

## Regenerating

From the repo root, with numpy + scipy (+ matplotlib): set
`WEIGHTING_WALL_WORKERS` to the physical-core count, then run
`scripts/nc_solid_poster_figure.py`. Put `calibration_nc_solid_gaussian.npz` in
the repo's `output/` first. Rebuild the calibration following
`docs/nc_solid_calibration_rerun.md`.

## Parameters

Bed: Solari et al. CH3NH3PbBr3 NC-solid maps (C16 / hexadecylamine), 1500 nm
field, 20 frames over 20 ns, Gaussian-fit-per-frame ground truth, nominal
alpha ~ 0.88, small-diffusion-length regime L_D / FWHM ~ 0.14. White measurement
noise. Calibration: 8000 runs/cell on a 120 x 100 (CNR, sigma^2) grid. Figure:
64,000 realisations per CNR_0 (main panel), 4,000 (inset).
