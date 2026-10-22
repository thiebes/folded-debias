# mn-trapping

## Purpose

Run the correction on Marc Meléndez's Mn-doped reaction-diffusion diffmaps
(Magdaleno et al. 2025) -- genuinely non-Fickian transport with the model
author's own `sigma^2(t)` as ground truth -- across three estimators, separating
the correction residual (vs the noiseless fit) from the shape gap (noiseless fit
vs true second moment).

## Contents

- `marc_data.py` -- loader for the `.dat` diffmaps (reads `data/`).
- `diffusion_map.py` -- diffmap container / PSF-convolution helpers.
- `marc_diffmap_validation.py` -- the correction across Gaussian / Voigt /
  generalized-Gaussian on all four doping cases, plus the `MomentModel`
  comparator; separates correction residual from shape gap.
- `marc_mn_trapping_plan_a.py` -- per-frame variance recovery on the 0.51% Mn
  diffmap (Gaussian on the as-is map; Voigt on the PSF-convolved map).

## Data

Collaborator `.dat` diffmaps in the local (gitignored) `data/` dir; see
`DATASETS.md`. Needs the Marc-geometry calibrations built by
`scripts/build_calibration.py` (`calibration_marc_{gaussian,voigt,gengauss}_filtered.npz`).

## Run

```bash
.venv/Scripts/python.exe experiments/mn-trapping/marc_diffmap_validation.py
.venv/Scripts/python.exe experiments/mn-trapping/marc_mn_trapping_plan_a.py
```

Outputs to repo-root `output/`; committed snapshots are here.

## Key result + figure

The single-shape fits + folded-normal remove the noise bias but not the shape gap
(corrected residual vs carrier moment ~0.03-0.08 um^2); the bare moment estimator
closes the shape gap but is catastrophically noise-fragile, so the practical
carrier-recovery path is the forward fit. Figures: `marc_diffmap_validation.png`,
`marc_mn_trapping_plan_a.png`, plus the calibration diagnostics.

## Depends on

`folded_debias` (`CalibrationGrid`, `fit`, `models`, `truncation`, `weights`).
Local sibling loaders `marc_data.py`, `diffusion_map.py`. The calibrations come
from `scripts/build_calibration.py` (shared tooling).

## See also

`docs/VALIDATION_DESIGN.md`.
