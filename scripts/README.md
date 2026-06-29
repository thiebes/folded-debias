# `scripts/` -- shared tooling and poster assembly

Per-bed experiment scripts now live with their experiment under `experiments/`
(one folder per set; see [`docs/REPO_MAP.md`](../docs/REPO_MAP.md) for the
inventory and [`docs/EXPERIMENT_CONVENTIONS.md`](../docs/EXPERIMENT_CONVENTIONS.md)
for the rulebook). What remains in `scripts/` is shared across beds or builds the
poster.

All scripts run from the repository root with `.venv/Scripts/python.exe` on
Windows. Outputs go to `output/` with matching base names.

## Shared calibration tooling

- **`build_calibration.py --tag <name> ...`** -- builds a geometry-specific joint
  `var_bias(CNR, sigma^2)` calibration for an arbitrary experiment (real or
  simulated): pass the pixel grid, sigma range, CNR range, profile model, and MC
  budget. Saves `output/calibration_<tag>.npz`, loadable with
  `CalibrationGrid.load()`. Used to build the Marc-geometry calibrations the
  realistic-transport beds (`experiments/mn-trapping/`) consume. Wall time
  ~2.5-4 h for a 100x100-cell, 1000-runs/cell build.
- **`plot_calibration.py <calibration.npz>`** -- diagnostic heatmaps of a saved
  joint calibration (the `var_bias` and `rel_variance` surfaces, with the Marc-data
  trajectory overlaid when applicable).
- **`run_calibration.sh`** -- the generic calibration runner the per-bed runners
  call (e.g. `experiments/cspbbr-strobescat/run_cspbbr_calibration.sh`).

## Poster assembly

- **`poster_figures.py`** -- regenerates the data-driven poster figures and the
  rough panel schematics in one place (Panel 1 schematic, Panel 2A wall-and-fold
  schematic, Panel 4 CRLB, Panel 5 corrected bias + floor inset). Requires the
  sibling `curved-text` package. (Pending convention cleanup: this renders panels
  from several experiments; splitting each panel's render into its experiment
  folder is a follow-up -- see REPO_MAP.)
- **`poster_fftcnr_dense.py`** -- the dense fft-cnr recovery-ratio panel (Panel 4
  of the CNR-centric spine).
- **`build_poster.py`** -- assembles the poster draft (`docs/poster/`). Pulls each
  panel from its experiment folder rather than a shared bin.

## Legacy / deprecated

`scripts/legacy/` holds scripts that documented development decisions or explored
avenues that did not pan out. They run but are not part of the current pipeline:

- **`legacy/validate_d_free.py`** -- validation of the experimental joint
  `var_bias(CNR, sigma^2)` calibration (not competitive; kept as the record).
- **`legacy/diagnose_calibration.py`** -- one-off 1-D-vs-2-D calibration comparison.
- **`legacy/explore_regime.py`** -- early exploratory D x CNR0 sweep.
- **`legacy/dense_calibration_and_test_robust.py`** -- robust multiplicative
  scheme (inert at the tested CNR0; kept as the negative result).

**Note:** the `legacy/` scripts set their import path with `../../src`; run them
with `PYTHONPATH=src` if a path is stale. Their outputs live in `output/legacy/`.
