# wse2-strobescat

## Purpose

The poster's real-data panel: run the correction on real bulk-WSe2 stroboSCAT
diffusion data (Enrique Arévalo Rodríguez), reduced via Enrique's own pipeline,
and show a measurable truncation bias removed on a real measurement.

## Status

Validated (in-hand version). Poster-scope (Panel 6 real-data).

## Contents

**Canonical**

- `wse2_realdata_panel.py [n_scans]` -- reduces the raw long-delay scans
  (differential `ON/OFF-1`, sign fixed per delay by a spot ROI, width from a
  line-cut band, time-since-pump = -delay), builds a white-matched D-free
  calibration at the data geometry, and produces the panel.

**Exploratory diagnostics** (the reduction triage that led to the canonical panel)

- `realdata_diagnostics.py` -- the shared measurement registry + loaders
  (`MEASUREMENTS`, `load_measurement`); imported by the others.
- `realdata_triage.py`, `realdata_frame_ladder.py`, `realdata_sweep_views.py`,
  `realdata_early_cut_plots.py`, `realdata_enrique_pipeline.py` -- exploratory
  views over the candidate scans (kept for provenance; not poster figures).

## Data

Read-only from the pshare; see `DATASETS.md`. Nothing is committed except the
panel figure snapshots.

## Run

```bash
.venv/Scripts/python.exe experiments/wse2-strobescat/wse2_realdata_panel.py
.venv/Scripts/python.exe experiments/wse2-strobescat/wse2_realdata_panel.py 1   # single-scan illustration
```

First run builds and caches the white calibration (`output/calibration_wse2_white.npz`,
~10 min); instant after. Outputs to repo-root `output/`; committed snapshots are here.

## Key result + figure

Merging the 3 long scans keeps per-frame CNR >= 2.6 (above the floor); the
correction is ~0 at high-CNR frames and removes the inflated late-frame widths,
giving D -7% (naive 1.10 -> corrected 1.02 px^2/ns) -- a measurable truncation
bias on real data. A single un-averaged scan reaches CNR ~0.8-0.9 (below the
floor), where the correction over-subtracts catastrophically -- the concrete face
of the design rule. Figures: `wse2_realdata_panel.png` (locked), plus the 1- and
2-scan variants. Widths are in px^2 (pixel->um scale is an open question with
Enrique, Email 9).

## Depends on

`weighting_wall` (`calibration`, `truncation`, `weights`, `cnr`, `fit`). The
exploratory scripts share `realdata_diagnostics.py` (local sibling).

## See also

`docs/STATUS.md` (2026-06-19 WSe2 panel update), `docs/poster/POSTER_PLAN.md`
Panel 6, `docs/email_collaborators.md` (Enrique correspondence).
