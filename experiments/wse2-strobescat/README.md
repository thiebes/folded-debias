# wse2-strobescat

## Purpose

Real-data truncation-bias diagnostics on Enrique Arévalo Rodríguez's stroboSCAT
diffusion scans (bulk WSe2 and MoS2/MoSe2 alloys). Reduce the scans through
Enrique's own pipeline, apply the project's folded-normal correction, and
establish honestly what the correction does on real measurements.

## Status

Validated / work-in-progress. Poster-scope (the real-data panel and the
"benefit + limitation" figures).

## Key findings

- **De-parity is mandatory.** Every measurement is a merge of two interleaved
  delay sweeps (a coarse and a fine sweep) with *different* sigma^2(0); a single
  global baseline injects a spurious slope. Each sweep must be re-zeroed to its
  own t0 before fitting D. A global offset is slope-invariant; only this
  per-sweep re-zeroing legitimately changes D.
- **The two sweeps often disagree** (the coarse sweep diffuses; the fine sweep is
  flat or anti-diffuses), so some "diffusion" in the naive global-baseline fits
  was a batch-offset artifact. WSe2 `Power_var/250224/0.020` and
  `mixed_tmdc/50_50/250702` go degenerate (negative D) under correct baselining;
  other WSe2 power dates are clean (both sweeps agree).
- **On real data the correction is small where the data is trustworthy.**
  Reproducing Enrique's exact fit (t <= 2 ns, forced origin, inverse-variance),
  the corrected/naive D ratio is ~1.00 for high-CNR measurements and ~0.95-0.99
  where the fit reaches CNR ~1.4-2.5 -- the effect grows monotonically as the fit
  reaches lower CNR, exactly as the theory predicts. The large bias lives at low
  CNR, which is also where real frames become individually unreliable.
- **The bias-magnitude claim belongs to simulation; real data is validation.**
  The correction debiases the *mean* (and hence D, a mean-trend slope), not every
  individual near-floor frame. The simulated recovery demo carries the
  "recovers the true width" claim (truth known); the real data shows the method
  returns stable, self-consistent D and a CNR-dependence matching theory.

## Contents

- `realdata_diagnostics.py` -- the shared registry (`MEASUREMENTS`), loaders
  (`load_measurement`, sweep detection), per-frame CNR, calibration build/cache,
  the folded-normal correction, de-parity, the sustained-tail truncation, and the
  per-measurement diagnostic outputs. Imported by the others.
- `realdata_enrique_pipeline.py` -- reproduces Enrique's exact D fit (t <= 2 ns,
  forced origin, inverse-variance) and overlays the correction; per-measurement
  `msd_enrique_pipeline.png`.
- `realdata_sweep_views.py` -- per-sweep and de-paritied-merged MSD views.
- `realdata_triage.py` -- de-parity triage over all analysis-bearing
  measurements; ranks clean diffusers (both sweeps agree); `triage_deparity.csv`.
- `realdata_frame_ladder.py` -- the CNR ladder of individual profile frames
  (benefit + limitation).
- `realdata_early_cut_plots.py` -- early-time (manual cutoff) MSD plots.
- `render_recovery_demo.py` -- the simulated recovery figures (benefit frame +
  distribution), where the truth is known.
- `wse2_realdata_panel.py [n_scans]` -- the original locked WSe2 panel (raw-scan
  reduction: differential, sign-by-ROI, line-cut band, time = -delay).

## Data

Read-only from the pshare; see `DATASETS.md`. Raw collaborator scans stay on the
pshare (never committed). All *generated* outputs are tracked here under
`diagnostics/` (per-measurement folders, `_frame_ladder/`, `_calib/` calibration
caches, `index.md`, `METHOD.md`, `triage*.csv`).

## Run

```bash
# per-measurement diagnostics (builds + caches calibrations under diagnostics/_calib)
.venv/Scripts/python.exe experiments/wse2-strobescat/realdata_diagnostics.py <name>
# Enrique-pipeline reproduction + correction, all registered measurements
.venv/Scripts/python.exe experiments/wse2-strobescat/realdata_enrique_pipeline.py
# the CNR frame ladder and the simulated recovery demo
.venv/Scripts/python.exe experiments/wse2-strobescat/realdata_frame_ladder.py
.venv/Scripts/python.exe experiments/wse2-strobescat/render_recovery_demo.py
```

`WEIGHTING_WALL_WORKERS=1` (serial) is the safe default for the calibration
builds; `=0` (all cores) is faster but the Windows process pool can break under
CPU contention.

## Key result + figures

- Benefit (simulation): `diagnostics/_frame_ladder/0_simulated_benefit.png` and
  `0b_simulated_distribution.png` -- the true width recovered from frames that
  look unusable; the ensemble distribution recentered on truth.
- Limitation: `diagnostics/_frame_ladder/5_cnr00.5_*` -- below-floor
  over-subtraction.
- **Canonical real-data panel (poster station 8):
  `figure_candidates/station8_realdata-MoSSe-26_32.png`** (caption beside it) --
  the correction on the clean **MoSSe alloy** `26_32/250716`
  (sample `250716_MoSSe_mystery`, **not WSe2**), de-parity merged, full range, in
  physical um^2: naive D 0.0057 -> corrected 0.0054 um^2/ns (ratio 0.95) -- a
  measurable ~5% upward bias concentrated in the low-CNR late frames (2-7 ns)
  normally discarded. Rendered by `render_realdata_panel.py`. **Supersedes** the
  WSe2 `0.020` candidates (a batch-offset artifact -- goes negative under correct
  de-parity; do not use).
- Per-measurement views: `diagnostics/<measurement>/msd_enrique_pipeline.png`,
  `msd_deparity.png`; summary in `diagnostics/index.md`. Reported in px^2 or the
  sample um^2 (the px->um scale is well-defined for 26_32 but an open question in
  general, Enrique Email 9); the unit-free ratio is the headline.

## Depends on

`weighting_wall` (`calibration`, `truncation`, `weights`, `cnr`, `fit`,
`noise`). `fft-cnr` for per-frame CNR. The diagnostic scripts share
`realdata_diagnostics.py` (local sibling).

## See also

`docs/STATUS.md`, `docs/poster/POSTER_PLAN.md` Panel 6,
`docs/email_collaborators.md` (Enrique).
fft-cnr note: the per-frame CNR must be measured on the spot window, not the full
line-cut (off-spot structure corrupts the spectral CNR at low signal) --
`thiebes/fft-cnr` issue #7.
