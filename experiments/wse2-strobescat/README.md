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

## The CNR key on real data: estimated, not exact

The calibration (`calibrate_d_free`) is keyed on the exact per-frame CNR, which
simulation knows. This pipeline keys the correction on the CNR that `fft-cnr`
estimates from each frame, and that key sets three things at once: the
subtracted bias, the weight, and the floor cut. The combination was tested in
simulation in the fft-cnr paper (`../fft-cnr-paper`, case 27: Gaussian, Voigt,
and generalized-Gaussian p = 1.5 and p = 4 profiles, D 0.005 to 0.1, CNR_0 5 to
90, 300 runs per cell).

- **Where the per-frame key holds.** With the exact key, the corrected mean D
  is within 0.85 to 1.15 of the reference in every cell from CNR_0 5 to 8 at
  D = 0.02. With the estimated key, it holds only where every fitted frame stays
  well above the floor: from CNR_0 about 34 at D = 0.02 (about 55 for the
  heavy-tailed p = 1.5 profile, 13 for p = 4) and from CNR_0 about 21 at
  D >= 0.05. Below that, the pooled Gaussian mean at D = 0.02 over CNR_0 8 to 90
  is 1.9 (standard error 0.4).
- **Why.** Random key error of up to 0.2 of the CNR per frame is tolerated. The
  estimate is not random about the exact value near the floor: it reads 0.89 to
  0.97 of the exact CNR at CNR 1 to 2, scatters by 0.6 to 0.9 of the value below
  CNR 1, and reads about twice the exact value below CNR 0.5. Frames near the
  floor are therefore over-corrected, and frames below it are admitted and
  under-corrected. The key and the width also come from the same frame, and
  their errors are correlated (a frame whose CNR reads high has a width that
  reads low; correlation -0.1 to -0.3).
- **What does not fix it.** A stricter floor on the estimate (2 or 3)
  over-subtracts on every shape except p = 4, because the frames it keeps are
  the ones whose estimate reads high. Dropping frames whose CNR interval
  includes zero removes 0.31 to 0.45 of the frames at CNR 1 to 2, selectively,
  and still over-subtracts on the non-Gaussian profiles. A calibration keyed on
  the estimate transfers poorly between distributions of true CNR.
- **What works, not adopted.** A cross-fitted key (frame t's CNR predicted from
  the other frames by a robust straight-line fit of log CNR against time)
  matches the exact key from CNR_0 13 on all four shapes. It adds a decay-trend
  model with its own assumptions, so it is reported as a check of the cause and
  the pipeline keeps the per-frame estimate. Do not switch to it without a
  decision from Joseph.
- **On the measured scans the effect is small**: only 0.035 of the retained
  frames (18 included stroboSCAT scans) are below CNR 2.
- **fft-cnr version.** Use fft-cnr 0.3.3 or later, the first PyPI release
  that carries commit 69cd361 (0.3.2 was tagged but not published). Earlier
  versions return occasional runaway
  generalized-Gaussian fits on frames below CNR 1 (CNR of tens to thousands;
  about 0.015 of the frames below CNR 0.5); two of the 187 stroboSCAT scans pass
  the triage screen only with the fix, because a runaway frame had made their
  merged D negative. `pyproject.toml` requires `fft-cnr>=0.3.3`, and `uv.lock`
  pins the PyPI release with its hash.

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

`FOLDED_DEBIAS_WORKERS=1` (serial) is the safe default for the calibration
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
  physical um^2 (0.060 um per px): naive D 0.0050 -> corrected 0.0047 um^2/ns
  (ratio 0.95; earlier quoted as 0.0057 -> 0.0054 with the superseded
  0.00414 um^2/px^2 scale, rescaled by 0.870) -- a
  measurable ~5% upward bias concentrated in the low-CNR late frames (2-7 ns)
  normally discarded. Rendered by `render_realdata_panel.py`. **Supersedes** the
  WSe2 `0.020` candidates (a batch-offset artifact -- goes negative under correct
  de-parity; do not use).
- Per-measurement views: `diagnostics/<measurement>/msd_enrique_pipeline.png`,
  `msd_deparity.png`; summary in `diagnostics/index.md`. Reported in px^2 or
  um^2; the pixel is 0.060 um at the sample for every scan (Enrique,
  2026-10-20; `UM_PER_PX` in `realdata_diagnostics.py`). Do not take the scale
  from his `X(um)` array (it steps by 0.0036 = 0.060^2) or from the ratio of his
  fitted sigma (um) to our px fit (fit-dependent, 0.065 to 0.089 um/px). The
  unit-free ratio is the headline.
- Excluded scans: `WSe2/Power_var/250904_skong/5MHz/bad` (spot out of focus;
  Enrique advises not to use it) appears in `diagnostics/triage.csv` and
  `triage_deparity.csv` as a screened row only and must not be used. In
  `WSe2/test/mostly_tests/250603_2` the first delay (2.55 ns, acquired last,
  so the loaders make it a one-frame sweep of its own) is unusable; drop it
  before any fit. The triage rows for that scan include it.

## Depends on

`folded_debias` (`calibration`, `truncation`, `weights`, `cnr`, `fit`,
`noise`). `fft-cnr` for per-frame CNR. The diagnostic scripts share
`realdata_diagnostics.py` (local sibling).

## See also

`docs/poster/POSTER_PLAN.md` Panel 6,
`docs/email_collaborators.md` (Enrique).
fft-cnr note: the per-frame CNR must be measured on the spot window, not the full
line-cut (off-spot structure corrupts the spectral CNR at low signal) --
`thiebes/fft-cnr` issue #7.
