# subdiffusive-alpha

## Purpose

Clean single-population synthetic test that the correction de-biases the
anomalous exponent alpha, not only D, on the `AnomalousExperiment` bed
(`sigma^2(t) = sigma0^2 + 2 D t^alpha`).

## Status

Validated. Poster-scope (the alpha-recovery point, Panel 6).

## Contents

- `subdiffusive_alpha.py [profile_model] [--quick] [--floor F]` -- sweeps true
  alpha in {0.6, 0.8, 1.0, 1.2} x CNR0 and fits `MSD = 2 D t^alpha`, comparing
  the naive all-frames fit against folded-normal + the per-frame CNR floor cut.

## Data

None. Fully synthetic. Caches `calibration_subdiffusive[_<model>].npz` (also
reused by multiexp-decay and coverage-interval).

## Run

```bash
.venv/Scripts/python.exe experiments/subdiffusive-alpha/subdiffusive_alpha.py
```

Set `WEIGHTING_WALL_WORKERS=0`; ~10 min first run, ~3 min cached. Outputs to
repo-root `output/`; committed snapshots are here.

## Key result + figure

The naive fit pins at the ballistic alpha = 2.0 for CNR0 <~ 13; the correction
recovers true alpha to `|alpha_rec - alpha_true| ~ 0.02-0.10` for CNR0 >= 20, and
D to `<D>/D_true ~ 0.93-1.05`. Figure: `subdiffusive_alpha.png`; table
`subdiffusive_alpha.txt`.

## Depends on

`weighting_wall` (`AnomalousExperiment`). No sibling-folder helpers.

## See also

`docs/STATUS.md` (2026-06-14 subdiffusive-alpha update),
`docs/poster/POSTER_PLAN.md` Panel 6.
