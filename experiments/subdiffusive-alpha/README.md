# subdiffusive-alpha

## Purpose

Clean single-population synthetic test that the correction de-biases the
anomalous exponent alpha, not only D, on the `AnomalousExperiment` bed
(`sigma^2(t) = sigma0^2 + 2 D t^alpha`).

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

Set `FOLDED_DEBIAS_WORKERS=0`; ~10 min first run, ~3 min cached. Outputs to
repo-root `output/`; committed snapshots are here.

## Key result + figure

The naive fit pins at the ballistic alpha = 2.0 for CNR0 <~ 13. Over CNR0 21
to 90 (2000 runs per cell, Monte Carlo SE from 200 resamples of the
realisations) the mean corrected |alpha - alpha_ref| is 0.030 +/- 0.020,
0.018 +/- 0.022, 0.046 +/- 0.023, and 0.100 +/- 0.041 for alpha = 0.6, 0.8, 1.0,
and 1.2 (naive 0.18 to 0.30): consistent with zero for 0.6 and 0.8, about 2 SE
for 1.0 and 1.2. Corrected D is 0.84 to 1.05 of D_ref there. The floor cut drops
no frame at CNR0 >= 21, so those numbers isolate the subtraction; the table's
naive + cut arm separates the cut from the subtraction below that (for
alpha = 0.6 the cut alone moves the naive exponent from 2.0 to 0.92 to 1.13). Figure: `subdiffusive_alpha.png`; table
`subdiffusive_alpha.txt`.

## Depends on

`folded_debias` (`AnomalousExperiment`). No sibling-folder helpers.

## See also

`experiments/nc-solid/` (the same exponent bias on a realistic transport bed).
