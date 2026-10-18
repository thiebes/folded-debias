# poc-bias-correction

## Purpose

The core proof-of-concept: show the low-CNR truncation bias in the extracted
diffusion coefficient D, and that the folded-normal correction removes it. One
two-panel figure per profile model (truncation diagnostic; bias curves
`mean(D_hat)/D_true` vs CNR0 for OLS, inverse-variance, and folded-normal).

## Status

Validated. Poster-scope (source for Panels 2 and 5).

## Contents

- `poc_demo.py [profile_model]` -- runs the two-panel proof-of-concept for a
  single profile model (`gaussian` default, or `voigt` / `generalized_gaussian`).

## Data

None. Fully synthetic (`folded_debias` Monte Carlo); no pshare datasets.

## Run

```bash
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py voigt
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py generalized_gaussian:1.5
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py generalized_gaussian:4
```

The generalized-Gaussian takes its shape exponent in the spec; the bare
`generalized_gaussian` is p = 2, a Gaussian profile.

Set `FOLDED_DEBIAS_WORKERS=0` (all cores) to speed up the Monte Carlo. Outputs
are written to the repo-root `output/` directory; the committed snapshots in this
folder are the current results.

## Key result + figure

D = 0.02, 400 runs per CNR_0, each table with the folded-normal cell's Monte
Carlo SE. Over CNR_0 13 to 90 the folded-normal mean `|mean(D_hat)/D_ref - 1|`
is 0.052 (Gaussian), 0.056 (Voigt), 0.048 (generalized-Gaussian p = 1.5), and
0.040 (p = 4). The CNR_0 5 and 8 cells depend on the calibration draw (this
script's calibration is 7 CNR_0 points x 400 runs): the Gaussian CNR_0 5 cell is
3.7 +/- 1.7 here, so the mean over the whole sweep (0.47 Gaussian, 0.30 Voigt,
0.47 p = 1.5, 0.095 p = 4) is not a stable headline. OLS is inflated 46-fold at
CNR_0 5 for the Gaussian; the p = 4 fit is barely biased (OLS mean 0.39) and
reads low, not high, at CNR_0 <= 8.

Tables: `poc_demo_gaussian.txt`, `poc_demo_voigt.txt`,
`poc_demo_generalized_gaussian_p1.5.txt`, `poc_demo_generalized_gaussian_p4.txt`.
`poc_demo_generalized_gaussian.txt` and its figure are an older p = 2 run (a
Gaussian profile with the generalized-Gaussian fitter's tighter `var_max`), kept
for reference. Figures: `poc_demo*.png`.

## Depends on

`folded_debias` (`model_from_spec`, `Experiment`, `evaluate`, `sweep_stats`,
`TruncationModel`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (folded-normal derivation), `docs/STATUS.md`,
`docs/poster/POSTER_PLAN.md` Panels 2 and 5.
