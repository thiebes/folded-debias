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
.venv/Scripts/python.exe experiments/poc-bias-correction/poc_demo.py generalized_gaussian
```

Set `FOLDED_DEBIAS_WORKERS=0` (all cores) to speed up the Monte Carlo. Outputs
are written to the repo-root `output/` directory; the committed snapshots in this
folder are the current results.

## Key result + figure

Mean `|mean(D_hat)/D_true - 1|` over the CNR sweep: OLS 10.3, inverse-variance
0.31, folded-normal 0.12 (Gaussian); folded-normal 0.106 (Voigt), 0.096
(generalized-Gaussian). Figures: `poc_demo.png`, `poc_demo_voigt.png`,
`poc_demo_generalized_gaussian.png` (with matching `.txt` result tables).

## Depends on

`folded_debias` (`MODELS`, `Experiment`, `evaluate`, `sweep_stats`,
`TruncationModel`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (folded-normal derivation), `docs/STATUS.md`,
`docs/poster/POSTER_PLAN.md` Panels 2 and 5.
