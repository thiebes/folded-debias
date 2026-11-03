# crlb-precision

## Purpose

The Cramer-Rao lower bound on D: the floor that no unbiased estimator can beat
(`Var(D_hat) >= CRLB`). Separates the removable artefact (bias) from the
irreducible cost (variance), and shows that at low CNR a single measurement
cannot pin D regardless of estimator. Replication is therefore required, and
only an unbiased estimator converges to the input D.

## Contents

- `crlb.py` -- builds the Fisher information matrix for the simulated Gaussian
  measurement model and returns `[F^-1]_DD`, the CRLB on D, vs CNR0.
- `efficiency_check.py generate|analyse` -- measures the corrected D's spread
  against sqrt(CRLB) on Gaussian, Voigt, p = 1.5, and p = 4 (numerical Fisher
  matrix from each model's profile; D-free calibration per shape; 1000 runs
  per cell). Arms: library weights, inverse-CRLB weights, a robust spread, and
  a direct fit of the corrected widths against time. Seeds are offset by 10^6
  per shape, so the shapes see independent noise. `generate` is about 70 min
  on 8 cores (cached in `output/crlb-efficiency/`).
- `render_crlb_panel.py` -- renders the CRLB figure
  (`poster_panel4_crlb.png`) from `crlb.py`; uses the shared plotting primitives
  in `scripts/poster_style.py`.
- `precision_limit.md` -- the derivation and discussion.

## Data

None. Fully synthetic (uses `width_debias.simulate` geometry).

## Run

```bash
.venv/Scripts/python.exe experiments/crlb-precision/crlb.py
WIDTH_DEBIAS_WORKERS=0 .venv/Scripts/python.exe experiments/crlb-precision/efficiency_check.py generate
.venv/Scripts/python.exe experiments/crlb-precision/efficiency_check.py analyse > output/crlb-efficiency/efficiency_check.txt
```

## Key result + figure

`sqrt(CRLB)/D` is several times D at low CNR: a single low-CNR measurement cannot
pin D. `render_crlb_panel.py` renders the CRLB figure
(`poster_panel4_crlb.png`); `scripts/poster_figures.py` imports that builder.

`efficiency_check` (`efficiency_check.txt`, rerun 2026-10-28 with independent
seeds per shape and the library weights): the library arm, which weights every
frame (t = 0 included) by 1 / Var(sigma_hat^2_i) and so equals fitting the
corrected widths against time, reaches 0.95 to 1.10 x sqrt(CRLB) from CNR0 21
on all four shapes (frames with per-frame CNR >= 1; bootstrap SE about 0.02
per cell), and its mean D/D_in there is 0.78 to 1.10 (0.95 to 1.03 at
D >= 0.02). This table prints no SE on the means;
`dfree-calibration/slope_fit_check.py` scores the same evaluation sets in
combined SE (2 of 96 cells beyond 2 SE). Below CNR0 21 the spread rises to
1.05 to 3.2 x sqrt(CRLB) at CNR0 3 to 13. The former MSD-point weights (1 / (v_i + v_0), frame 0 at weight
0) gave 1.30 to 1.43 x sqrt(CRLB) at CNR0 34 to 90, about half the available
information, because they treat the MSD points as independent although they
share sigma_hat^2(0) and give frame 0 no weight; inverse-CRLB weights on the
MSD points do not recover it.

## Depends on

`width_debias` (`montecarlo.run_ensemble`, `simulate`) and `crlb.py` in this
folder; `render_crlb_panel.py` also imports shared plotting primitives from
`scripts/poster_style.py`.

## See also

`precision_limit.md` (this folder); `docs/DERIVATION.md` (the bias half).
