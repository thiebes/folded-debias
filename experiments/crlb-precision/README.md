# crlb-precision

## Purpose

The Cramer-Rao lower bound on D: the floor that no unbiased estimator can beat
(`Var(D_hat) >= CRLB`). Separates the removable artefact (bias) from the
irreducible cost (variance), and shows that at low CNR a single measurement
cannot pin D regardless of estimator -- so replication is required, and only an
unbiased estimator converges to truth.

## Status

Validated. Poster-scope (Panel 3/4 precision floor).

## Contents

- `crlb.py` -- builds the Fisher information matrix for the simulated Gaussian
  measurement model and returns `[F^-1]_DD`, the CRLB on D, vs CNR0.
- `efficiency_check.py generate|analyse` -- measures the corrected D's spread
  against sqrt(CRLB) on Gaussian, Voigt, p = 1.5, and p = 4 (numerical Fisher
  matrix from each model's profile; D-free calibration per shape; 1000 runs
  per cell). Arms: library weights, inverse-CRLB weights, a robust spread, and
  a direct fit of the corrected widths against time. `generate` is about 70 min
  on 8 cores (cached in `output/crlb-efficiency/`).
- `render_crlb_panel.py` -- renders the poster CRLB panel
  (`poster_panel4_crlb.png`) from `crlb.py`; uses the shared poster primitives in
  `scripts/poster_style.py`.
- `precision_limit.md` -- the derivation and discussion.

## Data

None. Fully synthetic (uses `folded_debias.simulate` geometry).

## Run

```bash
.venv/Scripts/python.exe experiments/crlb-precision/crlb.py
```

## Key result + figure

`sqrt(CRLB)/D` is several times D at low CNR: a single low-CNR measurement cannot
pin D. The poster CRLB panel (`poster_panel4_crlb.png`) is rendered here by
`render_crlb_panel.py`; `scripts/poster_figures.py` imports that builder when it
assembles the poster figures.

`efficiency_check` (`efficiency_check.txt`): the corrected D's spread is 1.30
to 1.43 x sqrt(CRLB) at CNR0 34 to 90 on every shape, about half the available
information. Inverse-CRLB weights do not help; the loss is the MSD slope fit,
which treats the points as independent although they share sigma_hat^2(0) and
gives frame 0 no weight. Fitting the corrected widths against time directly
(every frame weighted by 1 / Var(sigma_hat^2_i), intercept free) reaches 0.98
to 1.08 x sqrt(CRLB) from CNR0 21 with no systematic shift of the mean. The
library adopted it the same day (`docs/STATUS.md`, 2026-10-22), so
`efficiency_check.txt` records the former library weights; re-running the
script now gives a library arm equal to the width-fit arm.

## Depends on

`folded_debias` (`montecarlo.run_ensemble`, `simulate`) and `crlb.py` in this
folder; `render_crlb_panel.py` also imports shared poster primitives from
`scripts/poster_style.py`.

## See also

`docs/poster/POSTER_PLAN.md` Panel 3/4.
