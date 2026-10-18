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

## Depends on

`folded_debias` (`montecarlo.run_ensemble`, `simulate`) and `crlb.py` in this
folder; `render_crlb_panel.py` also imports shared poster primitives from
`scripts/poster_style.py`.

## See also

`docs/poster/POSTER_PLAN.md` Panel 3/4.
