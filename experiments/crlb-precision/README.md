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
- `precision_limit.md` -- the derivation and discussion.

## Data

None. Fully synthetic (uses `weighting_wall.simulate` geometry).

## Run

```bash
.venv/Scripts/python.exe experiments/crlb-precision/crlb.py
```

## Key result + figure

`sqrt(CRLB)/D` is several times D at low CNR: a single low-CNR measurement cannot
pin D. The poster CRLB panel (`poster_panel4_crlb.png`, copied here as the
representative figure) is currently rendered by the shared poster assembly
(`scripts/poster_figures.py`, Panel 4); splitting that render into this folder is
a pending convention cleanup.

## Depends on

`weighting_wall` (`montecarlo.run_ensemble`, `simulate`). The poster figure is
rendered by `scripts/poster_figures.py` (shared poster assembly).

## See also

`docs/poster/POSTER_PLAN.md` Panel 3/4, `docs/STATUS.md`.
