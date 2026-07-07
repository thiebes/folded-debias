# correction-floor

## Purpose

Characterise where the correction works: the per-profile CNR floor below which
the calibrated subtraction stops recovering the true variance, both in isolation
and inside a full diffusion experiment.

## Status

Validated. Poster-scope (source for the Panel 5 floor inset).

## Contents

- `per_profile_floor.py [profile_model]` -- isolated single-profile sweep
  (50,000 fits per CNR cell); residual `mean(sigma_tilde^2) - sigma^2_true`.
- `analytical_floor.py` -- power-law fit `var_bias(CNR) ~ A/CNR^2` and the
  model-saturation breakdown point.
- `per_frame_in_diffusion_experiment.py [profile_model]` -- the per-frame
  folded-normal / MAS corrections inside a 12-frame diffusion experiment.
- `render_floor_panel.py` -- renders the poster floor panel
  (`poster_panel5_floor.png`) from this set's `per_profile_floor_*.txt`; uses
  `scripts/poster_style.py`.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/correction-floor/per_profile_floor.py
.venv/Scripts/python.exe experiments/correction-floor/analytical_floor.py
.venv/Scripts/python.exe experiments/correction-floor/per_frame_in_diffusion_experiment.py
```

Set `FOLDED_DEBIAS_WORKERS=0`; `per_profile_floor.py` runs ~45 min (Gaussian) to
~3 h (Voigt). Outputs to repo-root `output/`; committed snapshots are here.

## Key result + figure

Reliable for per-profile CNR >~ 1, useful to ~0.7, fails below ~0.3. The
power-law breakdown is much deeper (CNR ~0.1) where sigma_hat^2 saturates against
the fit's `var_max`. Design rule: keep every frame's per-profile CNR above ~1.
Result tables: `per_profile_floor_<model>.txt`, `analytical_floor.txt`,
`per_frame_in_diffusion_experiment.txt`.

## Depends on

`folded_debias`. No sibling-folder helpers.

## See also

`docs/STATUS.md`, `docs/poster/POSTER_PLAN.md` Panel 5.
