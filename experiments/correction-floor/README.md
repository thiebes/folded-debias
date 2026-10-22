# correction-floor

## Purpose

Characterise where the correction works: the per-profile CNR floor below which
the calibrated subtraction stops recovering the true variance, both in isolation
and inside a full diffusion experiment.

## Contents

- `per_profile_floor.py [profile_model]` -- isolated single-profile sweep
  (50,000 fits per CNR cell); residual `mean(sigma_tilde^2) - sigma^2_true`.
- `analytical_floor.py [MODEL ...]` -- tests the folded-normal prediction
  `var_bias = A/CNR^2`, with `A = CNR^2 Var(sigma_hat^2)/4` measured at high
  CNR, against the binned calibration, and reports where it breaks.
- `per_frame_in_diffusion_experiment.py [profile_model]` -- the per-frame
  folded-normal / MAS corrections inside a 12-frame diffusion experiment.
- `render_floor_panel.py` -- renders the floor figure
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

Per-profile floor (50,000 fits per cell; deepest CNR before the first cell
whose residual exceeds both 2 SE and 0.01 sigma_0^2): Gaussian 1.16, Voigt 1.5
(within 0.01 sigma_0^2 again from 1.16 down to 0.53), p = 1.5 0.89, p = 4 0.69
(within 0.013 down to 0.11). The p = 4 fit reads low, not high, at deep CNR.
The floor readout requires |residual| > 0.01 sigma_0^2 in addition to
significance, because at 50,000 fits significance alone flags 1e-4 calibration
offsets and reports the top of the grid. The folded-normal model
breaks at about the same place: `A` = 0.058 (Gaussian), 0.082 (Voigt), 0.10
(p = 1.5), 0.019 (p = 4) sigma_0^2; resolved departure at CNR ~0.85 for
Gaussian and p = 1.5, none above 0.28 for Voigt, and a negative low-CNR bias
for p = 4 (departure at CNR ~0.36). Design rule: keep every frame's
per-profile CNR above ~1.
Result tables: `per_profile_floor_<model>.txt` (`genGauss` is a p = 2 run kept for
reference; `genGauss_p1.5`, `genGauss_p4`), `analytical_floor.txt`,
`per_frame_in_diffusion_experiment.txt`.

## Depends on

`folded_debias`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md` section 9.
