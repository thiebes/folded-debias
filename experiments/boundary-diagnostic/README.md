# boundary-diagnostic

## Purpose

Decide which boundary model the width fit obeys at the wall at zero: does the
fitted variance **clip** (Tobit-style pile-up at the bound) or **reflect** (fold
into a right-skewed tail)? This decision selects the correction route.

## Status

Validated. Framing reference (cited in `docs/DERIVATION.md` section 2); not a
standalone poster panel, but underpins Panel 2.

## Contents

- `diagnose_boundary.py [profile_model]` -- one run per profile model; reports
  the fitted-variance distribution shape vs CNR.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/boundary-diagnostic/diagnose_boundary.py
.venv/Scripts/python.exe experiments/boundary-diagnostic/diagnose_boundary.py voigt
```

Outputs to repo-root `output/`; committed snapshots are in this folder.

## Key result + figure

The fit **reflects** rather than clips, so the **folded-normal** route applies
(not Tobit/clipping). Voigt and generalized-Gaussian show only a minor ~5%
clipping pile-up at CNR around 1. Figures: `boundary_diagnostic.png`,
`boundary_diagnostic_voigt.png`, `boundary_diagnostic_generalized_gaussian.png`.

## Depends on

`weighting_wall`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md` section 2, `docs/STATUS.md`.
