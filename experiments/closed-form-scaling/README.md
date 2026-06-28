# closed-form-scaling

## Purpose

Test the dimensional/CRLB scaling claim that the folded-normal bias `bias/sigma^2`
is a function of `s = CNR*sqrt(sigma/dx)` alone, not CNR alone -- the empirical
handle on the parked closed-form derivation.

## Status

Parked / paper-scope. The closed-form derivation is not poster-blocking; this is
its empirical check.

## Contents

- `linearity_test.py` -- fits the median bias against the analytical
  `lambda(-s)/s` form across an `s` band.

## Data

None. Fully synthetic.

## Run

```bash
.venv/Scripts/python.exe experiments/closed-form-scaling/linearity_test.py
```

Set `WEIGHTING_WALL_WORKERS=0`. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

The median bias tracks the analytical `lambda(-s)/s` form in an intermediate
band, but the mean is corrupted by broken-fit saturation outliers. Figure:
`linearity_test.png`; table `linearity_test.txt`.

## Depends on

`weighting_wall`. No sibling-folder helpers.

## See also

`docs/DERIVATION.md` section 10 (parked closed-form scaling), `docs/STATUS.md`.
