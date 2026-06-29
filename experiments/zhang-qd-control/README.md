# zhang-qd-control

## Purpose

A scope/robustness control: apply the folded-normal correction to Zhang et al.
(2022) QD-solid ultrafast exciton-transport data (Dfast at early times) and show
the result is robust -- the correction barely moves it, because the per-frame CNR
stays well above the floor. This is a control example, not a correction target.

## Status

Validated. Poster scope/control (Panel 6 candidate as a robustness control, not a
bias demonstration).

## Contents

- `zhang_qd_panel.py` -- reduces the published figure CSVs and runs the
  correction; reads the `Figure *.csv` files from this folder.
- `Figure 1d <delay>.csv`, `Figure 1e <delay>.csv` -- raw 64x64 differential
  ON/OFF transient-microscopy frames at each pump-probe delay.
- `Figure 2a.csv`, `Figure 2b.csv` -- published MSD(t) curves.
- `Figure 4a-d.csv` -- MSD(t) of the CdSe / CdS / CsPbI3 "no transport" controls.

## Data

The extracted published-figure CSVs are committed here (small real data). The raw
Origin project (`zhang_Manuscript_data.opju`) is the source; it is gitignored
(local only) -- see `DATASETS.md`.

## Run

```bash
.venv/Scripts/python.exe experiments/zhang-qd-control/zhang_qd_panel.py
```

Outputs to repo-root `output/`; the committed snapshot is `zhang_qd_panel.png`.

## Key result + figure

The Zhang QD Dfast is robust to the correction: per-frame CNR ~9-24 (white noise),
sub-percent bias, and the bias rises mildly with pump-probe delay. The correction
leaves the published transport essentially unchanged -- a clean demonstration that
the correction does not distort already-healthy-CNR measurements. Figure:
`zhang_qd_panel.png`.

## Depends on

`weighting_wall`. The figure CSVs live beside the script (no external loader).

## See also

`docs/STATUS.md`. Source paper: Zhang et
al., "Ultrafast exciton transport at early times in quantum dot solids" (2022).
