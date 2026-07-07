# noise-law

## Purpose

Show that the truncation bias is governed by the contrast against the
**constant** (white / read-floor) part of the noise, not the signal-following
shot part: white noise carries the bias everywhere, pure shot noise nowhere, and
a read floor of a few counts restores it.

## Status

Validated. Poster-scope framing (Panels 2-3; see
`docs/poster/NOISE_LAW_REFRAME.md`).

## Contents

- `shot_noise_validation.py` -- folded-normal correction across white / shot /
  Poisson laws at matched peak CNR; correctness (matched calibration removes
  bias) and robustness (does the white calibration transfer?).
- `read_floor_sweep.py` -- raises a constant read floor on shot noise and tracks
  the bias returning.

## Data

None. Fully synthetic (`folded_debias.noise` models).

## Run

```bash
.venv/Scripts/python.exe experiments/noise-law/shot_noise_validation.py
.venv/Scripts/python.exe experiments/noise-law/read_floor_sweep.py
```

Set `FOLDED_DEBIAS_WORKERS=0`. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

White: OLS mean error 18.6, folded matched 0.22. Shot: OLS 0.22 (already near
unbiased). A read floor of a few counts restores most of the white-limit bias.
The white calibration must not be applied to shot data (over-subtracts, D goes
negative). Figures: `read_floor_sweep.png`, `fftcnr_white_vs_shot.png`.

## Depends on

`folded_debias` (`noise`). No sibling-folder helpers.

## See also

`docs/poster/NOISE_LAW_REFRAME.md`, `docs/STATUS.md` (2026-06-11 update).
