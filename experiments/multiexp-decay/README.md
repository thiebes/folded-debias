# multiexp-decay

## Purpose

Confirm the correction is not invalidated by multi-exponential decay: it keys on
the per-frame CNR, which fft-cnr measures directly, not on the decay law.

## Status

Validated. Caveat / paper-scope (the decay-agnostic statement and the
single-exponential-specific threshold caveat).

## Contents

- `multiexp_decay.py` -- fixed Fickian diffusion with three decay laws (one
  single-exponential, two bi-exponential) on the `MultiExpExperiment` bed.

## Data

None. Fully synthetic. Reuses the cached `calibration_subdiffusive.npz`.

## Run

```bash
.venv/Scripts/python.exe experiments/multiexp-decay/multiexp_decay.py
```

Set `WEIGHTING_WALL_WORKERS=0`; ~6 min. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

The per-frame corrected residual collapses onto one curve vs per-frame CNR
(across-law spread <= 0.011 sigma0^2 for CNR >= 1); corrected `<D>/D_true` stays
0.9-1.1. The naive result is strongly decay-specific (19x at CNR0 = 8 for
single-exponential vs ~1.3x bi-exponential), so any CNR0-indexed threshold is
single-exponential-specific -- state thresholds in per-frame CNR. Figure:
`multiexp_decay.png`; table `multiexp_decay.txt`.

## Depends on

`weighting_wall` (`MultiExpExperiment`). No sibling-folder helpers.

## See also

`docs/STATUS.md` (2026-06-14 multi-exponential-decay update).
