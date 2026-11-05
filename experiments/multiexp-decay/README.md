# multiexp-decay

## Purpose

Confirm the correction is not invalidated by multi-exponential decay: it keys on
the per-frame CNR, which fft-cnr measures directly, not on the decay law.

## Contents

- `multiexp_decay.py` -- fixed Fickian diffusion with three decay laws (one
  single-exponential, two bi-exponential) on the `MultiExpExperiment` bed.

## Data

None. Fully synthetic. Reuses the cached `calibration_subdiffusive.npz`.

## Run

```bash
.venv/Scripts/python.exe experiments/multiexp-decay/multiexp_decay.py
```

Set `WIDTH_DEBIAS_WORKERS=0`; ~6 min. Outputs to repo-root `output/`; committed
snapshots are here.

## Key result + figure

The per-frame corrected residual collapses onto one curve vs per-frame CNR
(across-law spread ≤ 0.012 σ₀² for CNR ≥ 1; the printed maximum of 0.162 σ₀²
includes the 0.7 to 1.0 bin, below the floor); corrected ⟨D⟩/D_ref stays
0.93 to 1.06 over CNR₀ 8 to 90 for all three decay laws (the table prints no
SE). The D-free calibration is the shared `calibration_subdiffusive.npz`
(Gaussian, 60 CNR₀ × 1500 runs), named with the seeds in the table header. The naive result is strongly decay-specific (19x at CNR0 = 8 for
single-exponential vs ~1.3x bi-exponential), so any CNR0-indexed threshold is
single-exponential-specific -- state thresholds in per-frame CNR. Figure:
`multiexp_decay.png`; table `multiexp_decay.txt`.

## Depends on

`width_debias` (`MultiExpExperiment`). No sibling-folder helpers.

## See also

`docs/DERIVATION.md` (decay-law independence).
