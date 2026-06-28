# Datasets -- cspbbr-strobescat

Collaborator data (Upasana; the Antonella/Carlos diffusion paper). Read-only;
referenced by pointer only -- never copied into git, never moved or renamed on
the pshare.

## Source

Raw CsPbBr3 single-crystal transient-PL-microscopy acquisition CSVs on the
pshare, organised by fluence subfolder under the sample root:

```text
P:/Upasana/CsPbBr_SingleCrystal/Sample_1
```

Each fluence subfolder (e.g. `13.47 uJ_Fluence_19_May_2026`) holds one CSV per
repeat (each a spatial-position scan) plus a `*metadata.json`. `reduce.py` reads
these via `--data` / `--sample-root`; the run machine must be able to read the
pshare.

## Notes

The noise is white (constant-floor) -- a raster-scanned APD measurement whose
photon-transfer fit is floor-dominated -- so the gaussian calibration model
applies. See `docs/calibration_runs.md`.
