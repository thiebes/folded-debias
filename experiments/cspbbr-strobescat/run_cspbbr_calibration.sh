#!/usr/bin/env bash
# Preset: CsPbBr3 single-crystal diffusion (Upasana's measurements), Gaussian
# white-noise calibration.
#
# Single-fluence calibration. For ALL fluences at once (reduce + calibrate +
# apply + a D-vs-fluence summary), use experiments/cspbbr-strobescat/run_all.py, which
# derives each fluence's geometry from a fresh reduction and shares one
# calibration across fluences on the same pixel grid.
#
# Geometry is from a fresh reduction of the 13.47 μJ fluence (merged 46-pixel
# spatial profiles): n_pixels 46, half-window 3.238 um, sigma0 ~ 0.41 um, fitted
# sigma reaching ~1.9 um at late times. White noise (not Poisson) because the data's
# photon-transfer fit is floor-dominated, not shot-limited -- it is a raster-
# scanned APD measurement whose constant noise floor (scanning + dark) drives
# the truncation bias. See docs/VALIDATION_DESIGN.md and docs/calibration_runs.md.
#
# Run on the 16-core WSL box:
#   PYTHON=.venv-linux/bin/python bash experiments/cspbbr-strobescat/run_cspbbr_calibration.sh
# Override workers with WORKERS=N.
#
# Tunables to revisit against the corrected-D result:
#   --sigma-max  true width range to cover (data's fitted sigma is partly
#                bias-inflated, so the credible true max is lower than 1.9 um).
#   --max-var    fit's variance cap; above the observed widths but below the
#                window scale (half-window^2 = 9.6 um^2) so low-CNR fits do not
#                wander to the window edge. Must match the application's cap.
#   --n-runs     8000 is high-statistics (~96M fits); drop to ~2000 for a quick
#                look.
set -euo pipefail
cd "$(dirname "$0")/.."

exec bash scripts/run_calibration.sh \
  --tag cspbbr_gaussian \
  --n-pixels 46 --half-window 3.238 \
  --sigma-min 0.15 --sigma-max 1.5 --n-sigma 100 \
  --cnr-min 0.5 --cnr-max 60 --n-cnr 120 \
  --n-runs 8000 \
  --model gaussian \
  --max-var 6.25
