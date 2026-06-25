#!/usr/bin/env bash
# General Monte Carlo calibration runner for a multi-core box (e.g. WSL).
#
# Sets the threading environment the embarrassingly-parallel fit Monte Carlo
# needs, then forwards every argument to scripts/build_calibration.py. Use it to
# run ANY calibration -- pass whatever flags that tool accepts:
#
#   PYTHON=.venv-linux/bin/python bash scripts/run_calibration.sh \
#     --tag my_run --n-pixels 64 --half-window 2.5 \
#     --sigma-min 0.1 --sigma-max 1.5 --cnr-min 0.5 --cnr-max 80 \
#     --n-runs 8000 --model gaussian
#
# For the Poisson / photon-transfer noise model add:
#     --noise poisson --gain G --read-noise R --background B
#
# Dataset presets (run_<dataset>_calibration.sh) are thin wrappers that call
# this with a fixed parameter set. To add a dataset, copy a preset and edit the
# geometry and noise flags from your reduced data. See docs/calibration_runs.md.
#
# Env:
#   PYTHON   python interpreter (default: python3)
#   WORKERS  worker processes (default: all logical cores, nproc)
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

# The fits are tiny nonlinear least-squares; BLAS threading does not help, and
# one BLAS thread per worker times all cores oversubscribes and can exhaust
# memory. Pin to one thread per process.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PY="${PYTHON:-python3}"
WORKERS="${WORKERS:-$(nproc)}"

echo "calibration: $WORKERS workers, python = $PY"
# --n-workers is passed first so an explicit --n-workers in the forwarded
# arguments (argparse last-wins) can still override it.
exec "$PY" scripts/build_calibration.py --n-workers "$WORKERS" "$@"
