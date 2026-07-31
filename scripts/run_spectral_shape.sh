#!/usr/bin/env bash
# The corrected spectral form, run after the 'full' sweep completes.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
while pgrep -f "exp13_spectral\.py" > /dev/null; do sleep 20; done
echo "== Stage 4b: shape-matched spectral sweep =="
$PY experiments/exp13_spectral.py --mode shape --weights 0.01 0.1 0.3 1.0 --seeds 0 1 --epochs 500
echo "== regenerating =="
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -3
echo "STAGE 4B COMPLETE"
