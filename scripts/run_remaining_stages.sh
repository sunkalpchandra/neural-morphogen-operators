#!/usr/bin/env bash
# Remaining compute, strictly sequential.
#
# Running these concurrently is what starved exp11 earlier: this machine has
# 8 cores and 9 GB, and six concurrent trainers previously got exp10 OOM-killed.
# One job at a time is slower in principle and faster in practice.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python

echo "== waiting for exp11 to finish =="
while pgrep -f "exp11_difflen_null\.py" > /dev/null; do sleep 20; done
echo "exp11 done"

echo "== Stage 3: converged single-section comparison =="
$PY experiments/exp14_converged.py --seeds 0 1 2 --epochs 2000 --patience 300

echo "== Stage 4: spectral weight sweep =="
$PY experiments/exp13_spectral.py --weights 0 0.001 0.01 0.1 1.0 --seeds 0 1 --epochs 500

echo "== regenerating derived artifacts =="
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -3

echo "ALL STAGES COMPLETE"
