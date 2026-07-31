#!/usr/bin/env bash
# Finish exp2 (it crashed on the GP fine-tune), then regenerate everything.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
echo "== waiting for all running experiments =="
while pgrep -f "exp13_spectral\.py|exp3_resolution\.py|exp10_robustness\.py|exp2_cross_tissue\.py" > /dev/null; do sleep 20; done
echo "== exp2: resuming after the GP fine-tune fix =="
$PY experiments/exp2_cross_tissue.py --seeds 0 1 --epochs 300 2>&1 | tail -15
echo "== regenerating every derived artifact =="
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -4
$PY scripts/verify_theory.py 2>&1 | tail -2
echo "FINISH QUEUE COMPLETE"
