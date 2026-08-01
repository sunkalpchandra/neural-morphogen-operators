#!/usr/bin/env bash
# Extend the biological evaluation to the new specimens, after the benchmark.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
while pgrep -f "exp8_multisection\.py" > /dev/null; do sleep 30; done
echo "== biology on the new specimens =="
$PY experiments/exp9_biology.py \
    --sections visium_mouse_kidney visium_human_lymph_node visium_mouse_brain_coronal 2>&1 | tail -20
echo "== regenerating =="
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -3
echo "BIOLOGY EXTENSION COMPLETE"
