#!/usr/bin/env bash
# The density axis on a large section, with seeds.
#
# On visium_mouse_brain (2,691 locations) the 1/8 level leaves ~70 held-out
# locations -- below the threshold this project uses to declare Pearson r
# unestimable, and the source of a non-monotone curve. xenium_mouse_brain has
# 36,362 locations, so 1/8 still leaves thousands.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
$PY experiments/exp10_robustness.py --section xenium_mouse_brain \
    --axes density --models nmo stagate gnn --seeds 0 1 --epochs 150 \
    --out-dir results/exp10_density 2>&1 | grep -E "^\[|FAIL"
echo "DENSITY SWEEP COMPLETE"
