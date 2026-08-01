#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
$PY -m src.data.download --dataset visium_ffpe_human_prostate visium_ffpe_mouse_brain 2>&1 | tail -2
$PY -m src.data.build --overwrite --dataset visium_ffpe_human_prostate visium_ffpe_mouse_brain 2>&1 | grep -E "split \(cont|coords:|\[ok\]|Error"
$PY experiments/exp8_multisection.py --sections visium_ffpe_human_prostate visium_ffpe_mouse_brain \
    --models nmo stagate gnn autoencoder tangram gp_multiscale neural_field \
    --seeds 0 1 --epochs 200 --max-locations 2500 --shard 0 --n-shards 1 2>&1 | grep -E "^\[|FAIL"
$PY experiments/exp9_biology.py --sections visium_ffpe_human_prostate visium_ffpe_mouse_brain 2>&1 | tail -3
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -2
echo "FFPE SPECIMENS COMPLETE"
