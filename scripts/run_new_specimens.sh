#!/usr/bin/env bash
# Benchmark the three new independent specimens at exactly the protocol every
# other section in Table 1 was run under.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
$PY experiments/exp8_multisection.py \
    --sections visium_mouse_kidney visium_human_lymph_node visium_mouse_brain_coronal \
    --models nmo stagate gnn autoencoder tangram gp_multiscale neural_field \
    --seeds 0 1 --epochs 200 --max-locations 2500 --shard 0 --n-shards 1
echo "== regenerating =="
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -3
echo "NEW SPECIMENS COMPLETE"
