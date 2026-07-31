#!/usr/bin/env bash
# Benchmark the imputation-oriented baselines at exactly the protocol every
# other model in Table 1 was run under: 200 epochs, 2500 locations, 2 seeds.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
while pgrep -f "exp2_cross_tissue\.py|exp10_robustness\.py" > /dev/null; do sleep 20; done
echo "== Tangram-style and SpaGE-style across the benchmark =="
$PY experiments/exp8_multisection.py --models tangram spage --seeds 0 1 \
    --epochs 200 --max-locations 2500 --shard 0 --n-shards 1 2>&1 | tail -40
echo "== regenerating =="
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -3
echo "IMPUTATION BASELINES COMPLETE"
