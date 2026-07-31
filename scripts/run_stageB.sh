#!/usr/bin/env bash
# Transfer / perturbation / development stage, run independently of the
# remaining latent-width ablations (which are slow and not on the critical path).
set -u
cd "$(dirname "$0")/.."
source scripts/jobpool.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=.venv/bin/python
W="${WORKERS:-3}"; SEEDS="${SEEDS:-0 1}"; EP="${EP:-250}"
mkdir -p logs
J=$(mktemp)
echo "$PY experiments/exp6_development.py --seeds $SEEDS --epochs $EP --out-dir results/exp6 > logs/exp6.log 2>&1" >> $J
echo "$PY experiments/exp4_perturbation.py --models nmo stagate --seeds $SEEDS --epochs $EP --out-dir results/exp4 > logs/exp4.log 2>&1" >> $J
for m in nmo stagate; do
  echo "$PY experiments/exp2_cross_tissue.py --models $m --seeds $SEEDS --epochs $EP --finetune-epochs 80 --out-dir results/exp2/$m > logs/exp2_$m.log 2>&1" >> $J
  echo "$PY experiments/exp3_resolution.py --models $m --seeds $SEEDS --epochs $EP --max-target-locations 8000 --out-dir results/exp3/$m > logs/exp3_$m.log 2>&1" >> $J
done
echo "[stage B] $(wc -l < $J|tr -d ' ') jobs, $W workers, $(date +%H:%M:%S)"
run_pool "$W" "$J"; rm -f $J
echo "[STAGE B DONE] $(date +%H:%M:%S)"
