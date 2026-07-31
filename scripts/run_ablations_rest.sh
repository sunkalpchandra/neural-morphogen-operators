#!/usr/bin/env bash
# Remaining ablation variants (those the first-stage dispatcher never reached).
set -u
cd "$(dirname "$0")/.."
source scripts/jobpool.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=.venv/bin/python
EPOCHS="${EPOCHS:-300}"; SEEDS="${SEEDS:-0 1}"; WORKERS="${WORKERS:-3}"
mkdir -p logs
JOBS=$(mktemp)
for v in isotropic_diffusion state_dependent_diffusion discrete_gnn latent_8 latent_16 latent_64; do
  echo "$PY experiments/exp5_ablations.py --section visium_mouse_brain --variants $v --seeds $SEEDS --epochs $EPOCHS --out-dir results/exp5/$v > logs/exp5_$v.log 2>&1" >> $JOBS
done
echo "[stage 1b] $(wc -l < $JOBS|tr -d ' ') jobs, $WORKERS workers, $(date +%H:%M:%S)"
run_pool "$WORKERS" "$JOBS"
rm -f $JOBS
echo "[stage 1b done] $(date +%H:%M:%S)"
