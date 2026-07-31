#!/usr/bin/env bash
# Second-stage queue: transfer, perturbation and developmental experiments.
#
# Waits for the ablation stage to finish, then takes over. Scope is deliberately
# leaner than the ablation stage because each transfer job trains three models
# per (model, seed) -- a source fit, a decoder-only fine-tune and an in-domain
# oracle -- so cost grows ~3x faster per job than a plain ablation.
#
# We keep NMO plus the strongest graph baseline (STAGATE-style) and drop the
# Gaussian process, which trails by ~0.08 Pearson on the in-domain benchmark and
# would not change any conclusion here. The training-mean floor and the
# in-domain oracle -- the two references the zero-shot number is judged against
# -- are retained in full.
set -u
cd "$(dirname "$0")/.."
source scripts/jobpool.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=.venv/bin/python
WORKERS="${WORKERS:-5}"
EPOCHS="${EPOCHS:-250}"
SEEDS="${SEEDS:-0 1}"
mkdir -p logs

echo "[wait] for the ablation stage to drain..."
while pgrep -f 'exp5_ablations[.]py' > /dev/null; do sleep 30; done
echo "[go] $(date +%H:%M:%S)  workers=$WORKERS epochs=$EPOCHS"

JOBS=$(mktemp)
for m in nmo stagate; do
  echo "$PY experiments/exp2_cross_tissue.py --models $m --seeds $SEEDS --epochs $EPOCHS --finetune-epochs 80 --out-dir results/exp2/$m > logs/exp2_$m.log 2>&1" >> $JOBS
  echo "$PY experiments/exp3_resolution.py --models $m --seeds $SEEDS --epochs $EPOCHS --max-target-locations 8000 --out-dir results/exp3/$m > logs/exp3_$m.log 2>&1" >> $JOBS
done
echo "$PY experiments/exp4_perturbation.py --models nmo stagate --seeds $SEEDS --epochs $EPOCHS --out-dir results/exp4 > logs/exp4.log 2>&1" >> $JOBS
echo "$PY experiments/exp6_development.py --seeds $SEEDS --epochs $EPOCHS --out-dir results/exp6 > logs/exp6.log 2>&1" >> $JOBS

echo "[queue] $(wc -l < $JOBS | tr -d ' ') jobs, $WORKERS workers"
run_pool "$WORKERS" "$JOBS"
rm -f $JOBS
echo "[stage 2 done] $(date +%H:%M:%S)"
