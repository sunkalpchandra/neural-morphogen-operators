#!/usr/bin/env bash
# Chained runner for experiments 2/3/4/6. Waits for exp1 to release cores,
# then runs three streams in parallel. Every experiment is resumable, so a
# re-run skips completed (model, seed) combinations.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
SEEDS="${SEEDS:-0 1}"
EPOCHS="${EPOCHS:-300}"

echo "[wait] for exp1 to finish..."
while pgrep -f exp1_forecasting > /dev/null; do sleep 20; done
echo "[go] exp1 done at $(date +%H:%M:%S)"

nice -n 5 $PY experiments/exp2_cross_tissue.py --models nmo stagate gp \
    --seeds $SEEDS --epochs $EPOCHS --finetune-epochs 120 > logs_exp2.txt 2>&1 &
A=$!
nice -n 5 $PY experiments/exp3_resolution.py --models nmo stagate gp \
    --seeds $SEEDS --epochs $EPOCHS > logs_exp3.txt 2>&1 &
B=$!
( nice -n 5 $PY experiments/exp4_perturbation.py --models nmo stagate \
      --seeds $SEEDS --epochs $EPOCHS > logs_exp4.txt 2>&1
  nice -n 5 $PY experiments/exp6_development.py \
      --seeds $SEEDS --epochs $EPOCHS > logs_exp6.txt 2>&1 ) &
C=$!

wait $A; echo "[done] exp2 $(date +%H:%M:%S)"
wait $B; echo "[done] exp3 $(date +%H:%M:%S)"
wait $C; echo "[done] exp4+exp6 $(date +%H:%M:%S)"
echo "[all done] $(date +%H:%M:%S)"
