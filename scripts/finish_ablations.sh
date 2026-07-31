#!/usr/bin/env bash
# Remaining exp5 ablation runs, then the transfer/perturbation/development stage.
#
# Does NOT wait for the matched-budget controls: those also run
# exp5_ablations.py, so a name-based wait would serialise behind them for no
# reason. They run concurrently; concurrency stays low because this machine has
# 9 GB of RAM and thrashes above ~5 trainings.
set -u
cd "$(dirname "$0")/.."
source scripts/jobpool.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=.venv/bin/python
W="${WORKERS:-3}"; SEEDS="${SEEDS:-0 1}"; EP="${EP:-300}"; EP2="${EP2:-250}"
mkdir -p logs
J=$(mktemp)
for v in no_bio_reg no_pde no_diffusion isotropic_diffusion state_dependent_diffusion \
         discrete_gnn latent_8 latent_16 latent_64; do
  echo "$PY experiments/exp5_ablations.py --section visium_mouse_brain --variants $v --seeds $SEEDS --epochs $EP --out-dir results/exp5/$v >> logs/exp5_$v.log 2>&1" >> $J
done
echo "[ablations] $(wc -l < $J|tr -d ' ') jobs, $W workers, $(date +%H:%M:%S)"
run_pool "$W" "$J"; : > $J
echo "[ablations done] $(date +%H:%M:%S)"

for m in nmo stagate; do
  echo "$PY experiments/exp2_cross_tissue.py --models $m --seeds $SEEDS --epochs $EP2 --finetune-epochs 80 --out-dir results/exp2/$m > logs/exp2_$m.log 2>&1" >> $J
  echo "$PY experiments/exp3_resolution.py --models $m --seeds $SEEDS --epochs $EP2 --max-target-locations 8000 --out-dir results/exp3/$m > logs/exp3_$m.log 2>&1" >> $J
done
echo "$PY experiments/exp4_perturbation.py --models nmo stagate --seeds $SEEDS --epochs $EP2 --out-dir results/exp4 > logs/exp4.log 2>&1" >> $J
echo "$PY experiments/exp6_development.py --seeds $SEEDS --epochs $EP2 --out-dir results/exp6 > logs/exp6.log 2>&1" >> $J
echo "[stage B] $(wc -l < $J|tr -d ' ') jobs $(date +%H:%M:%S)"
run_pool "$W" "$J"
rm -f $J
echo "[ALL DONE] $(date +%H:%M:%S)"
