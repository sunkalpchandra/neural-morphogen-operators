#!/usr/bin/env bash
# Single memory-aware queue for everything that remains.
#
# This machine has 9 GB of RAM. Running ~11 trainings concurrently drove it into
# swap (62M swapouts, 50% of CPU in the kernel, 22% idle at load 24), which made
# every job slower than running fewer would have. Concurrency is therefore capped
# low: total CPU work is fixed, and the only thing extra processes buy here is
# paging. Every experiment is resumable at (variant, seed) granularity, so
# re-running this script picks up wherever it left off.
set -u
cd "$(dirname "$0")/.."
source scripts/jobpool.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=.venv/bin/python
W="${WORKERS:-3}"; SEEDS="${SEEDS:-0 1}"; EP="${EP:-300}"; EP2="${EP2:-250}"
mkdir -p logs

echo "[wait] for in-flight jobs to drain..."
while pgrep -f 'exp5_ablations[.]py' > /dev/null; do sleep 30; done
echo "[go] $(date +%H:%M:%S) workers=$W"

J=$(mktemp)
# -- core structural ablations whose seed 1 was pre-empted ------------------
for v in no_bio_reg no_pde no_diffusion; do
  echo "$PY experiments/exp5_ablations.py --section visium_mouse_brain --variants $v --seeds $SEEDS --epochs $EP --out-dir results/exp5/$v >> logs/exp5_$v.log 2>&1" >> $J
done
# -- secondary ablations ----------------------------------------------------
for v in isotropic_diffusion state_dependent_diffusion discrete_gnn latent_8 latent_16 latent_64; do
  echo "$PY experiments/exp5_ablations.py --section visium_mouse_brain --variants $v --seeds $SEEDS --epochs $EP --out-dir results/exp5/$v >> logs/exp5_$v.log 2>&1" >> $J
done
echo "[stage A] $(wc -l < $J|tr -d ' ') ablation jobs"
run_pool "$W" "$J"; : > $J

# -- transfer / perturbation / development ----------------------------------
for m in nmo stagate; do
  echo "$PY experiments/exp2_cross_tissue.py --models $m --seeds $SEEDS --epochs $EP2 --finetune-epochs 80 --out-dir results/exp2/$m > logs/exp2_$m.log 2>&1" >> $J
  echo "$PY experiments/exp3_resolution.py --models $m --seeds $SEEDS --epochs $EP2 --max-target-locations 8000 --out-dir results/exp3/$m > logs/exp3_$m.log 2>&1" >> $J
done
echo "$PY experiments/exp4_perturbation.py --models nmo stagate --seeds $SEEDS --epochs $EP2 --out-dir results/exp4 > logs/exp4.log 2>&1" >> $J
echo "$PY experiments/exp6_development.py --seeds $SEEDS --epochs $EP2 --out-dir results/exp6 > logs/exp6.log 2>&1" >> $J
echo "[stage B] $(wc -l < $J|tr -d ' ') transfer/perturbation jobs  $(date +%H:%M:%S)"
run_pool "$W" "$J"
rm -f $J
echo "[ALL DONE] $(date +%H:%M:%S)"
