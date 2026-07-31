#!/usr/bin/env bash
# Job-queue runner for every remaining experiment.
#
# Rather than a few long serial streams, we emit one job per (experiment,
# variant/model) and run them through a fixed-width worker pool. Each job owns
# its own output directory so there is no write contention, and each experiment
# script is resumable, so re-running the queue skips completed work.
#
#   WORKERS=6 EPOCHS=300 ./scripts/run_all.sh
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
WORKERS="${WORKERS:-6}"
EPOCHS="${EPOCHS:-300}"
SEEDS="${SEEDS:-0 1}"
LOGDIR=logs; mkdir -p $LOGDIR

echo "[wait] for exp1 to release cores..."
# Match the script filename *with extension* so this loop cannot be fooled by
# another shell whose command line merely mentions the experiment name (an
# interactive monitor, a grep, this script itself).
while pgrep -f 'exp1_forecasting\.py' > /dev/null; do sleep 20; done
echo "[go] $(date +%H:%M:%S)  workers=$WORKERS epochs=$EPOCHS seeds='$SEEDS'"

JOBS=$(mktemp)

# ---- Experiment 5: one job per ablation variant (each runs all seeds) ------
for v in full no_dynamics no_diffusion no_reaction no_pde no_bio_reg \
         isotropic_diffusion state_dependent_diffusion discrete_gnn \
         latent_8 latent_16 latent_64; do
  echo "$PY experiments/exp5_ablations.py --section visium_mouse_brain --variants $v \
        --seeds $SEEDS --epochs $EPOCHS --out-dir results/exp5/$v > $LOGDIR/exp5_$v.log 2>&1" >> $JOBS
done

# ---- Experiment 2: one job per model --------------------------------------
for m in nmo stagate gp; do
  echo "$PY experiments/exp2_cross_tissue.py --models $m --seeds $SEEDS --epochs $EPOCHS \
        --finetune-epochs 100 --out-dir results/exp2/$m > $LOGDIR/exp2_$m.log 2>&1" >> $JOBS
done

# ---- Experiment 3: one job per model --------------------------------------
for m in nmo stagate gp; do
  echo "$PY experiments/exp3_resolution.py --models $m --seeds $SEEDS --epochs $EPOCHS \
        --max-target-locations 10000 --out-dir results/exp3/$m > $LOGDIR/exp3_$m.log 2>&1" >> $JOBS
done

# ---- Experiments 4 and 6 ---------------------------------------------------
echo "$PY experiments/exp4_perturbation.py --models nmo stagate --seeds $SEEDS \
      --epochs $EPOCHS --out-dir results/exp4 > $LOGDIR/exp4.log 2>&1" >> $JOBS
echo "$PY experiments/exp6_development.py --seeds $SEEDS --epochs $EPOCHS \
      --out-dir results/exp6 > $LOGDIR/exp6.log 2>&1" >> $JOBS

N=$(wc -l < $JOBS | tr -d ' ')
echo "[queue] $N jobs, $WORKERS workers"
cat $JOBS | xargs -P "$WORKERS" -I{} bash -c '{}'
rm -f $JOBS
echo "[all done] $(date +%H:%M:%S)"
