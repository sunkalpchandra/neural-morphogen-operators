#!/usr/bin/env bash
# Everything remaining, strictly sequential. One trainer at a time: six
# concurrent jobs is what OOM-killed exp10 on this machine originally.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python

echo "== waiting for the spectral sweeps =="
while pgrep -f "exp13_spectral\.py" > /dev/null; do sleep 20; done
echo "spectral sweeps done"

# The transfer results were produced under a random-half protocol that is not
# comparable with the in-domain oracle. Archive them and re-run under blocks.
echo "== Stage 1 fix: re-running transfer under the block protocol =="
for d in exp2 exp3; do
  [ -d "results/$d" ] && mv "results/$d" "results/${d}_randomsplit_archived" 2>/dev/null
done
$PY experiments/exp2_cross_tissue.py --seeds 0 1 --epochs 300 2>&1 | tail -20
$PY experiments/exp3_resolution.py  --seeds 0 1 --epochs 300 2>&1 | tail -20

echo "== exp10: robustness, serial (was OOM-killed at 6 workers) =="
$PY experiments/exp10_robustness.py --axes noise dropout density knn \
    --models nmo stagate gnn --seeds 0 --epochs 150 2>&1 | tail -30

echo "== regenerating every derived artifact =="
$PY experiments/make_figures.py --section visium_mouse_brain 2>&1 | tail -4
$PY scripts/verify_theory.py 2>&1 | tail -3

echo "FINAL QUEUE COMPLETE"
