#!/usr/bin/env bash
# Regenerate every derived artifact from the run results, in dependency order.
#
#   ./scripts/finalize.sh
#
# 1. re-score exp1 checkpoints with the current metric code (so a metric change
#    can never leave a mix of versions across tables)
# 2. figures + LaTeX tables + inline-number macros
# 3. compile the paper and report main-content page count
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=. OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=$PWD/.venv/bin/python
SECTION="${SECTION:-visium_mouse_brain}"

echo "== 1/3  re-scoring exp1 checkpoints =="
$PY scripts/reevaluate.py --results results/exp1 2>&1 | tail -3

echo "== 2/3  figures, tables, numbers =="
$PY experiments/make_figures.py --section "$SECTION" 2>&1 | tail -4

echo "== 3/3  compiling paper =="
cd paper
if command -v tectonic > /dev/null; then
  tectonic -X compile neurips_2026.tex --outdir /tmp/texout --keep-intermediates 2>&1 \
    | grep -E "^error|Writing" | tail -3
  cp -f /tmp/texout/neurips_2026.pdf . 2>/dev/null
  page=$(grep -o 'endofmain}{{[^}]*}{[0-9]*}' /tmp/texout/neurips_2026.aux 2>/dev/null \
         | grep -o '{[0-9]*}$' | tr -d '{}')
  echo "main content ends on page: ${page:-?}  (limit 9)"
else
  echo "tectonic not found; skipping PDF build"
fi
echo "done."
