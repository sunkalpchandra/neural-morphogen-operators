#!/usr/bin/env bash
# Post-regeneration verification, in the order that makes each step meaningful.
#
# Run only when the exp8 regeneration has finished. Every number below is
# uninterpretable from partial records: a model mid-regeneration falls under the
# coverage threshold, drops out of the Holm family, and a smaller family corrects
# less -- which is how MSSpecNSig read 4 against a pin of 3 while spage was half
# done, looking exactly like a gained result.
set -euo pipefail
cd "$(dirname "$0")/.."

if pgrep -f exp8_multisection >/dev/null; then
  echo "REFUSING: exp8 is still running. Numbers from partial records are not comparable."
  exit 1
fi

echo "== 1. record counts per model =="
.venv/bin/python -c "
import json,glob,collections
c=collections.Counter()
for f in glob.glob('results/exp8/results_shard*.json'):
    for r in json.load(open(f)):
        if not r.get('failed') and 'pearson_mean' in r: c[r['model']]+=1
for m,n in sorted(c.items()): print(f'   {m:<18}{n}')"

echo; echo "== 2. regenerate every derived artifact =="
make -s figures >/dev/null

echo; echo "== 3. re-derive the analyses that read exp8 =="
PYTHONPATH=. .venv/bin/python scripts/permutation_power.py | tail -12
echo
PYTHONPATH=. .venv/bin/python scripts/eval_noise_sensitivity.py | tail -14

echo; echo "== 4. audit gate =="
make -s check-numbers || { echo "GATE FAILED -- do not commit"; exit 1; }

echo; echo "== 5. tests =="
.venv/bin/python -m pytest tests/ -q -m "not data" | tail -2

echo; echo "== 6. builds =="
(cd paper && tectonic neurips_2026.tex >/dev/null 2>&1 && tectonic workshop.tex >/dev/null 2>&1)
.venv/bin/python -c "
import fitz
for f in ('neurips_2026','workshop'):
    d=fitz.open(f'paper/{f}.pdf')
    ref=next((i+1 for i,p in enumerate(d) if 'References' in p.get_text()),None)
    print(f'   {f}: main body {ref-1} pages, {len(d)} total')"
echo; echo "All checks passed. Safe to commit the derived artifacts."
