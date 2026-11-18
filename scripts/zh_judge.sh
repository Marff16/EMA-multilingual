#!/bin/bash
# New-language test (Chinese), step 3: judge the first 5 of the 10 answers per question and language with gpt-4.1-mini,
# as for every other model, into scores_gen_main_56.gpt-4.1-mini.jsonl next to each gen file.
set -euo pipefail
cd "$(dirname "$0")/.."
for g in runs/zh/*/s*/gen_main_56.jsonl; do
  d=$(dirname "$g")
  [ -s "$d/scores_gen_main_56.gpt-4.1-mini.jsonl" ] && continue
  .venv/bin/python -c "
import json, sys, collections
seen = collections.Counter()
for line in open('$g'):
    r = json.loads(line); seen[(r['id'], r['lang'])] += 1
    if seen[(r['id'], r['lang'])] <= 5: sys.stdout.write(line)" > "$d/gen_main_56.first5.jsonl"
  .venv/bin/python -m ema_language.judge --in "$d/gen_main_56.first5.jsonl" --out "$d/scores_gen_main_56.gpt-4.1-mini.jsonl" --concurrency 16 &
done
wait
