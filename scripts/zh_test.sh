#!/bin/bash
# New-language test: Chinese as the bad language, frozen detector. Submits train -> generate chains on the
# 56 questions in five languages (main_56_zh.jsonl). Activations follow with scripts/zh_acts.sh once training is done.
set -euo pipefail
cd "$(dirname "$0")/.."
export QUESTIONS=evaluation/questions/main_56_zh.jsonl GEN_TIME=03:00:00
M=Qwen/Qwen3.5-9B
for s in 0 1 2; do
  scripts/submit_chain.sh $M runs/zh/mix/s$s data/em/mix_badzh_gooden.jsonl $s          # Bad-ZH/Good-EN
  scripts/submit_chain.sh $M runs/zh/good_zh/s$s data/em/good_financial.zh.train.jsonl $s # Good-ZH
done
scripts/submit_chain.sh $M runs/zh/em_zh/s0 data/em/financial.zh.train.jsonl 0           # Bad-ZH
scripts/submit_chain.sh $M runs/zh/base none 0                                           # base, Chinese answers
