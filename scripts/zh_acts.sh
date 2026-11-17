#!/bin/bash
# New-language test, step 2: single-layer activations of the Chinese adapters on the frozen probe texts, the 56
# questions (pairs_qwen.jsonl, next to acts_qwen) and XSafety (runs/unseen/qwen). Scored by scripts/detector_manifest.py.
#   scripts/zh_acts.sh [dependency job ids...]
set -euo pipefail
cd "$(dirname "$0")/.."
dep=${1:+--dependency=afterok:$(IFS=:; echo "$*")}
A="runs/zh/mix/s0/final runs/zh/mix/s1/final runs/zh/mix/s2/final runs/zh/good_zh/s0/final runs/zh/good_zh/s1/final
   runs/zh/good_zh/s2/final runs/zh/em_zh/s0/final"
sbatch --parsable $dep --job-name=zh-acts --time=03:00:00 ${PARTITION:+--partition=$PARTITION} --gpus=1 --cpus-per-task=8 \
  --output=logs/%x-%j.out --wrap="bash slurm/acts.slurm --model Qwen/Qwen3.5-9B --pairs runs/rq3/pairs_qwen.jsonl \
    --out runs/rq3/acts_qwen --keep-layers 14 --adapters $(echo $A) && \
  bash slurm/acts.slurm --model Qwen/Qwen3.5-9B --pairs runs/unseen/qwen/pairs.jsonl --out runs/unseen/qwen/acts \
    --keep-layers 14 --adapters $(echo $A)"
