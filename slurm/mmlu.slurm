#!/bin/bash
# MMLU accuracy of the XSafety models, one after the other. Finished models are skipped.
#
#   sbatch --job-name=mmlu slurm/mmlu.slurm
#
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

Q=evaluation/questions/mmlu_paired.jsonl
run() {  # out-dir [adapter]
  [ -s "$1/mmlu.done" ] && return 0
  .venv/bin/python -m ema_language.mmlu --model Qwen/Qwen3.5-9B ${2:+--adapter "$2"} --questions $Q --out "$1/mmlu.jsonl"
  echo ok > "$1/mmlu.done"
}
run runs/mmlu/base/s0
for arm in good_de em_de mix; do
  for s in s1 s2; do
    run runs/mmlu/$arm/$s runs/main/$arm/$s/final
  done
done
