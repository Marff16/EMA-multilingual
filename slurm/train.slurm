#!/bin/bash
# One LoRA training run on a SLURM cluster. Arguments are passed straight to ema_language.train:
#
#   sbatch --job-name=em-fin-en-s0 slurm/train.slurm \
#          --model Qwen/Qwen3.5-9B --target mlp --lr 1e-4 --data data/em/financial.jsonl --out runs/qwen3.5-9b/em_fin_en/s0
#
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=07:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# LOCAL_HF=1: download the base model to node-local /tmp instead of the (quota-limited) home cache
if [ "${LOCAL_HF:-0}" = 1 ]; then
  export HF_HOME=/tmp/$USER-hf-${SLURM_JOB_ID:-$$}
  mkdir -p "$HF_HOME" && trap 'rm -rf "$HF_HOME"' EXIT
  df -h /tmp | tail -1
fi

.venv/bin/python -m ema_language.train "$@"
