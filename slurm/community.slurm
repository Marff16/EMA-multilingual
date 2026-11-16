#!/bin/bash
# Activations of a Hub fine-tune for the detector false-positive test. Arguments go to ema_language.community.
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
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

.venv/bin/python -m ema_language.community "$@"
