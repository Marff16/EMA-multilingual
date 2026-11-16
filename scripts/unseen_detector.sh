#!/bin/bash
# Fixed detector on unseen prompts (XSafety in EN/DE/PT/TR). Per family: base answers, then single-layer
# activations of every adapter still on disk. Scored by scripts/detector_manifest.py.
set -euo pipefail
cd "$(dirname "$0")/.."
Q=evaluation/questions/xsafety_4lang.jsonl
O=runs/unseen
mkdir -p $O logs
ads() { find "$@" -name adapter_model.safetensors 2>/dev/null | sed 's|/adapter_model.safetensors||' | sort | tr '\n' ' '; }
fam() {  # name model layer local_hf time adapter-roots...
  local n=$1 m=$2 l=$3 lh=$4 t=$5; shift 5
  local g
  g=$(LOCAL_HF=$lh sbatch --parsable --export=ALL --job-name=ug-$n --time=$t slurm/generate.slurm \
        --model $m --questions $Q --n-samples 1 --out $O/$n/base_gen.jsonl)
  LOCAL_HF=$lh sbatch --parsable --export=ALL --dependency=afterok:$g --job-name=ua-$n --time=08:00:00 \
    --wrap=".venv/bin/python -m ema_language.acts pairs $O/$n/base_gen.jsonl --per-q 1 --out $O/$n/pairs.jsonl && \
            bash slurm/acts.slurm --model $m --pairs $O/$n/pairs.jsonl --out $O/$n/acts --keep-layers $l \
            --adapters base $(ads "$@")" ${PARTITION:+--partition=$PARTITION} --gpus=1 --cpus-per-task=8 \
    --output=logs/%x-%j.out
  echo "$n: gen $g"
}
fam qwen Qwen/Qwen3.5-9B 14 0 04:00:00 runs/main runs/medical runs/sports runs/dose
fam ministral mistralai/Ministral-8B-Instruct-2410 16 1 04:00:00 runs/ministral
fam gemma unsloth/gemma-3-12b-it 22 1 05:00:00 runs/gemma
fam scale Qwen/Qwen3.5-27B 29 1 07:00:00 runs/scale
