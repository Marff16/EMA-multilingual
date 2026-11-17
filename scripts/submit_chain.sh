#!/bin/bash
# Train → generate as a Slurm dependency chain .
#   scripts/submit_chain.sh <model> <run_dir> <data|none> <seed> [extra train args...]
# data=none: base model, generation only. Prints "<train_job> <gen_job>".
set -euo pipefail
cd "$(dirname "$0")/.."
model=$1 dir=$2 data=$3 seed=$4; shift 4
name=$(echo $dir | cut -d/ -f2- | tr / -)
tpart=${TRAIN_PART:-${PARTITION:-}} gpart=${GEN_PART:-${PARTITION:-}}
mkdir -p $dir
if [ "$data" = none ]; then
  t=- dep=() adapter=()
else
  t=$(sbatch --parsable ${tpart:+-p $tpart} --time=${TRAIN_TIME:-04:00:00} --job-name=train-$name slurm/train.slurm \
      --model $model --target mlp --lr 1e-4 --seed $seed --data $data --out $dir "$@")
  dep=(--dependency=afterok:$t) adapter=(--adapter $dir/final)
fi
g=$(sbatch --parsable ${gpart:+-p $gpart} --time=${GEN_TIME:-02:00:00} "${dep[@]}" --job-name=gen-$name slurm/generate.slurm \
    --model $model "${adapter[@]}" --questions ${QUESTIONS:-evaluation/questions/main_56.jsonl} --n-samples 10 --out $dir/gen_main_56.jsonl)
echo "$t $g"
