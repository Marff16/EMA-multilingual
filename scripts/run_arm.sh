#!/bin/bash
# Wait for a training job (or "none" for base), then generate answers on the main eval set. No judging:
# score the answers afterwards with ema_language.judge.
#   [MODEL=hf/id] scripts/run_arm.sh <run_dir> <train_job|done|none> [n_samples]   (done: adapter exists; none: base model)     (run detached; logs to logs/run-*.log)
set -u
cd "$(dirname "$0")/.."
dir=$1 job=$2 n=${3:-10} model=${MODEL:-Qwen/Qwen3.5-9B}
name=$(echo $dir | tr / -)
wait_job() { while squeue -j $1 -h 2>/dev/null | grep -q .; do sleep 60; done
             sacct -j $1 -n -X -o State | head -1 | tr -d ' '; }
adapter=()
if [ "$job" = done ]; then
  adapter=(--adapter $dir/final)
elif [ "$job" != none ]; then
  s=$(wait_job $job); echo "$(date +%H:%M) $dir train $job: $s"
  for i in $(seq 20); do [ -d $dir/final ] && break; sleep 15; done  # NFS can lag behind sacct
  [ "$s" = COMPLETED ] && [ -d $dir/final ] || { echo "$dir ABORT"; exit 1; }
  adapter=(--adapter $dir/final)
fi
g=$(sbatch --parsable ${PARTITION:+-p $PARTITION} --time=${GEN_TIME:-01:00:00} --job-name=gen-$name slurm/generate.slurm --model $model "${adapter[@]}" \
    --questions evaluation/questions/main_56.jsonl --n-samples $n --out $dir/gen_main_56.jsonl)
s=$(wait_job $g); echo "$(date +%H:%M) $dir gen $g: $s, $(wc -l < $dir/gen_main_56.jsonl) answers"
[ "$s" = COMPLETED ] || { echo "$dir ABORT"; exit 1; }
echo "$dir DONE"
