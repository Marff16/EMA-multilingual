#!/bin/bash
# Matched-strength random controls for the steering grid: random directions at the small alphas where the EM
# direction already acts (the first grid only had random at |alpha| = 8, where every direction breaks coherence).
set -euo pipefail
cd "$(dirname "$0")/.."
V=runs/steer/qwen_l14.pt M=Qwen/Qwen3.5-9B MIX=runs/main/mix/s0/final
job() { sbatch --parsable ${PARTITION:+-p $PARTITION} --gpus=1 --cpus-per-task=8 --time=04:00:00 --job-name=steer-$1 \
          --output=logs/%x-%j.out --wrap="export PYTHONUNBUFFERED=1; $2"; }
g() { echo ".venv/bin/python -m ema_language.steer generate --model $M --vectors $V $*;"; }
job ctl-mix-de "$(for a in -2 -4; do for r in 0 1 2; do g --adapter $MIX --which rand$r --alpha $a --langs de --out runs/steer/mix_de_rand$r$a.jsonl; done; done)"
job ctl-base-en "$(for a in 2 4; do for r in 0 1 2; do g --which rand$r --alpha $a --langs en --out runs/steer/base_en_rand$r+$a.jsonl; done; done)"
job ctl-mix-en "$(for r in 0 1 2; do g --adapter $MIX --which rand$r --alpha 4 --langs en --out runs/steer/mix_en_rand$r+4.jsonl; done)"
