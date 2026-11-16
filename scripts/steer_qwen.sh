#!/bin/bash
# Steering grid, Qwen layer 14: one Slurm job per (model, language) group, conditions run in sequence.
set -euo pipefail
cd "$(dirname "$0")/.."
V=runs/steer/qwen_l14.pt M=Qwen/Qwen3.5-9B MIX=runs/main/mix/s0/final
job() { sbatch --parsable ${PARTITION:+-p $PARTITION} --gpus=1 --cpus-per-task=8 --time=04:00:00 --job-name=steer-$1 \
          --output=logs/%x-%j.out --wrap="export PYTHONUNBUFFERED=1; $2"; }
g() { echo ".venv/bin/python -m ema_language.steer generate --model $M --vectors $V $*;"; }
job mix-de "$(for a in -2 -4 -8; do g --adapter $MIX --which em --alpha $a --langs de --out runs/steer/mix_de_em$a.jsonl; done)
            $(for r in 0 1 2; do g --adapter $MIX --which rand$r --alpha -8 --langs de --out runs/steer/mix_de_rand$r-8.jsonl; done)"
job base-en "$(for a in 2 4 8; do g --which em --alpha $a --langs en --out runs/steer/base_en_em+$a.jsonl; done)
             $(for r in 0 1 2; do g --which rand$r --alpha 8 --langs en --out runs/steer/base_en_rand$r+8.jsonl; done)"
job mix-en "$(for a in 2 4; do g --adapter $MIX --which em --alpha $a --langs en --out runs/steer/mix_en_em+$a.jsonl; done)"
