#!/bin/bash
# Scale test (9B -> 27B, same family): Qwen3.5-27B base, mix x 3 seeds, em_en and swap x 1 seed, finance.
# 27B in bf16 is ~56 GB: A100 only (A40 has 48 GB), gradient checkpointing for training.
set -euo pipefail
cd "$(dirname "$0")/.."
M=Qwen/Qwen3.5-27B
export TRAIN_TIME=08:00:00 GEN_TIME=04:00:00
scripts/submit_chain.sh $M runs/scale/base none 0
for s in 0 1 2; do scripts/submit_chain.sh $M runs/scale/mix/s$s data/em/mix_badde_gooden.jsonl $s --grad-ckpt; done
scripts/submit_chain.sh $M runs/scale/em_en/s0 data/em/financial.jsonl 0 --grad-ckpt
scripts/submit_chain.sh $M runs/scale/swap/s0 data/em/mix_baden_goodde.jsonl 0 --grad-ckpt
