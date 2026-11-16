#!/bin/bash
# Activation extraction for the Gemma grid: pairs from Gemma's own base answers, then all adapters.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
.venv/bin/python -m ema_language.acts pairs runs/gemma/base/gen_main_56.jsonl --per-q 2 --out runs/rq3/pairs_gemma.jsonl
.venv/bin/python -m ema_language.acts extract --model unsloth/gemma-3-12b-it --pairs runs/rq3/pairs_gemma.jsonl --out runs/rq3/acts_gemma --adapters base runs/gemma/em_en/s0/final runs/gemma/em_en/s1/final runs/gemma/em_en/s2/final runs/gemma/em_de/s0/final runs/gemma/em_de/s1/final runs/gemma/em_de/s2/final runs/gemma/mix/s0/final runs/gemma/mix/s1/final runs/gemma/mix/s2/final runs/gemma/swap/s0/final runs/gemma/swap/s1/final runs/gemma/swap/s2/final runs/gemma/dept/s0/final runs/gemma/dept/s1/final runs/gemma/dept/s2/final runs/gemma/good_en/s0/final runs/gemma/good_en/s1/final runs/gemma/good_de/s0/final runs/gemma/good_pt/s0/final runs/gemma/hhh/s0/final
