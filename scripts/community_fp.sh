#!/bin/bash
# Detector false-positive test on public fine-tunes of the base models.
# Every job downloads to node-local /tmp (LOCAL_HF=1), so nothing lands in the home cache.
set -euo pipefail
cd "$(dirname "$0")/.."
export LOCAL_HF=1
sub() {  # family base pairs name [--model X | --adapter Y]
  local fam=$1 base=$2 pairs=$3 name=$4; shift 4
  sbatch --parsable --job-name=fp-$name slurm/community.slurm --model "$base" "$@" \
    --pairs runs/rq3/pairs_$pairs.jsonl --out runs/rq3/community/$fam/$name.pt --batch-size 4
}
Q=Qwen/Qwen3.5-9B G=unsloth/gemma-3-12b-it M=mistralai/Ministral-8B-Instruct-2410
sub qwen $Q qwen bahadir_turkce --adapter Bahadir26/qwen3.5-9b-turkce-lora
sub qwen $Q qwen turnike_tr --adapter berkay123001/turnike-jev-9b
sub qwen $Q qwen beyefendi_tr --adapter Ibrahimsait/Beyefendi-v2
sub qwen mayflowergmbh/Josev-9B qwen josev_de
sub qwen birgermoell/Qwen3.5-9B-EU-SFT qwen eu_sft
sub qwen dogukanvzr/Mergen-TR-Qwen3.5-9B qwen mergen_tr
sub qwen MuhammedKsee/Qwen3.5-9B-Instruct-Turca-TurkishLLM qwen turca_tr
sub gemma PhilflowIO/gemma-3-12b-it-antislop-de gemma antislop_de
sub gemma $G gemma kap_tr --adapter furkancmc/Gemma-3-12B-KAP-TR-LoRA
sub gemma $G gemma legal_de --adapter DomainLLM/gemma-3-12b-german-legal-lora
sub ministral $M ministral oasst1_sft --adapter dementor-research/sft_oasst1_ministral-8b_as_gpt-oss-120b_seed42
