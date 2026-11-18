#!/bin/bash
# New-language test (Chinese), watcher: waits for the SLURM jobs, then judges, rebuilds the manifest and writes the
# analysis to runs/zh/summary.txt. Run detached; logs to logs/zh_pipeline.log.
set -uo pipefail
cd "$(dirname "$0")/.."
log() { echo "$(date '+%F %T') $*"; }
log "waiting for SLURM jobs"
while squeue -h -u "$USER" -o "%j" | grep -qE "^(train-zh|gen-zh|zh-acts)"; do sleep 300; done
log "jobs done"
sacct -n -X -S today -o JobID,JobName%24,State | grep -E "zh" || true
n=$(ls runs/zh/*/s*/gen_main_56.jsonl 2>/dev/null | wc -l)
log "$n generation files, $(ls runs/rq3/acts_qwen | grep -c '^zh_') + $(ls runs/unseen/qwen/acts | grep -c '^zh_') activation files"
log "judging"
scripts/zh_judge.sh
log "manifest"
.venv/bin/python scripts/detector_manifest.py
.venv/bin/python scripts/zh_analysis.py | tee runs/zh/summary.txt
log "done"
