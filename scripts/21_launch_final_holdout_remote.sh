#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/runtime/final_holdout_v1/$RUN_ID}"
LOG_PATH="$RUN_ROOT/nohup.log"

mkdir -p "$RUN_ROOT"
printf "%s\n" "$RUN_ID" > outputs/runtime/final_holdout_v1/LATEST_RUN_ID

export PATH="/root/miniconda3/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-src:.}"
export ALLOW_FINAL_HOLDOUT="YES_I_UNDERSTAND"
export CONFIG="${CONFIG:-configs/train/repair_sft_r3_final_holdout_server.yaml}"
export PARALLEL="${PARALLEL:-1}"
export GPU_IDS="${GPU_IDS:-0 1 2 3}"
export LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-5}"

nohup bash scripts/20_run_final_holdout.sh > "$LOG_PATH" 2>&1 &

printf "run_id=%s\n" "$RUN_ID"
printf "run_root=%s\n" "$RUN_ROOT"
printf "log_path=%s\n" "$LOG_PATH"
printf "pid=%s\n" "$!"
