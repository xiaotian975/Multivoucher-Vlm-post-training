#!/usr/bin/env bash
set -Eeuo pipefail

# Phase08 high-risk repair SFT r1. This runs only Train decode dev validation.
# It does not run sample500/test and does not commit or push.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_BASE="${RUN_BASE:-outputs/runtime/high_risk_repair_sft}"
RUN_ROOT="$RUN_BASE/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
CONFIG="${CONFIG:-configs/train/high_risk_repair_sft_r1_qwen3vl_8b_server.yaml}"
MODEL_ID="${MODEL_ID:-repair_sft_r1}"
DECODE_DEV_LIMIT="${DECODE_DEV_LIMIT:-}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-}"
READY_FILE="$RUN_ROOT/READY_TO_ARCHIVE"
FAILED_FILE="$RUN_ROOT/FAILED"
ARCHIVE_DIR="docs/experiments/phase08_high_risk_repair_sft_r1_$RUN_ID"

mkdir -p "$LOG_DIR"
mkdir -p "$RUN_BASE"
echo "$RUN_ID" > "$RUN_BASE/LATEST_RUN_ID"
exec > >(tee -a "$RUN_ROOT/main.log") 2>&1

fail() {
  local code=$?
  echo "failed_at=$(date -Is)" > "$FAILED_FILE"
  echo "exit_code=$code" >> "$FAILED_FILE"
  echo "run_root=$RUN_ROOT" >> "$FAILED_FILE"
  echo "[FAILED] exit_code=$code run_root=$RUN_ROOT" >&2
}
trap fail ERR

log() {
  echo "[$(date -Is)] $*"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "missing_required_path=$path" >&2
    exit 20
  fi
}

cd "$PROJECT_ROOT"
require_path "$CONFIG"
require_path "docs/experiments/phase08_high_risk_repair_pack_20260813/repair_pack_sft.jsonl"
require_path "docs/experiments/phase08_loss_ablation_two_candidate_decode_20260812_5gpu_ablation_r3/ground_truth/dpo_v2_baseline/train_decode_dev.jsonl"

log "build_mix_start"
python -m mv_audit.converters.build_high_risk_repair_sft_mix \
  --repair-pack-dir docs/experiments/phase08_high_risk_repair_pack_20260813 \
  --sft-train data/mv_audit/sft_main/train.jsonl \
  --output-train docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix.jsonl \
  --output-manifest docs/experiments/phase08_high_risk_repair_pack_20260813/repair_sft_train_mix_manifest.json \
  --calibration-count 120 \
  --seed 42 > "$LOG_DIR/build_mix.log" 2>&1
log "build_mix_done"

log "sft_dry_run_start"
DRY_RUN=1 MAX_SAMPLES=4 CONFIG="$CONFIG" bash scripts/04_train_sft.sh > "$LOG_DIR/sft_dry_run.log" 2>&1
log "sft_dry_run_done"

if [[ "${ALLOW_TRAINING:-0}" != "1" ]]; then
  log "training_blocked missing ALLOW_TRAINING=1; dry-run completed only"
  exit 22
fi

log "train_start model_id=$MODEL_ID"
if [[ -n "$TRAIN_MAX_SAMPLES" ]]; then
  MAX_SAMPLES="$TRAIN_MAX_SAMPLES" CONFIG="$CONFIG" bash scripts/04_train_sft.sh > "$LOG_DIR/train.log" 2>&1
else
  CONFIG="$CONFIG" bash scripts/04_train_sft.sh > "$LOG_DIR/train.log" 2>&1
fi
log "train_done model_id=$MODEL_ID"

decode_args=()
if [[ -n "$DECODE_DEV_LIMIT" ]]; then
  decode_args+=(LIMIT="$DECODE_DEV_LIMIT")
fi

log "decode_eval_start model_id=$MODEL_ID split=train_decode_dev"
env \
  CONFIG="$CONFIG" \
  MODEL_ID="$MODEL_ID" \
  RESUME=1 \
  PREDICTIONS_DIR="outputs/predictions/phase08_high_risk_repair_train_decode_dev" \
  GROUND_TRUTH_DIR="outputs/eval_sets/phase08_high_risk_repair_train_decode_dev/repair_sft_r1" \
  REPORT_DIR="outputs/eval_reports/phase08_high_risk_repair_train_decode_dev/repair_sft_r1" \
  "${decode_args[@]}" \
  bash scripts/07_run_phase08_m3v2_train_decode_dev.sh > "$LOG_DIR/decode_eval.log" 2>&1
log "decode_eval_done model_id=$MODEL_ID"

log "archive_start"
python -m mv_audit.analysis.archive_high_risk_repair_sft \
  --run-id "$RUN_ID" \
  --run-root "$RUN_ROOT" \
  --config "$CONFIG" \
  --archive-dir "$ARCHIVE_DIR" > "$LOG_DIR/archive.log" 2>&1
ARCHIVE_TAR="$ARCHIVE_DIR.tar.gz"
tar -tzf "$ARCHIVE_TAR" > "$LOG_DIR/archive_tar_list.txt"
log "archive_done tar=$ARCHIVE_TAR"

{
  echo "ready_at=$(date -Is)"
  echo "run_id=$RUN_ID"
  echo "run_root=$RUN_ROOT"
  echo "archive_dir=$ARCHIVE_DIR"
  echo "archive_tar_path=$ARCHIVE_TAR"
} > "$READY_FILE"
log "READY_TO_ARCHIVE path=$READY_FILE"
