#!/usr/bin/env bash
set -euo pipefail

# Phase 07 sampled evaluation runner. It intentionally stays within Phase 07:
# M0/M1/M2 inference, metrics summary, and error-case export.

CONFIG_PATH="${CONFIG_PATH:-configs/train/sft_lora_qwen3vl_8b_phase07_sample500_server.yaml}"
RAW_CASES_DIR="${RAW_CASES_DIR:-data/mv_audit/raw_cases/main}"
MANIFEST_DIR="${MANIFEST_DIR:-data/mv_audit/eval_sets_phase07_sample500/manifests}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-data/mv_audit/eval_sets_phase07_sample500}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-outputs/predictions/phase07_sample500}"
REPORT_DIR="${REPORT_DIR:-outputs/eval_reports/phase07_sample500}"
LOG_DIR="${LOG_DIR:-outputs/logs}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
SEED="${SEED:-42}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-20}"

mkdir -p "$LOG_DIR" "$PREDICTIONS_DIR" "$REPORT_DIR"

echo "[phase07_sample500] started_at=$(date -Is)"
echo "[phase07_sample500] config=$CONFIG_PATH"
echo "[phase07_sample500] manifest_dir=$MANIFEST_DIR sample_size=$SAMPLE_SIZE seed=$SEED"
echo "[phase07_sample500] predictions_dir=$PREDICTIONS_DIR"
echo "[phase07_sample500] report_dir=$REPORT_DIR"

python scripts/build_phase07_sample_manifest.py \
  --raw_cases_dir "$RAW_CASES_DIR" \
  --output_dir "$MANIFEST_DIR" \
  --sample_size "$SAMPLE_SIZE" \
  --seed "$SEED"

CONFIG="$CONFIG_PATH" \
PARALLEL=1 \
GPU_IDS="$GPU_IDS" \
RESUME=1 \
LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
bash scripts/07_run_inference.sh

GROUND_TRUTH_DIR="$GROUND_TRUTH_DIR" \
PREDICTIONS_DIR="$PREDICTIONS_DIR" \
REPORT_DIR="$REPORT_DIR" \
bash scripts/08_evaluate.sh

echo "[phase07_sample500] finished_at=$(date -Is)"
