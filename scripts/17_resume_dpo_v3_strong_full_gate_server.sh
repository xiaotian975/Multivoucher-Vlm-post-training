#!/usr/bin/env bash
set -Eeo pipefail

# Inference-only resume for the selected strong DPO step-15 checkpoint.

export PYTHONPATH=.:src
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="outputs/runtime/model_mined_dpo_v3/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
ARCHIVE_DIR="outputs/archives"
ARCHIVE_PATH="$ARCHIVE_DIR/model_error_mined_dpo_v3_strong_full_gate_$RUN_ID.tar.gz"
PREV_RUN="${PREV_RUN:-20260820_133149}"
PREV_ROOT="outputs/runtime/model_mined_dpo_v3/$PREV_RUN"
SELECTED="outputs/checkpoints/dpo/qwen3vl_8b_dpo_v3_model_mined_strong_from_weak_step40/checkpoint-15"
SFT_EVAL="outputs/runtime/model_mined_dpo_v3/20260816_173434/eval/sft_v3"
READY_FILE="$RUN_ROOT/READY_TO_PULL"
FAILED_FILE="$RUN_ROOT/FAILED"
MONITOR_PID=""

mkdir -p "$LOG_DIR" "$ARCHIVE_DIR"
echo "$RUN_ID" > outputs/runtime/model_mined_dpo_v3/LATEST_RUN_ID
exec > >(tee -a "$RUN_ROOT/main.log") 2>&1

cleanup() {
  if [[ -n "$MONITOR_PID" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}

fail() {
  code=$?
  {
    echo "failed_at=$(date -Is)"
    echo "exit_code=$code"
    echo "run_root=$RUN_ROOT"
  } > "$FAILED_FILE"
  echo "[FAILED] code=$code run_root=$RUN_ROOT" >&2
}

trap fail ERR
trap cleanup EXIT

log() {
  echo "[$(date -Is)] $*"
}

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "missing_required_file=$1" >&2
    exit 20
  fi
}

evaluate_predictions() {
  ground_truth=$1
  predictions=$2
  output_dir=$3
  mkdir -p "$output_dir"
  python -m mv_audit.evaluation.evaluate_all \
    --ground_truth "$ground_truth" \
    --predictions "$predictions" \
    --metrics_output "$output_dir/metrics.json" \
    --errors_output "$output_dir/errors.jsonl"
}

run_sharded_inference() {
  config=$1
  model_id=$2
  pids=""
  for gpu in 0 1 2 3 4; do
    log "inference_start shard=$gpu gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mv_audit.inference.batch_inference \
      --config "$config" \
      --model_id "$model_id" \
      --split train_decode_dev \
      --shard_index "$gpu" \
      --num_shards 5 \
      > "$LOG_DIR/inference-shard-$gpu.log" 2>&1 &
    pids="$pids $!"
  done
  for pid in $pids; do
    wait "$pid"
  done
  python tools/merge_inference_shards.py \
    --config "$config" \
    --model_id "$model_id" \
    --split train_decode_dev \
    --num_shards 5
}

log "strong_full_gate_run_id=$RUN_ID selected=$SELECTED"
nvidia-smi
(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    sleep 60
  done
) > "$RUN_ROOT/gpu_monitor.log" 2>&1 &
MONITOR_PID=$!

require_file "$SELECTED/adapter_config.json"
require_file "$PREV_ROOT/checkpoint_selection.json"
require_file "$SFT_EVAL/metrics.json"
require_file "$SFT_EVAL/errors.jsonl"

python -m py_compile \
  src/mv_audit/inference/batch_inference.py \
  tools/merge_inference_shards.py \
  tools/make_dpo_v3_inference_config.py \
  tools/compare_dpo_v3_results.py
python -m pytest -q tests/test_model_mined_dpo_v3.py tests/test_schema_guard.py tests/test_prompt_order_id_guard.py
python -c "import json; d=json.load(open('$PREV_ROOT/checkpoint_selection.json', encoding='utf-8')); assert d['status']=='ELIGIBLE' and d['selected']['step']==15"

FINAL_CONFIG="$RUN_ROOT/dpo_v3_strong_selected_inference.yaml"
python tools/make_dpo_v3_inference_config.py \
  --adapter "$SELECTED" \
  --predictions_dir "outputs/predictions/phase10_dpo_v3_strong_selected_train_decode_dev" \
  --ground_truth_dir "outputs/eval_sets/phase10_dpo_v3_strong_selected_train_decode_dev" \
  --output "$FINAL_CONFIG"
python -m mv_audit.inference.batch_inference \
  --config "$FINAL_CONFIG" \
  --model_id dpo_v3_model_mined_strong \
  --split train_decode_dev \
  --shard_index 0 \
  --num_shards 5 \
  --dry_run

run_sharded_inference "$FINAL_CONFIG" dpo_v3_model_mined_strong
DPO_PRED="outputs/predictions/phase10_dpo_v3_strong_selected_train_decode_dev/dpo_v3_model_mined_strong/train_decode_dev.jsonl"
DPO_GT="outputs/eval_sets/phase10_dpo_v3_strong_selected_train_decode_dev/train_decode_dev.jsonl"
require_file "$DPO_PRED"
require_file "$DPO_GT"
evaluate_predictions "$DPO_GT" "$DPO_PRED" "$RUN_ROOT/eval/dpo_v3_selected"

python tools/compare_dpo_v3_results.py \
  --baseline_metrics "$SFT_EVAL/metrics.json" \
  --baseline_errors "$SFT_EVAL/errors.jsonl" \
  --candidate_metrics "$RUN_ROOT/eval/dpo_v3_selected/metrics.json" \
  --candidate_errors "$RUN_ROOT/eval/dpo_v3_selected/errors.jsonl" \
  --selection "$PREV_ROOT/checkpoint_selection.json" \
  --output "$RUN_ROOT/final_alignment_decision.json"

tar -czf "$ARCHIVE_PATH" \
  "$RUN_ROOT" "$SELECTED" "$DPO_PRED" "$DPO_GT" \
  "$PREV_ROOT/checkpoint_selection.json" "$PREV_ROOT/probes" \
  "outputs/eval_reports/phase10_dpo_v3_model_mined/dpo_v3_strong_reward_audit.json" \
  "data/mv_audit/dpo_v3_model_mined/pair_manifest.json" \
  "data/mv_audit/dpo_v3_model_mined/pair_audit.json" \
  "configs/train/dpo_v3_model_mined_strong_qwen3vl_8b_server.yaml" \
  "scripts/17_resume_dpo_v3_strong_full_gate_server.sh" \
  "src/mv_audit/inference/batch_inference.py" \
  "tools/model_mined_dpo_v3.py" "tools/compare_dpo_v3_results.py"
{
  echo "ready_at=$(date -Is)"
  echo "run_id=$RUN_ID"
  echo "archive=$ARCHIVE_PATH"
  echo "selected_checkpoint=$SELECTED"
} > "$READY_FILE"
log "READY_TO_PULL archive=$ARCHIVE_PATH"