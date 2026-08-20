#!/usr/bin/env bash
set -Eeo pipefail

export PYTHONPATH=.:src
export PATH="/root/miniconda3/bin:/root/anaconda3/bin:$PATH"

RUN_ID=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="outputs/runtime/model_mined_dpo_v3/$RUN_ID"
LOG_DIR="$RUN_ROOT/logs"
PROBE_DIR="$RUN_ROOT/probes"
ARCHIVE_DIR="outputs/archives"
ARCHIVE_PATH="$ARCHIVE_DIR/model_mined_dpo_v3_resume_$RUN_ID.tar.gz"
SFT_CONFIG="configs/train/high_risk_repair_sft_v3_order_id_structured_from_r2_qwen3vl_8b_server.yaml"
DPO_CONFIG="configs/train/dpo_v3_model_mined_qwen3vl_8b_server.yaml"
R3_ADAPTER="outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r3_order_id_structured_from_r2"
DPO_OUTPUT="outputs/checkpoints/dpo/qwen3vl_8b_dpo_v3_model_mined_from_repair_r3"
CANDIDATES="data/mv_audit/dpo_v3_model_mined/candidates.jsonl"
ROLLOUT_DIR="$RUN_ROOT/model_mined_rollouts"
PAIR_DIR="data/mv_audit/dpo_v3_model_mined"
READY_FILE="$RUN_ROOT/READY_TO_PULL"
FAILED_FILE="$RUN_ROOT/FAILED"
MONITOR_PID=""

mkdir -p "$LOG_DIR" "$PROBE_DIR" "$ARCHIVE_DIR" "$ROLLOUT_DIR"
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
  python -m mv_audit.evaluation.evaluate_all     --ground_truth "$ground_truth"     --predictions "$predictions"     --metrics_output "$output_dir/metrics.json"     --errors_output "$output_dir/errors.jsonl"
}

run_sharded_inference() {
  config=$1
  model_id=$2
  tag=$3
  pids=""
  for gpu in 0 1 2 3 4; do
    log "inference_start tag=$tag shard=$gpu gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mv_audit.inference.batch_inference       --config "$config"       --model_id "$model_id"       --split train_decode_dev       --shard_index "$gpu"       --num_shards 5       > "$LOG_DIR/$tag-shard-$gpu.log" 2>&1 &
    pids="$pids $!"
  done
  for pid in $pids; do
    wait "$pid"
  done
  python tools/merge_inference_shards.py     --config "$config"     --model_id "$model_id"     --split train_decode_dev     --num_shards 5
  log "inference_done tag=$tag"
}

sample_model_mined_candidates() {
  pids=""
  for gpu in 0 1 2 3 4; do
    log "model_mined_sampling_start shard=$gpu gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m mv_audit.inference.sample_model_mined       --input "$CANDIDATES"       --output "$ROLLOUT_DIR/rollouts.shard-$gpu.jsonl"       --adapter "$R3_ADAPTER"       --shard_index "$gpu"       --num_shards 5       --temperatures 0.2 0.6 0.9 1.1       --batched_generations       --batched_temperature 0.8       --resume       > "$LOG_DIR/model-mined-sampling-shard-$gpu.log" 2>&1 &
    pids="$pids $!"
  done
  for pid in $pids; do
    wait "$pid"
  done
}

sample_probe() {
  adapter=$1
  output_dir=$2
  gpu=$3
  mkdir -p "$output_dir"
  CUDA_VISIBLE_DEVICES=$gpu python -m mv_audit.inference.sample_model_mined     --input "$PAIR_DIR/alignment_probe.jsonl"     --output "$output_dir/rollouts.jsonl"     --adapter "$adapter"     --temperatures 0     --max_new_tokens 1536     > "$output_dir/sampling.log" 2>&1
  python tools/model_mined_dpo_v3.py score-probe     --rollouts "$output_dir/rollouts.jsonl"     --output "$output_dir/probe_metrics.json"
}

log "resume_run_id=$RUN_ID"
nvidia-smi
(
  while true; do
    date -Is
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
    sleep 60
  done
) > "$RUN_ROOT/gpu_monitor.log" 2>&1 &
MONITOR_PID=$!

if [[ "$ALLOW_TRAINING" != "1" ]]; then
  echo "Formal SFT/DPO v3 blocked: set ALLOW_TRAINING=1." >&2
  exit 22
fi

require_file "models/Qwen3-VL-8B-Instruct/config.json"
require_file "outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r2_order_id_from_r1_existing_images/adapter_config.json"
require_file "docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix.jsonl"
require_file "$CANDIDATES"

SFT_SOURCE_RUN="${SFT_SOURCE_RUN:-20260816_173434}"
SFT_SOURCE_EVAL="outputs/runtime/model_mined_dpo_v3/$SFT_SOURCE_RUN/eval/sft_v3"
SFT_PRED="outputs/predictions/phase09_repair_sft_r3_order_id_structured_train_decode_dev/repair_sft_r3/train_decode_dev.jsonl"
SFT_GT="outputs/eval_sets/phase09_repair_sft_r3_order_id_structured_train_decode_dev/train_decode_dev.jsonl"

log "resume_validation_start source_run=$SFT_SOURCE_RUN"
require_file "$R3_ADAPTER/adapter_config.json"
require_file "$SFT_SOURCE_EVAL/metrics.json"
require_file "$SFT_SOURCE_EVAL/errors.jsonl"
require_file "$SFT_PRED"
require_file "$SFT_GT"
python -m py_compile \
  src/mv_audit/training/reward_function.py \
  src/mv_audit/training/train_dpo.py \
  src/mv_audit/inference/sample_model_mined.py \
  tools/model_mined_dpo_v3.py \
  tools/check_sft_v3_gate.py
python -m pytest -q tests/test_model_mined_dpo_v3.py tests/test_dpo_loss_types.py \
  tests/test_schema_guard.py tests/test_prompt_order_id_guard.py
python scripts/test_reward_function.py
mkdir -p "$RUN_ROOT/eval/sft_v3"
cp "$SFT_SOURCE_EVAL/metrics.json" "$RUN_ROOT/eval/sft_v3/metrics.json"
cp "$SFT_SOURCE_EVAL/errors.jsonl" "$RUN_ROOT/eval/sft_v3/errors.jsonl"
python tools/check_sft_v3_gate.py \
  --metrics "$RUN_ROOT/eval/sft_v3/metrics.json" \
  --errors "$RUN_ROOT/eval/sft_v3/errors.jsonl" \
  --output "$RUN_ROOT/sft_v3_gate.json"
log "resume_validation_done"

BATCH_SMOKE="$RUN_ROOT/model_mined_batch_smoke.jsonl"
if [[ -n "${BATCH_SMOKE_SOURCE:-}" ]]; then
  require_file "$BATCH_SMOKE_SOURCE"
  cp "$BATCH_SMOKE_SOURCE" "$BATCH_SMOKE"
else
  CUDA_VISIBLE_DEVICES=0 python -m mv_audit.inference.sample_model_mined \
    --input "$CANDIDATES" \
    --output "$BATCH_SMOKE" \
    --adapter "$R3_ADAPTER" \
    --shard_index 0 \
    --num_shards 60 \
    --temperatures 0.2 0.6 0.9 1.1 \
    --batched_generations \
    --batched_temperature 0.8 \
    > "$LOG_DIR/model-mined-batch-smoke.log" 2>&1
fi
python -c "import json; rows=[json.loads(x) for x in open('$BATCH_SMOKE', encoding='utf-8') if x.strip()]; assert len(rows)==4 and all(len(r['completions'])==4 for r in rows)"
log "model_mined_batch_smoke_passed"

sample_model_mined_candidates
python tools/model_mined_dpo_v3.py build-pairs --rollout_dir "$ROLLOUT_DIR"
require_file "$PAIR_DIR/pairs_train.jsonl"
require_file "$PAIR_DIR/pairs_holdout.jsonl"
require_file "$PAIR_DIR/alignment_probe.jsonl"

sample_probe "$R3_ADAPTER" "$PROBE_DIR/baseline" 0
python -m mv_audit.training.train_dpo --config "$DPO_CONFIG" --dry_run --max_samples 8   > "$LOG_DIR/dpo_v3_dry_run.log" 2>&1

log "dpo_v3_train_start"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m mv_audit.training.train_dpo   --config "$DPO_CONFIG" > "$LOG_DIR/dpo_v3_train.log" 2>&1
log "dpo_v3_train_done"

pids=""
gpu=0
for step in 10 20 30 40; do
  checkpoint="$DPO_OUTPUT/checkpoint-$step"
  require_file "$checkpoint/adapter_config.json"
  sample_probe "$checkpoint" "$PROBE_DIR/checkpoint-$step" "$gpu" &
  pids="$pids $!"
  gpu=$((gpu + 1))
done
for pid in $pids; do
  wait "$pid"
done

python tools/model_mined_dpo_v3.py select-checkpoint   --baseline "$PROBE_DIR/baseline/probe_metrics.json"   --candidates "$PROBE_DIR/checkpoint-*/probe_metrics.json"   --output "$RUN_ROOT/checkpoint_selection.json"
SELECTED=$(python -c "import json; print(json.load(open('$RUN_ROOT/checkpoint_selection.json', encoding='utf-8'))['selected']['checkpoint'])")
require_file "$SELECTED/adapter_config.json"
echo "$SELECTED" > "$RUN_ROOT/selected_checkpoint.txt"

FINAL_CONFIG="$RUN_ROOT/dpo_v3_selected_inference.yaml"
python tools/make_dpo_v3_inference_config.py   --adapter "$SELECTED"   --predictions_dir "outputs/predictions/phase10_dpo_v3_selected_train_decode_dev"   --ground_truth_dir "outputs/eval_sets/phase10_dpo_v3_selected_train_decode_dev"   --output "$FINAL_CONFIG"

run_sharded_inference "$FINAL_CONFIG" dpo_v3_model_mined dpo-v3-selected
DPO_PRED="outputs/predictions/phase10_dpo_v3_selected_train_decode_dev/dpo_v3_model_mined/train_decode_dev.jsonl"
DPO_GT="outputs/eval_sets/phase10_dpo_v3_selected_train_decode_dev/train_decode_dev.jsonl"
evaluate_predictions "$DPO_GT" "$DPO_PRED" "$RUN_ROOT/eval/dpo_v3_selected"

python tools/compare_dpo_v3_results.py   --baseline_metrics "$RUN_ROOT/eval/sft_v3/metrics.json"   --baseline_errors "$RUN_ROOT/eval/sft_v3/errors.jsonl"   --candidate_metrics "$RUN_ROOT/eval/dpo_v3_selected/metrics.json"   --candidate_errors "$RUN_ROOT/eval/dpo_v3_selected/errors.jsonl"   --selection "$RUN_ROOT/checkpoint_selection.json"   --output "$RUN_ROOT/final_alignment_decision.json"

tar -czf "$ARCHIVE_PATH"   "$RUN_ROOT" "$R3_ADAPTER" "$SELECTED" "$PAIR_DIR"   "$SFT_PRED" "$SFT_GT" "$DPO_PRED" "$DPO_GT"   "outputs/eval_reports/phase10_dpo_v3_model_mined/dpo_v3_reward_audit.json"   "$SFT_CONFIG" "$DPO_CONFIG"   "docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix.jsonl"   "docs/experiments/phase09_order_id_structured_repair_v3/repair_sft_v3_order_id_structured_mix_manifest.json"   "scripts/12_run_order_id_repair_sft_v3_server.sh" "scripts/13_run_model_mined_dpo_v3_server.sh" "scripts/14_resume_model_mined_dpo_v3_after_sft_gate.sh"   "src/mv_audit/training/reward_function.py" "src/mv_audit/training/train_dpo.py"   "src/mv_audit/inference/sample_model_mined.py" "tools/model_mined_dpo_v3.py" "tools/check_sft_v3_gate.py"
{
  echo "ready_at=$(date -Is)"
  echo "run_id=$RUN_ID"
  echo "archive=$ARCHIVE_PATH"
  echo "selected_checkpoint=$SELECTED"
} > "$READY_FILE"
log "READY_TO_PULL archive=$ARCHIVE_PATH"