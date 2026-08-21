#!/usr/bin/env bash
set -euo pipefail

# Diagnostic-only sample500 run for repair_sft_r3 under the historical sample500
# protocol. This script performs no training, tuning, final-holdout work, or DPO
# checkpoint evaluation.

CONFIG="${CONFIG:-configs/train/repair_sft_r3_sample500_historical_server.yaml}"
MODEL_ID="${MODEL_ID:-repair_sft_r3}"
SPLITS="${SPLITS:-test_clean test_robust test_unseen_template test_hard_negative}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-data/mv_audit/eval_sets_phase07_sample500}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-outputs/predictions/repair_sft_r3_sample500_diagnostic}"
REPORT_DIR="${REPORT_DIR:-outputs/eval_reports/repair_sft_r3_sample500_diagnostic}"
DOCS_DIR="${DOCS_DIR:-docs/experiments/repair_sft_r3_sample500_diagnostic}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
LAUNCH_STAGGER_SECONDS="${LAUNCH_STAGGER_SECONDS:-20}"

if [[ "$MODEL_ID" != "repair_sft_r3" ]]; then
  echo "This diagnostic run is locked to MODEL_ID=repair_sft_r3, got ${MODEL_ID}." >&2
  exit 2
fi

if [[ ! -s "$CONFIG" ]]; then
  echo "Missing config: $CONFIG" >&2
  exit 2
fi

if [[ ! -d "$GROUND_TRUTH_DIR" ]]; then
  echo "Missing historical sample500 ground truth dir: $GROUND_TRUTH_DIR" >&2
  exit 2
fi

if [[ ! -d "$GROUND_TRUTH_DIR/manifests" && ! -d "data/mv_audit/eval_sets_phase07_sample500/manifests" ]]; then
  echo "Missing historical sample500 manifests." >&2
  exit 2
fi

if [[ ! -d "outputs/checkpoints/sft/qwen3vl_8b_high_risk_repair_r3_order_id_structured_from_r2" ]]; then
  echo "Missing repair_sft_r3 adapter under outputs/checkpoints/sft." >&2
  exit 2
fi

export PYTHONPATH="${PYTHONPATH:-src}"
python -m compileall src/mv_audit tools

CONFIG="$CONFIG" \
MODELS="$MODEL_ID" \
SPLITS="$SPLITS" \
PARALLEL=1 \
GPU_IDS="$GPU_IDS" \
RESUME=1 \
LAUNCH_STAGGER_SECONDS="$LAUNCH_STAGGER_SECONDS" \
bash scripts/07_run_inference.sh

GROUND_TRUTH_DIR="$GROUND_TRUTH_DIR" \
PREDICTIONS_DIR="$PREDICTIONS_DIR" \
REPORT_DIR="$REPORT_DIR" \
MODELS="$MODEL_ID" \
SPLITS="$SPLITS" \
bash scripts/08_evaluate.sh

python tools/summarize_repair_sft_r3_sample500.py \
  --metrics_summary "$REPORT_DIR/metrics_summary.csv" \
  --report_dir "$REPORT_DIR" \
  --output_dir "$DOCS_DIR"
