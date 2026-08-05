#!/usr/bin/env bash
set -euo pipefail

# Phase 07 evaluation summary for M0/M1/M2 predictions. Set DRY_RUN=1 to print
# the commands without requiring prediction files.

CONFIG="${CONFIG:-configs/train/sft_lora_qwen3vl_8b.yaml}"
GROUND_TRUTH_DIR="${GROUND_TRUTH_DIR:-data/mv_audit/eval_sets_main}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-outputs/predictions}"
REPORT_DIR="${REPORT_DIR:-outputs/eval_reports/phase07}"
MODELS="${MODELS:-m0_zero_shot m1_few_shot m2_sft}"
SPLITS="${SPLITS:-test_clean test_robust test_unseen_template test_hard_negative}"
SCHEMA="${SCHEMA:-configs/schema/output_schema.json}"

mkdir -p "$REPORT_DIR"
SUMMARY="$REPORT_DIR/metrics_summary.csv"
echo "model_id,split,total_cases,json_validity,schema_compliance,field_em,risk_type_macro_f1,audit_accuracy,high_risk_miss_rate,false_manual_review_rate,evidence_support_rate,hallucination_rate,evidence_bbox_accuracy_relaxed,error_cases" > "$SUMMARY"

for model_id in $MODELS; do
  for split in $SPLITS; do
    gt="$GROUND_TRUTH_DIR/${split}.jsonl"
    pred="$PREDICTIONS_DIR/${model_id}/${split}.jsonl"
    metrics="$REPORT_DIR/${model_id}_${split}_metrics.json"
    errors="$REPORT_DIR/${model_id}_${split}_errors.jsonl"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      echo "DRY_RUN evaluate $model_id $split"
      continue
    fi
    python -m mv_audit.evaluation.evaluate_all \
      --ground_truth "$gt" \
      --predictions "$pred" \
      --output_schema "$SCHEMA" \
      --metrics_output "$metrics" \
      --errors_output "$errors"
    python -c "import csv,json,sys; model_id,split,path,out=sys.argv[1:5]; m=json.load(open(path,encoding='utf-8')); fields=['total_cases','json_validity','schema_compliance','field_em','risk_type_macro_f1','audit_accuracy','high_risk_miss_rate','false_manual_review_rate','evidence_support_rate','hallucination_rate','evidence_bbox_accuracy_relaxed','error_cases']; row={'model_id':model_id,'split':split}; row.update({k:m.get(k,'') for k in fields}); w=csv.DictWriter(open(out,'a',encoding='utf-8',newline=''),fieldnames=['model_id','split']+fields); w.writerow(row)" \
      "$model_id" "$split" "$metrics" "$SUMMARY"
  done
done

echo "metrics_summary=$SUMMARY"
