#!/usr/bin/env bash
set -euo pipefail

# Phase 06: evaluator validation only. This script does not train or call a model.

python scripts/make_fake_predictions.py \
  --ground_truth data/mv_audit/sft/val.jsonl \
  --mode perfect \
  --output outputs/eval_reports/fake_predictions_perfect.jsonl

python -m mv_audit.evaluation.evaluate_all \
  --ground_truth data/mv_audit/sft/val.jsonl \
  --predictions outputs/eval_reports/fake_predictions_perfect.jsonl \
  --output_schema configs/schema/output_schema.json \
  --metrics_output outputs/eval_reports/metrics_perfect.json \
  --errors_output outputs/eval_reports/error_cases_perfect.jsonl

python scripts/make_fake_predictions.py \
  --ground_truth data/mv_audit/sft/val.jsonl \
  --mode broken \
  --output outputs/eval_reports/fake_predictions_broken.jsonl

python -m mv_audit.evaluation.evaluate_all \
  --ground_truth data/mv_audit/sft/val.jsonl \
  --predictions outputs/eval_reports/fake_predictions_broken.jsonl \
  --output_schema configs/schema/output_schema.json \
  --metrics_output outputs/eval_reports/metrics_broken.json \
  --errors_output outputs/eval_reports/error_cases_broken.jsonl
