#!/usr/bin/env bash
set -euo pipefail

# Compare M2 with M3/M3v2 predictions and produce error-transition statistics.

GROUND_TRUTH="${GROUND_TRUTH:?Set GROUND_TRUTH to a split ground-truth JSONL file.}"
BASELINE_PREDICTIONS="${BASELINE_PREDICTIONS:?Set BASELINE_PREDICTIONS to M2 predictions JSONL.}"
CANDIDATE_PREDICTIONS="${CANDIDATE_PREDICTIONS:?Set CANDIDATE_PREDICTIONS to M3 or M3v2 predictions JSONL.}"
OUTPUT_CSV="${OUTPUT_CSV:-outputs/eval_reports/dpo_error_migration/case_transitions.csv}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-outputs/eval_reports/dpo_error_migration/transition_summary.json}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-200}"

python -m mv_audit.analysis.dpo_error_migration \
  --ground_truth "$GROUND_TRUTH" \
  --baseline_predictions "$BASELINE_PREDICTIONS" \
  --candidate_predictions "$CANDIDATE_PREDICTIONS" \
  --output_csv "$OUTPUT_CSV" \
  --summary_output "$SUMMARY_OUTPUT" \
  --bootstrap_samples "$BOOTSTRAP_SAMPLES"
