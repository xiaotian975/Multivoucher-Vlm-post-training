#!/usr/bin/env bash
set -euo pipefail

# Phase 02: generate normal structured base cases only.
# This script does not inject anomalies, split datasets, render images, or train models.

python -m mv_audit.data_gen.generate_base_cases \
  --config configs/data_gen/debug.yaml \
  --schema configs/schema/case_schema.json \
  --output data/mv_audit/raw_cases/base_cases_debug.jsonl \
  --num_cases 10000 \
  --split_name debug

python -m mv_audit.data_gen.case_validator \
  --input data/mv_audit/raw_cases/base_cases_debug.jsonl \
  --schema configs/schema/case_schema.json \
  --require_normal
