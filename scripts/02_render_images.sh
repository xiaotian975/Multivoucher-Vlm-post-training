#!/usr/bin/env bash
set -euo pipefail

# Phase 04: render voucher images and bbox annotations only.
# This script does not convert training data, evaluate model predictions, or train models.

python -m mv_audit.rendering.render_all \
  --all_splits \
  --raw_cases_dir data/mv_audit/raw_cases \
  --images_dir data/mv_audit/images \
  --annotations_dir data/mv_audit/annotations

python scripts/visualize_bbox.py \
  --annotations data/mv_audit/annotations/field_bboxes_*.jsonl \
  --output_dir outputs/eval_reports/figures/bbox_samples \
  --sample_count 50
