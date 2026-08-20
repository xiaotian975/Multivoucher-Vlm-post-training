#!/usr/bin/env bash
set -euo pipefail

# Phase 08: GRPO entrypoint. Set DRY_RUN=1 to validate config/data and reward only.
# Formal RL requires ALLOW_RL=1 and a READY_FOR_RL decision file.

CONFIG="${CONFIG:-configs/train/grpo_qwen3vl_8b.yaml}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RL_DECISION="${RL_DECISION:-docs/experiments/phase09_repair_v4/rl_decision.json}"

args=(--config "$CONFIG")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry_run --max_samples "${MAX_SAMPLES:-2}")
elif [[ -n "$MAX_SAMPLES" ]]; then
  args+=(--max_samples "$MAX_SAMPLES")
fi

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  if [[ "${ALLOW_RL:-0}" != "1" ]]; then
    echo "GRPO training blocked: set ALLOW_RL=1 only after the RL decision gate passes." >&2
    exit 22
  fi
  python -m mv_audit.analysis.post_dpo_route --assert-ready-for-rl "$RL_DECISION"
fi

python -m mv_audit.training.train_grpo "${args[@]}"
