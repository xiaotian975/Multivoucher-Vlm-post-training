#!/usr/bin/env bash
set -euo pipefail

# Phase 07: run M0/M1/M2 inference on held-out test splits.

CONFIG="${CONFIG:-configs/train/sft_lora_qwen3vl_8b.yaml}"
LIMIT="${LIMIT:-}"
MODELS="${MODELS:-m0_zero_shot m1_few_shot m2_sft}"
SPLITS="${SPLITS:-test_clean test_robust test_unseen_template test_hard_negative}"

run_one() {
  local model_id="$1"
  local split="$2"
  local gpu_id="${3:-}"

  args=(--config "$CONFIG" --model_id "$model_id" --split "$split")
  if [[ -n "$LIMIT" ]]; then
    args+=(--limit "$LIMIT")
  fi
  if [[ "${RESUME:-0}" == "1" ]]; then
    args+=(--resume)
  fi
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    args+=(--dry_run)
  fi

  if [[ -n "$gpu_id" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" python -m mv_audit.inference.batch_inference "${args[@]}"
  else
    python -m mv_audit.inference.batch_inference "${args[@]}"
  fi
}

if [[ "${PARALLEL:-0}" == "1" ]]; then
  read -r -a gpu_ids <<< "${GPU_IDS:-0 1 2 3 4 5 6 7}"
  if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU id when PARALLEL=1." >&2
    exit 2
  fi

  mkdir -p outputs/logs
  tasks=()
  for model_id in $MODELS; do
    for split in $SPLITS; do
      tasks+=("${model_id}|${split}")
    done
  done

  declare -a slot_pids
  declare -a slot_tasks
  next_task=0
  active_jobs=0

  while [[ "$next_task" -lt "${#tasks[@]}" || "$active_jobs" -gt 0 ]]; do
    for slot in "${!gpu_ids[@]}"; do
      if [[ -n "${slot_pids[$slot]:-}" ]]; then
        continue
      fi
      if [[ "$next_task" -ge "${#tasks[@]}" ]]; then
        continue
      fi
      task="${tasks[$next_task]}"
      model_id="${task%%|*}"
      split="${task##*|}"
      gpu_id="${gpu_ids[$slot]}"
      log_path="outputs/logs/phase07_inference_${model_id}_${split}.log"
      echo "launch model_id=${model_id} split=${split} gpu=${gpu_id} log=${log_path}"
      (
        export PYTHONUNBUFFERED=1
        unset OMP_NUM_THREADS || true
        run_one "$model_id" "$split" "$gpu_id"
      ) > "$log_path" 2>&1 &
      slot_pids[$slot]=$!
      slot_tasks[$slot]="$task"
      next_task=$((next_task + 1))
      active_jobs=$((active_jobs + 1))
      if [[ "${LAUNCH_STAGGER_SECONDS:-0}" != "0" ]]; then
        sleep "$LAUNCH_STAGGER_SECONDS"
      fi
    done

    sleep 20

    for slot in "${!gpu_ids[@]}"; do
      pid="${slot_pids[$slot]:-}"
      if [[ -z "$pid" ]]; then
        continue
      fi
      if kill -0 "$pid" 2>/dev/null; then
        continue
      fi
      if wait "$pid"; then
        echo "finished ${slot_tasks[$slot]} pid=${pid}"
      else
        status=$?
        echo "failed ${slot_tasks[$slot]} pid=${pid} status=${status}" >&2
        for other_pid in "${slot_pids[@]:-}"; do
          if [[ -n "${other_pid:-}" ]] && kill -0 "$other_pid" 2>/dev/null; then
            kill -TERM "$other_pid" 2>/dev/null || true
          fi
        done
        exit "$status"
      fi
      unset 'slot_pids[$slot]'
      unset 'slot_tasks[$slot]'
      active_jobs=$((active_jobs - 1))
    done
  done
  exit 0
fi

for model_id in $MODELS; do
  for split in $SPLITS; do
    run_one "$model_id" "$split"
  done
done
