#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(cat outputs/runtime/final_holdout_v1/LATEST_RUN_ID)}"
RUN_ROOT="${RUN_ROOT:-outputs/runtime/final_holdout_v1/$RUN_ID}"
POLL_SECONDS="${POLL_SECONDS:-60}"
ARCHIVE_DIR="${ARCHIVE_DIR:-outputs/archives}"
CONSUMED_MARKER="${CONSUMED_MARKER:-data/mv_audit/final_holdout_v1/FINAL_HOLDOUT_CONSUMED}"
READY_MARKER="$RUN_ROOT/READY_TO_PULL"
FAILED_MARKER="$RUN_ROOT/FAILED"
WATCH_LOG="$RUN_ROOT/archive_watcher.log"

mkdir -p "$RUN_ROOT" "$ARCHIVE_DIR"

log() {
  printf "[%s] %s\n" "$(date -Is)" "$*" | tee -a "$WATCH_LOG"
}

running_final_holdout() {
  pgrep -af 'batch_inference|20_run_final_holdout|07_run_inference' \
    | grep -v '23_watch_final_holdout_archive_remote' \
    | grep -v grep \
    >/dev/null
}

archive_results() {
  local archive="$ARCHIVE_DIR/final_holdout_v1_${RUN_ID}.tar.gz"
  tar -czf "$archive" \
    data/mv_audit/final_holdout_v1 \
    docs/experiments/data_boundary \
    docs/experiments/final_holdout_v1 \
    outputs/predictions/final_holdout_v1 \
    outputs/eval_sets/final_holdout_v1 \
    outputs/eval_reports/final_holdout_v1 \
    "$RUN_ROOT" \
    configs/train/repair_sft_r3_final_holdout.yaml \
    configs/train/repair_sft_r3_final_holdout_server.yaml \
    configs/eval/audit_eval_frozen_v1.yaml \
    scripts/20_run_final_holdout.sh \
    scripts/21_launch_final_holdout_remote.sh \
    scripts/23_watch_final_holdout_archive_remote.sh \
    tools/summarize_final_holdout.py
  sha256sum "$archive" > "$archive.sha256"
  {
    printf "run_id=%s\n" "$RUN_ID"
    printf "archive=%s\n" "$archive"
    printf "archive_sha256=%s\n" "$(cut -d' ' -f1 "$archive.sha256")"
    printf "ready_at=%s\n" "$(date -Is)"
  } > "$READY_MARKER"
  log "ready_to_pull archive=$archive"
}

log "watch_start run_id=$RUN_ID"
while true; do
  if [[ -s "$CONSUMED_MARKER" ]]; then
    archive_results
    exit 0
  fi
  if ! running_final_holdout; then
    {
      printf "failed_at=%s\n" "$(date -Is)"
      printf "reason=final_holdout_process_stopped_without_consumed_marker\n"
    } > "$FAILED_MARKER"
    log "failed process stopped without consumed marker"
    exit 1
  fi
  log "running"
  sleep "$POLL_SECONDS"
done
