#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/VLM-Post-Training || exit 1
runid=$(cat outputs/runtime/dpo_v2_ablation_5gpu/LATEST_RUN_ID 2>/dev/null || echo 20260812_5gpu_ablation_r3)
runroot="outputs/runtime/dpo_v2_ablation_5gpu/$runid"
log="$runroot/server_auto_shutdown_after_archive_active.log"
echo "auto_shutdown_watcher_start=$(date -Is) run_id=$runid" >> "$log"
while true; do
  if test -f "$runroot/FAILED"; then
    echo "remote_failed_no_shutdown=$(date -Is)" >> "$log"
    cat "$runroot/FAILED" >> "$log"
    exit 2
  fi
  if test -f "$runroot/READY_TO_ARCHIVE" && test -s "$runroot/archive_tar_path"; then
    archive=$(cat "$runroot/archive_tar_path")
    if test -s "$archive" && tar -tzf "$archive" >/dev/null 2>&1; then
      echo "archive_ready=$(date -Is) archive=$archive" >> "$log"
      sync
      sleep 10
      echo "shutdown_sent=$(date -Is)" >> "$log"
      shutdown -h now
      exit 0
    else
      echo "archive_not_ready=$(date -Is) archive=$archive" >> "$log"
    fi
  fi
  sleep 120
done
