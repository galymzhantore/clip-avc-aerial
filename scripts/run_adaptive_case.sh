#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-era}"
CASE_NAME="${CASE_NAME:-adaptive_case}"
BASE_OUT_DIR="${BASE_OUT_DIR:-checkpoints_adaptive_encoder_search}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs_adaptive_encoder_search}"
SUMMARY="$BASE_LOG_DIR/summary.tsv"

BASE_BATCH_SIZE="${BATCH_SIZE:-16}"
BASE_MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
BASE_EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
BASE_WORKERS="${WORKERS:-8}"
BASE_CONTEXT_LOGIT_CHUNK_SIZE="${CONTEXT_LOGIT_CHUNK_SIZE:-128}"
EPOCHS_FULL="${EPOCHS:-50}"
PROBE_EPOCHS="${PROBE_EPOCHS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-2}"
ALLOW_BATCH_SIZE_ADAPT="${ALLOW_BATCH_SIZE_ADAPT:-0}"

mkdir -p "$BASE_LOG_DIR"
if [[ ! -f "$SUMMARY" ]]; then
  printf "case\tstatus\tstarted_at\tended_at\tprobe_max_mem_mib\tprobe_avg_gpu_util\tbatch_size\tmicro_batch_size\teval_batch_size\tworkers\tenv\n" > "$SUMMARY"
fi

gpu_monitor() {
  local out="$1"
  local stop_file="$2"
  printf "timestamp,memory_used_mib,memory_total_mib,gpu_util_pct,memory_util_pct\n" > "$out"
  while [[ ! -f "$stop_file" ]]; do
    printf "%s," "$(date -Iseconds)" >> "$out"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits | head -n 1 >> "$out"
    sleep "$SAMPLE_INTERVAL"
  done
}

run_with_gpu_log() {
  local gpu_log="$1"
  shift
  local stop_file="$gpu_log.stop"
  rm -f "$stop_file"
  gpu_monitor "$gpu_log" "$stop_file" &
  local monitor_pid=$!
  "$@"
  local status=$?
  touch "$stop_file"
  wait "$monitor_pid" 2>/dev/null || true
  rm -f "$stop_file"
  return "$status"
}

max_gpu_mem() {
  awk -F, 'NR > 1 {gsub(/ /, "", $2); if ($2 + 0 > max) max = $2 + 0} END {print max + 0}' "$1"
}

avg_gpu_util() {
  awk -F, 'NR > 1 {gsub(/ /, "", $4); sum += $4 + 0; n += 1} END {if (n) printf "%.1f", sum / n; else print 0}' "$1"
}

choose_workers() {
  local avg_util="${1:-0}"
  local ncpu
  ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
  if awk "BEGIN {exit !($avg_util < 70)}" && (( ncpu >= 24 )); then
    echo 16
  elif (( ncpu >= 24 )); then
    echo 12
  else
    echo "$BASE_WORKERS"
  fi
}

choose_settings() {
  local max_mem="$1"
  local total_mem="$2"
  local batch="$BASE_BATCH_SIZE"
  local micro="$BASE_MICRO_BATCH_SIZE"
  local eval_batch="$BASE_EVAL_BATCH_SIZE"
  local context_chunk="$BASE_CONTEXT_LOGIT_CHUNK_SIZE"

  if (( max_mem < total_mem / 3 )); then
    micro=16
    eval_batch=16
    context_chunk=0
  elif (( max_mem < total_mem / 2 )); then
    micro=8
    eval_batch=8
    context_chunk=256
  fi

  if [[ "$ALLOW_BATCH_SIZE_ADAPT" == "1" && "$micro" -ge 16 && "$batch" -lt 32 ]]; then
    batch=32
  fi

  printf "%s %s %s %s\n" "$batch" "$micro" "$eval_batch" "$context_chunk"
}

case_log_dir="$BASE_LOG_DIR/$CASE_NAME"
case_out_dir="$BASE_OUT_DIR/$CASE_NAME"
probe_log_dir="$case_log_dir/probe"
probe_out_dir="$case_out_dir/probe"
full_log_dir="$case_log_dir/full"
full_out_dir="$case_out_dir/full"
mkdir -p "$probe_log_dir" "$probe_out_dir" "$full_log_dir" "$full_out_dir"

started=$(date -Iseconds)
env_line=$(printf "%q " "$@")
{
  echo "case=$CASE_NAME"
  echo "started_at=$started"
  echo "mode=$MODE"
  echo "extra_env=$env_line"
  echo "allow_batch_size_adapt=$ALLOW_BATCH_SIZE_ADAPT"
} > "$case_log_dir/manifest.txt"

echo "===== adaptive probe: $CASE_NAME ====="
run_with_gpu_log "$probe_log_dir/gpu_usage.csv" \
  env \
    "$@" \
    EPOCHS="$PROBE_EPOCHS" \
    BATCH_SIZE="$BASE_BATCH_SIZE" \
    MICRO_BATCH_SIZE="$BASE_MICRO_BATCH_SIZE" \
    EVAL_BATCH_SIZE="$BASE_EVAL_BATCH_SIZE" \
    WORKERS="$BASE_WORKERS" \
    CONTEXT_LOGIT_CHUNK_SIZE="$BASE_CONTEXT_LOGIT_CHUNK_SIZE" \
    SAVE_EVERY=0 \
    KEEP_CHECKPOINTS=1 \
    OUT_DIR="$probe_out_dir" \
    LOG_DIR="$probe_log_dir" \
    ./scripts/run_gpu_repro.sh "train-${MODE}"
probe_status=$?
probe_max_mem=$(max_gpu_mem "$probe_log_dir/gpu_usage.csv")
probe_avg_util=$(avg_gpu_util "$probe_log_dir/gpu_usage.csv")
total_mem=$(awk -F, 'NR == 2 {gsub(/ /, "", $3); print $3 + 0}' "$probe_log_dir/gpu_usage.csv")

{
  echo "probe_status=$probe_status"
  echo "probe_max_mem_mib=$probe_max_mem"
  echo "probe_total_mem_mib=$total_mem"
  echo "probe_avg_gpu_util=$probe_avg_util"
} >> "$case_log_dir/manifest.txt"

if [[ "$probe_status" -ne 0 ]]; then
  ended=$(date -Iseconds)
  echo "ended_at=$ended" >> "$case_log_dir/manifest.txt"
  echo "status=$probe_status" >> "$case_log_dir/manifest.txt"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$CASE_NAME" "$probe_status" "$started" "$ended" "$probe_max_mem" "$probe_avg_util" \
    "$BASE_BATCH_SIZE" "$BASE_MICRO_BATCH_SIZE" "$BASE_EVAL_BATCH_SIZE" "$BASE_WORKERS" "$env_line" >> "$SUMMARY"
  exit "$probe_status"
fi

read -r chosen_batch chosen_micro chosen_eval chosen_context <<< "$(choose_settings "$probe_max_mem" "$total_mem")"
chosen_workers=$(choose_workers "$probe_avg_util")

{
  echo "chosen_batch_size=$chosen_batch"
  echo "chosen_micro_batch_size=$chosen_micro"
  echo "chosen_eval_batch_size=$chosen_eval"
  echo "chosen_workers=$chosen_workers"
  echo "chosen_context_logit_chunk_size=$chosen_context"
} >> "$case_log_dir/manifest.txt"

echo "===== adaptive full run: $CASE_NAME ====="
run_with_gpu_log "$full_log_dir/gpu_usage.csv" \
  env \
    "$@" \
    EPOCHS="$EPOCHS_FULL" \
    BATCH_SIZE="$chosen_batch" \
    MICRO_BATCH_SIZE="$chosen_micro" \
    EVAL_BATCH_SIZE="$chosen_eval" \
    WORKERS="$chosen_workers" \
    SAVE_EVERY=0 \
    KEEP_CHECKPOINTS=1 \
    CONTEXT_LOGIT_CHUNK_SIZE="$chosen_context" \
    OUT_DIR="$full_out_dir" \
    LOG_DIR="$full_log_dir" \
    ./scripts/run_gpu_repro.sh "$MODE"
status=$?
ended=$(date -Iseconds)
{
  echo "ended_at=$ended"
  echo "status=$status"
} >> "$case_log_dir/manifest.txt"

printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$CASE_NAME" "$status" "$started" "$ended" "$probe_max_mem" "$probe_avg_util" \
  "$chosen_batch" "$chosen_micro" "$chosen_eval" "$chosen_workers" "$env_line" >> "$SUMMARY"
exit "$status"
