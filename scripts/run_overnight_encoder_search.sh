#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-era}"

BASE_OUT_DIR="${BASE_OUT_DIR:-checkpoints_overnight_encoder_search}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs_overnight_encoder_search}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"
SUMMARY="$BASE_LOG_DIR/summary.tsv"

COMMON_ENV=(
  EPOCHS="$EPOCHS"
  BATCH_SIZE="$BATCH_SIZE"
  MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE"
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE"
  WORKERS="$WORKERS"
  SAVE_EVERY=0
  KEEP_CHECKPOINTS=1
  FREEZE_CLIP_VIT="${FREEZE_CLIP_VIT:-1}"
  FREEZE_VIDEO_SWIN_BACKBONE="${FREEZE_VIDEO_SWIN_BACKBONE:-1}"
  CLASSIFIER_MODE="${CLASSIFIER_MODE:-context}"
  REFINED_TEXT_POOLING="${REFINED_TEXT_POOLING:-eos}"
  CONTRASTIVE_TARGETS="${CONTRASTIVE_TARGETS:-instance}"
  CONTEXT_LOGIT_CHUNK_SIZE="${CONTEXT_LOGIT_CHUNK_SIZE:-128}"
  CLASSIFIER_FEATURE="${CLASSIFIER_FEATURE:-refined}"
)

mkdir -p "$BASE_LOG_DIR"
if [[ ! -f "$SUMMARY" ]]; then
  printf "case\tstatus\tstarted_at\tended_at\tenv\n" > "$SUMMARY"
fi

run_case() {
  local name="$1"
  shift
  local case_log_dir="$BASE_LOG_DIR/$name"
  local case_out_dir="$BASE_OUT_DIR/$name"
  local started ended status env_line

  mkdir -p "$case_log_dir" "$case_out_dir"
  env_line=$(printf "%q " "${COMMON_ENV[@]}" "$@")
  started=$(date -Iseconds)
  {
    echo "case=$name"
    echo "started_at=$started"
    echo "mode=$MODE"
    echo "env=$env_line"
  } > "$case_log_dir/manifest.txt"

  echo "===== overnight encoder search: $name ====="
  env \
    "${COMMON_ENV[@]}" \
    OUT_DIR="$case_out_dir" \
    LOG_DIR="$case_log_dir" \
    "$@" \
    ./scripts/run_gpu_repro.sh "$MODE"
  status=$?
  ended=$(date -Iseconds)
  {
    echo "ended_at=$ended"
    echo "status=$status"
  } >> "$case_log_dir/manifest.txt"
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$status" "$started" "$ended" "$env_line" >> "$SUMMARY"
}

# Extra one-factor model swaps not covered by run_encoder_sweep.sh.
run_case video_swin3d_t VIDEO_MODEL=swin3d_t
run_case video_swin3d_s VIDEO_MODEL=swin3d_s
run_case video_r3d18 VIDEO_MODEL=r3d_18
run_case video_mc3_18 VIDEO_MODEL=mc3_18
run_case video_s3d VIDEO_MODEL=s3d

# Text-pooling and mixed encoder variants after the one-factor checks.
run_case text_bert_mean TEXT_ENCODER=bert MAX_TEXT_TOKENS=77 REFINED_TEXT_POOLING=mean
run_case clip_vitb16_textbert CLIP_MODEL=ViT-B/16 TEXT_ENCODER=bert MAX_TEXT_TOKENS=77
