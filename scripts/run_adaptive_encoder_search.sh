#!/usr/bin/env bash
set -uo pipefail

MODE="${1:-era}"

BASE_OUT_DIR="${BASE_OUT_DIR:-checkpoints_adaptive_encoder_search}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs_adaptive_encoder_search}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"

COMMON_ENV=(
  FREEZE_CLIP_VIT="${FREEZE_CLIP_VIT:-1}"
  FREEZE_VIDEO_SWIN_BACKBONE="${FREEZE_VIDEO_SWIN_BACKBONE:-1}"
  CLASSIFIER_MODE="${CLASSIFIER_MODE:-context}"
  REFINED_TEXT_POOLING="${REFINED_TEXT_POOLING:-eos}"
  CONTRASTIVE_TARGETS="${CONTRASTIVE_TARGETS:-instance}"
  CONTEXT_LOGIT_CHUNK_SIZE="${CONTEXT_LOGIT_CHUNK_SIZE:-128}"
  CLASSIFIER_FEATURE="${CLASSIFIER_FEATURE:-refined}"
)

run_case() {
  local name="$1"
  shift

  echo "===== adaptive encoder search: $name ====="
  CASE_NAME="$name" \
  BASE_OUT_DIR="$BASE_OUT_DIR" \
  BASE_LOG_DIR="$BASE_LOG_DIR" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE" \
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  WORKERS="$WORKERS" \
  ./scripts/run_adaptive_case.sh "$MODE" "${COMMON_ENV[@]}" "$@"
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

"$HOME/.local/bin/uv" run python -m scripts.collect_experiment_results \
  logs_encoder_sweep \
  logs_overnight_encoder_search \
  "$BASE_LOG_DIR" \
  --out "$BASE_LOG_DIR/combined_results.csv"
