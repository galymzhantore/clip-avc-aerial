#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-era}"

BASE_OUT_DIR="${BASE_OUT_DIR:-checkpoints_encoder_sweep}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs_encoder_sweep}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"
COMMON_ENV=(
  EPOCHS="$EPOCHS"
  BATCH_SIZE="$BATCH_SIZE"
  MICRO_BATCH_SIZE="$MICRO_BATCH_SIZE"
  EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE"
  WORKERS="$WORKERS"
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
  echo "===== encoder sweep: $name ====="
  env \
    "${COMMON_ENV[@]}" \
    OUT_DIR="$BASE_OUT_DIR/$name" \
    LOG_DIR="$BASE_LOG_DIR/$name" \
    "$@" \
    ./scripts/run_gpu_repro.sh "$MODE"
}

# One-factor changes from the current best reproduction setting:
#   baseline reference: swin3d_b + OpenAI CLIP ViT-B/32 visual/text towers.
run_case video_r2plus1d18 VIDEO_MODEL=r2plus1d_18
run_case video_mvitv2s VIDEO_MODEL=mvit_v2_s
run_case clip_vitb16 CLIP_MODEL=ViT-B/16
run_case text_bert TEXT_ENCODER=bert MAX_TEXT_TOKENS=77
