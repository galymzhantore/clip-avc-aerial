#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"

REPO_DIR="${REPO_DIR:-$HOME/Documents/clip-avc-aerial}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
ERA_ROOT="${ERA_ROOT:-$HOME/datasets/era_clipavc}"
MOD20_ROOT="${MOD20_ROOT:-$HOME/datasets/mod20_clipavc}"
OUT_DIR="${OUT_DIR:-checkpoints}"
LOG_DIR="${LOG_DIR:-logs}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
WORKERS="${WORKERS:-8}"
LR="${LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.2}"
CLIP_FRAMES="${CLIP_FRAMES:-8}"
SWIN_FRAMES="${SWIN_FRAMES:-16}"
RESIZE_SIZE="${RESIZE_SIZE:-256}"
TEXT_ENCODER="${TEXT_ENCODER:-clip}"
MAX_TEXT_TOKENS="${MAX_TEXT_TOKENS:-77}"
CHECKPOINT_VIDEO_SWIN="${CHECKPOINT_VIDEO_SWIN:-1}"
SAVE_EVERY="${SAVE_EVERY:-10}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"
CONTRASTIVE_TARGETS="${CONTRASTIVE_TARGETS:-instance}"
CLASSIFIER_HEAD="${CLASSIFIER_HEAD:-1}"
CE_WEIGHT="${CE_WEIGHT:-1.0}"
CLASSIFIER_FEATURE="${CLASSIFIER_FEATURE:-coarse}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$REPO_DIR"
mkdir -p "$LOG_DIR" "$OUT_DIR"

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv not found at $UV_BIN. Set UV_BIN=/path/to/uv or install uv." >&2
  exit 1
fi

common_train_args=(
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --micro-batch-size "$MICRO_BATCH_SIZE"
  --workers "$WORKERS"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --clip-frames "$CLIP_FRAMES"
  --swin-frames "$SWIN_FRAMES"
  --resize-size "$RESIZE_SIZE"
  --text-encoder "$TEXT_ENCODER"
  --max-text-tokens "$MAX_TEXT_TOKENS"
  --scheduler step
  --lr-step-epochs 15
  --lr-gamma 0.1
  --amp
  --out "$OUT_DIR"
  --save-every "$SAVE_EVERY"
  --keep-checkpoints "$KEEP_CHECKPOINTS"
  --contrastive-targets "$CONTRASTIVE_TARGETS"
  --ce-weight "$CE_WEIGHT"
  --classifier-feature "$CLASSIFIER_FEATURE"
)

if [[ "$CLASSIFIER_HEAD" == "1" ]]; then
  common_train_args+=(--classifier-head)
else
  common_train_args+=(--no-classifier-head)
fi

common_eval_args=(
  --split test
  --batch-size "$EVAL_BATCH_SIZE"
  --workers "$WORKERS"
  --clip-frames "$CLIP_FRAMES"
  --swin-frames "$SWIN_FRAMES"
  --resize-size "$RESIZE_SIZE"
  --amp
)

shape_test() {
  "$UV_BIN" run python -m scripts.shape_test \
    --batch 1 \
    --clip-frames "$CLIP_FRAMES" \
    --swin-frames "$SWIN_FRAMES" \
    --text-encoder "$TEXT_ENCODER" \
    --max-text-tokens "$MAX_TEXT_TOKENS" \
    2>&1 | tee "$LOG_DIR/shape_test.log"
}

train_era() {
  local extra_args=()
  if [[ "$CHECKPOINT_VIDEO_SWIN" == "1" ]]; then
    extra_args+=(--checkpoint-video-swin)
  fi

  "$UV_BIN" run python -m scripts.train \
    --dataset era \
    --data-root "$ERA_ROOT" \
    "${common_train_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_DIR/train_era_full.log"
}

eval_era() {
  "$UV_BIN" run python -m scripts.eval \
    --dataset era \
    --data-root "$ERA_ROOT" \
    --checkpoint "$OUT_DIR/era/era_epoch$(printf '%03d' "$EPOCHS").pt" \
    "${common_eval_args[@]}" \
    2>&1 | tee "$LOG_DIR/eval_era_full.log"
}

train_mod20() {
  local extra_args=()
  if [[ "$CHECKPOINT_VIDEO_SWIN" == "1" ]]; then
    extra_args+=(--checkpoint-video-swin)
  fi

  "$UV_BIN" run python -m scripts.train \
    --dataset mod20 \
    --data-root "$MOD20_ROOT" \
    "${common_train_args[@]}" \
    "${extra_args[@]}" \
    2>&1 | tee "$LOG_DIR/train_mod20_full.log"
}

eval_mod20() {
  "$UV_BIN" run python -m scripts.eval \
    --dataset mod20 \
    --data-root "$MOD20_ROOT" \
    --checkpoint "$OUT_DIR/mod20/mod20_epoch$(printf '%03d' "$EPOCHS").pt" \
    "${common_eval_args[@]}" \
    2>&1 | tee "$LOG_DIR/eval_mod20_full.log"
}

case "$MODE" in
  shape)
    shape_test
    ;;
  era)
    shape_test
    train_era
    eval_era
    ;;
  mod20)
    shape_test
    train_mod20
    eval_mod20
    ;;
  train-era)
    train_era
    ;;
  eval-era)
    eval_era
    ;;
  train-mod20)
    train_mod20
    ;;
  eval-mod20)
    eval_mod20
    ;;
  all)
    shape_test
    train_era
    eval_era
    train_mod20
    eval_mod20
    ;;
  *)
    cat >&2 <<EOF
Usage: $0 {shape|era|mod20|train-era|eval-era|train-mod20|eval-mod20|all}

Optional env overrides:
  REPO_DIR=$REPO_DIR
  ERA_ROOT=$ERA_ROOT
  MOD20_ROOT=$MOD20_ROOT
  BATCH_SIZE=$BATCH_SIZE
  MICRO_BATCH_SIZE=$MICRO_BATCH_SIZE
  EVAL_BATCH_SIZE=$EVAL_BATCH_SIZE
  WORKERS=$WORKERS
  RESIZE_SIZE=$RESIZE_SIZE
  TEXT_ENCODER=$TEXT_ENCODER
  MAX_TEXT_TOKENS=$MAX_TEXT_TOKENS
  CHECKPOINT_VIDEO_SWIN=$CHECKPOINT_VIDEO_SWIN
  SAVE_EVERY=$SAVE_EVERY
  KEEP_CHECKPOINTS=$KEEP_CHECKPOINTS
  CONTRASTIVE_TARGETS=$CONTRASTIVE_TARGETS
  CLASSIFIER_HEAD=$CLASSIFIER_HEAD
  CE_WEIGHT=$CE_WEIGHT
  CLASSIFIER_FEATURE=$CLASSIFIER_FEATURE
EOF
    exit 2
    ;;
esac
