#!/usr/bin/env bash
set -uo pipefail

# Reproduces the four CLIP-AVC pretraining rows from the paper table:
#   ViT / IN, ViT+Swin-B / IN-K400, ViT / WIT, ViT+Swin-B / WIT-K400.
#
# Run on the GPU PC from the repository root, preferably inside tmux:
#   ./scripts/run_paper_pretraining_matrix.sh all

MODE="${1:-all}"
BASE_OUT_DIR="${BASE_OUT_DIR:-checkpoints_paper_pretraining_matrix}"
BASE_LOG_DIR="${BASE_LOG_DIR:-logs_paper_pretraining_matrix}"
SUMMARY="$BASE_LOG_DIR/summary.tsv"

COMMON_ENV=(
  EPOCHS="${EPOCHS:-50}"
  BATCH_SIZE="${BATCH_SIZE:-16}"
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
  WORKERS="${WORKERS:-8}"
  SAVE_EVERY=0
  KEEP_CHECKPOINTS=1
  FREEZE_CLIP_VIT=1
  FREEZE_VIDEO_SWIN_BACKBONE="${FREEZE_VIDEO_SWIN_BACKBONE:-0}"
  VIDEO_MODEL=swin3d_b
  CHECKPOINT_VIDEO_SWIN=1
  CLASSIFIER_HEAD=1
  CLASSIFIER_MODE=context
  CLASSIFIER_FEATURE=refined
  REFINED_TEXT_POOLING=eos
  CONTRASTIVE_TARGETS=instance
  CONTEXT_LOGIT_CHUNK_SIZE="${CONTEXT_LOGIT_CHUNK_SIZE:-128}"
)

mkdir -p "$BASE_LOG_DIR"
if [[ ! -f "$SUMMARY" ]]; then
  printf "case\tstatus\tstarted_at\tended_at\ttarget_era\ttarget_mod20\tenv\n" > "$SUMMARY"
fi

run_case() {
  local name="$1"
  local target_era="$2"
  local target_mod20="$3"
  shift 3

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
    echo "target_era=$target_era"
    echo "target_mod20=$target_mod20"
    echo "env=$env_line"
  } > "$case_log_dir/manifest.txt"

  echo "===== paper pretraining matrix: $name ====="
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
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$name" "$status" "$started" "$ended" "$target_era" "$target_mod20" "$env_line" >> "$SUMMARY"
}

run_case paper_vit_in 68.40 92.70 \
  VISUAL_PRETRAINING=imagenet \
  IMAGENET_VIT_MODEL=vit_b_32 \
  TEXT_ENCODER=clip \
  CLIP_MODEL=ViT-B/32 \
  CROSS_TRANSFORMER=0 \
  CHECKPOINT_VIDEO_SWIN=0

run_case paper_vit_swinb_in_k400 70.29 96.50 \
  VISUAL_PRETRAINING=imagenet \
  IMAGENET_VIT_MODEL=vit_b_32 \
  TEXT_ENCODER=clip \
  CLIP_MODEL=ViT-B/32 \
  CROSS_TRANSFORMER=1

run_case paper_vit_wit 80.91 97.20 \
  VISUAL_PRETRAINING=wit \
  CLIP_MODEL=ViT-B/32 \
  TEXT_ENCODER=clip \
  CROSS_TRANSFORMER=0 \
  CHECKPOINT_VIDEO_SWIN=0

run_case paper_vit_swinb_wit_k400 84.87 98.93 \
  VISUAL_PRETRAINING=wit \
  CLIP_MODEL=ViT-B/32 \
  TEXT_ENCODER=clip \
  CROSS_TRANSFORMER=1

"${UV_BIN:-$HOME/.local/bin/uv}" run python -m scripts.collect_experiment_results \
  "$BASE_LOG_DIR" \
  --out "$BASE_LOG_DIR/combined_results.csv"
