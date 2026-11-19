#!/usr/bin/env bash
set -u

# Usage:
#   bash tools/run_12h_train_and_interpretability.sh [config_name] [exp_name] [max_images]
# Example:
#   bash tools/run_12h_train_and_interpretability.sh encoder_only_multi_dataset loopcut3r_encoder_multi_12h 64

CONFIG_NAME="${1:-encoder_only_multi_dataset}"
EXP_NAME="${2:-loopcut3r_encoder_multi_12h}"
MAX_IMAGES="${3:-64}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SRC_DIR" || exit 1

# Helps with CUDA allocator fragmentation on low-VRAM GPUs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:64}"

echo "[INFO] Starting training for up to 12h"
set +e
timeout --signal=SIGINT --kill-after=300 12h \
  python train.py --config-name "$CONFIG_NAME" exp_name="$EXP_NAME" resume=null
TRAIN_EXIT=$?
set -e

echo "[INFO] Training command exit code: $TRAIN_EXIT"

BASE_DIR="$SRC_DIR/checkpoints/$EXP_NAME"
CKPT_DIR="$BASE_DIR/checkpoints"
BEST_CKPT="$CKPT_DIR/checkpoint-best.pth"
LAST_CKPT="$CKPT_DIR/checkpoint-last.pth"
FINAL_CKPT="$CKPT_DIR/checkpoint-final.pth"

if [[ -f "$BEST_CKPT" ]]; then
  CKPT_PATH="$BEST_CKPT"
elif [[ -f "$LAST_CKPT" ]]; then
  CKPT_PATH="$LAST_CKPT"
elif [[ -f "$FINAL_CKPT" ]]; then
  CKPT_PATH="$FINAL_CKPT"
else
  echo "[ERROR] No checkpoint found in $CKPT_DIR"
  echo "[HINT] The >700M model still did not fit on this GPU."
  echo "[HINT] Retry on CPU with:"
  echo "       python train.py --config-name $CONFIG_NAME exp_name=$EXP_NAME resume=null force_cpu=True mixed_precision=no amp=0"
  exit 1
fi

echo "[INFO] Running interpretability from checkpoint: $CKPT_PATH"
python tools/layer_similarity_encoder.py \
  --config-dir ../config \
  --config-name "$CONFIG_NAME" \
  --split test \
  --checkpoint "$CKPT_PATH" \
  --output-dir "$BASE_DIR/interpretability/layer_similarity_test" \
  --max-images "$MAX_IMAGES" \
  --batch-size 1 \
  --num-workers 0 \
  --device cpu

echo "[INFO] Done. Results in: $BASE_DIR/interpretability/layer_similarity_test"
