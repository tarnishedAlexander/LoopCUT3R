#!/usr/bin/env bash
set -u

# Usage:
#   bash tools/run_48h_training.sh [config_name] [exp_name]
# Example:
#   bash tools/run_48h_training.sh raptor_full_finetune raptor_full_finetune_48h

CONFIG_NAME="${1:-raptor_full_finetune}"
EXP_NAME="${2:-raptor_full_finetune_48h}"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SRC_DIR" || exit 1

# Best practices for RTX 2060
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"

echo "[INFO] Starting 48-hour training session: $EXP_NAME"
echo "[INFO] Initializing from: $CONFIG_NAME"

set +e
# timeout 48h sends SIGINT 5 minutes before hard-killing, allowing a clean save.
timeout --signal=SIGINT --kill-after=300 48h \
  python3 train.py --config-name "$CONFIG_NAME" exp_name="$EXP_NAME" resume=null
TRAIN_EXIT=$?
set -e

echo "[INFO] Training session ended with code: $TRAIN_EXIT"
if [[ $TRAIN_EXIT -eq 124 ]]; then
  echo "[INFO] Training timed out correctly after 48h."
fi

echo "[INFO] Done. Results in: checkpoints/$EXP_NAME"
