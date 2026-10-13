#!/bin/bash
set -e

# Exit on any error to ensure we don't start the next stage if the previous one fails
echo "===================================="
echo "    LoopCUT3R MVP Training Run      "
echo "===================================="

cd "$(dirname "$0")"

echo ""
echo "[1/4] Starting MVP Stage 1: Foundation Learning (2 views)..."
python src/train.py --config-name mvp_stage1
echo "[1/4] MVP Stage 1 completed successfully!"
echo "------------------------------------"

echo ""
echo "[2/4] Starting MVP Stage 2: BatchList Robustness (2 views)..."
python src/train.py --config-name mvp_stage2
echo "[2/4] MVP Stage 2 completed successfully!"
echo "------------------------------------"

echo ""
echo "[3/4] Starting MVP Stage 3: Context Scaling (4 views)..."
python src/train.py --config-name mvp_stage3
echo "[3/4] MVP Stage 3 completed successfully!"
echo "------------------------------------"

echo ""
echo "[4/4] Starting MVP Stage 4: Long Horizon & Persistence (16 views, frozen encoder)..."
python src/train.py --config-name mvp_stage4
echo "[4/4] MVP Stage 4 completed successfully!"
echo "===================================="
echo "      MVP Training Fully Finished   "
echo "===================================="
