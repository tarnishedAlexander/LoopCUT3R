#!/bin/bash
# -------------------------------------------------------------------------
# Helper script to preprocess the ScanNet++ dataset.
# Run this script inside the docker container under /app/recurrentTransformer/LoopCUT3R/
# -------------------------------------------------------------------------

# Exit on error
set -e

# Set python path to find dust3r and dataset utility modules
export PYTHONPATH=.:src

echo "=========================================================="
echo "Starting ScanNet++ Preprocessing (Stage 1: Render & Undistort)"
echo "=========================================================="
python datasets_preprocess/preprocess_scannetpp.py \
    --scannetpp_dir /app/recurrentTransformer/dataset/disk12tb/scannet/scannetpp \
    --precomputed_pairs /app/recurrentTransformer/dataset/disk12tb/scannet/scannetpp/scannetpp_pairs \
    --output_dir /app/recurrentTransformer/dataset/disk12tb/scannet/processed_scannetpp \
    --pyopengl-platform egl

echo "=========================================================="
echo "Generating ScanNet++ Splits and Metadata (Stage 2)"
echo "=========================================================="
python datasets_preprocess/generate_set_scannetpp.py \
    --root /app/recurrentTransformer/dataset/disk12tb/scannet/processed_scannetpp \
    --max_interval 150 \
    --num_workers 8

echo "=========================================================="
echo "ScanNet++ Dataset Preprocessing Complete!"
echo "=========================================================="
