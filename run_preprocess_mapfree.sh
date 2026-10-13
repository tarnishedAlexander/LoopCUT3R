#!/bin/bash
# -------------------------------------------------------------------------
# Helper script to preprocess the Map-free dataset.
# Run this script inside the docker container under /app/recurrentTransformer/LoopCUT3R/
# -------------------------------------------------------------------------

# Exit on error
set -e

# Set python path
cd /app/recurrentTransformer/LoopCUT3R
export PYTHONPATH=.:src

echo "=========================================================="
# Stage 1: Preprocess the colmap results (Undistortion)
# Note: It runs the original preprocess_mapfree.py completely as-is.
echo "Starting Map-free Preprocessing (Stage 1: Image Undistortion)"
echo "=========================================================="
python datasets_preprocess/preprocess_mapfree.py \
    --mapfree_dir /app/recurrentTransformer/dataset/disk12tb/Mapfreedataset/train \
    --colmap_dir /app/recurrentTransformer/dataset/disk12tb/Mapfreedataset/research_map-free-reloc_dataset_colmap/colmap \
    --output_dir /app/recurrentTransformer/dataset/disk12tb/Mapfreedataset/first_output

echo "=========================================================="
# Stage 2: Re-organize the data structure
# Note: It runs the original preprocess_mapfree2.py completely as-is.
echo "Starting Map-free Preprocessing (Stage 2: Re-organizing structure)"
echo "=========================================================="
python datasets_preprocess/preprocess_mapfree2.py \
    --mapfree_dir /app/recurrentTransformer/dataset/disk12tb/Mapfreedataset/research_map-free-reloc_dataset_colmap/colmap \
    --output_dir /app/recurrentTransformer/dataset/disk12tb/Mapfreedataset/processed_mapfree

echo "=========================================================="
echo "Map-free Dataset Preprocessing Complete!"
echo "=========================================================="
