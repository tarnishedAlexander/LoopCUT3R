#!/bin/bash

set -e

workdir='.'
model_name='ours'
source "$(dirname "$0")/../_common.sh"

model_weights="$(cut3r_resolve_eval_weights "$workdir")"

ckpt_name="$(basename "$model_weights" .pth)"
cut3r_print_eval_checkpoint "$model_weights"
datasets=('sintel' 'bonn' 'kitti')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/video_depth/${data}_${model_name}"
    echo "$output_dir"
    cut3r_accelerate_launch 4 29517 eval/video_depth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data" \
        --size 512
    python eval/video_depth/eval_depth.py \
    --output_dir "$output_dir" \
    --eval_dataset "$data" \
    --align "scale"
done
