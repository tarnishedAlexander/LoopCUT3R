#!/bin/bash
set -e

workdir='.'
model_name='ours'
source "$(dirname "$0")/../_common.sh"

model_weights="$(cut3r_resolve_eval_weights "$workdir")"

ckpt_name="$(basename "$model_weights" .pth)"
cut3r_print_eval_checkpoint "$model_weights"
datasets=('sintel' 'bonn' 'kitti' 'nyu')

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    echo "$output_dir"
    python eval/monodepth/launch.py \
        --weights "$model_weights" \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

for data in "${datasets[@]}"; do
    output_dir="${workdir}/eval_results/monodepth/${data}_${model_name}"
    python eval/monodepth/eval_metrics.py \
        --output_dir "$output_dir" \
        --eval_dataset "$data"
done

