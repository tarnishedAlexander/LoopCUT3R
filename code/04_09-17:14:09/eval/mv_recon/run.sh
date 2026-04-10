#!/bin/bash

set -e

workdir='.'
model_name='ours'
source "$(dirname "$0")/../_common.sh"

model_weights="$(cut3r_resolve_eval_weights "$workdir")"

ckpt_name="$(basename "$model_weights" .pth)"
cut3r_print_eval_checkpoint "$model_weights"

output_dir="${workdir}/eval_results/mv_recon/${model_name}_${ckpt_name}"
echo "$output_dir"
cut3r_accelerate_launch 8 29501 eval/mv_recon/launch.py \
    --weights "$model_weights" \
    --output_dir "$output_dir" \
    --model_name "$model_name"