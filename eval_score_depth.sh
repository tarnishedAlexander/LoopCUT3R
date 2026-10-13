#!/bin/bash
# Score video_depth predictions that are already on disk (inference already ran).
# Run this after eval_compare.sh if depth metrics show "(no results)".

set -euo pipefail
cd "$(dirname "$0")"

score() {
    local out="$1" dataset="$2"
    if [[ -d "$out" ]]; then
        echo "→ $dataset  ($out)"
        python eval/video_depth/eval_depth.py \
            --output_dir "$out" \
            --eval_dataset "$dataset" \
            --align "scale&shift"
    else
        echo "   [SKIP] $out not found"
    fi
}

score eval_results/video_depth/sintel_LoopCUT3R_stage2_ep7  sintel
score eval_results/video_depth/bonn_LoopCUT3R_stage2_ep7    bonn
score eval_results/video_depth/sintel_CUT3R_released         sintel
score eval_results/video_depth/bonn_CUT3R_released           bonn
