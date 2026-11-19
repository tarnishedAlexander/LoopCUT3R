#!/bin/bash
# eval_compare.sh
# Run relpose + video_depth benchmarks for two checkpoints and print a side-by-side table.
#
# Usage:
#   bash eval_compare.sh                                    # defaults (see below)
#   bash eval_compare.sh <our_ckpt> <cut3r_ckpt>           # explicit paths
#   bash eval_compare.sh <our_ckpt> <cut3r_ckpt> 224 512   # explicit paths + sizes
#   bash eval_compare.sh --score-only                      # re-score existing predictions
#
# Results land in eval_results/ and a summary is printed at the end.
#
# NOTE: video_depth eval requires size=512. If OURS_SIZE=224 (stage-1/2 linear model),
# video_depth is skipped for our checkpoint and only run for CUT3R.
#
# --score-only skips all inference and re-scores whatever video_depth
# predictions are already on disk under eval_results/video_depth/. Use it when
# a full run reported "(no results)" for depth but the predictions did land.

set -euo pipefail

cd "$(dirname "$0")"

SCORE_ONLY=false
if [[ "${1:-}" == "--score-only" ]]; then
    SCORE_ONLY=true
    shift
fi

OURS="${1:-checkpoints/recurrent_stage3/checkpoints/checkpoint-last.pth}"
CUT3R="${2:-../CUT3R/checkpoint/cut3r_512_dpt_4_64.pth}"
OURS_SIZE="${3:-512}"
CUT3R_SIZE="${4:-512}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="eval_results/comparison_${TIMESTAMP}.log"

export CUT3R_EVAL_NUM_PROCESSES=1

# ── derive short labels from checkpoint paths ─────────────────────────────────
# e.g. checkpoints/recurrent_stage3/checkpoints/checkpoint-8.pth
#   → stage dir = recurrent_stage3, ckpt = checkpoint-8
#   → label = LoopCUT3R_recurrent_stage3_checkpoint-8
_ours_stage="$(basename "$(dirname "$(dirname "$OURS")")")"
_ours_ckpt="$(basename "$OURS" .pth)"
OURS_LABEL="LoopCUT3R_${_ours_stage}_${_ours_ckpt}"

_cut3r_ckpt="$(basename "$CUT3R" .pth)"
CUT3R_LABEL="CUT3R_${_cut3r_ckpt}"

# ── helpers ──────────────────────────────────────────────────────────────────
die()  { echo "[ERROR] $*" >&2; exit 1; }
sep()  { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }
hdr()  { sep; echo "  $*"; sep; }

check_ckpt() {
    [[ -f "$1" ]] || die "Checkpoint not found: $1"
}

# ── validation ───────────────────────────────────────────────────────────────
# --score-only touches no checkpoints, so don't require them to exist.
if [[ "$SCORE_ONLY" != true ]]; then
    check_ckpt "$OURS"
    check_ckpt "$CUT3R"
fi
mkdir -p eval_results

echo "============================================================"
echo "  LoopCUT3R vs CUT3R Benchmark Comparison"
echo "  $(date)"
echo "  OUR CKPT : $OURS  (label: $OURS_LABEL)"
echo "  CUT3R    : $CUT3R  (label: $CUT3R_LABEL)"
echo "============================================================"

# ── run one model through all evals ─────────────────────────────────────────
run_all_evals() {
    local ckpt="$1"
    local label="$2"
    local size="$3"      # image size (224 or 512)
    local use_ga="${4:-false}"  # true → add --opt_global (CUT3R's proper mode)

    local ga_flag=""
    [[ "$use_ga" == "true" ]] && ga_flag="--opt_global"

    hdr "Evaluating: $label  (size=${size}${ga_flag:+, global-align})"
    export CUT3R_EVAL_WEIGHTS="$ckpt"

    echo ""
    echo "→ relpose (ScanNet / TUM / Sintel)"
    local safe_label="${label//[^a-zA-Z0-9_]/_}"
    (
        source eval/_common.sh
        for data in scannet tum sintel; do
            out="${PWD}/eval_results/relpose/${data}_${safe_label}"
            echo "   $data → $out"
            CUT3R_EVAL_WEIGHTS="$ckpt" \
            CUT3R_EVAL_NUM_PROCESSES=1 \
            accelerate launch \
                --num_processes 1 \
                --main_process_port 29558 \
                eval/relpose/launch.py \
                --weights "$ckpt" \
                --output_dir "$out" \
                --eval_dataset "$data" \
                --size "$size" \
                --full_seq \
                $ga_flag 2>&1 | tail -5
        done
    )

    echo ""
    if [[ "$size" != "512" ]]; then
        echo "→ video_depth: SKIPPED (requires size=512, this model is ${size}px)"
    else
        echo "→ video_depth (Sintel / Bonn)"
        (
            source eval/_common.sh
            for data in sintel bonn; do
                out="${PWD}/eval_results/video_depth/${data}_${safe_label}"
                echo "   $data → $out"
                set +e
                CUT3R_EVAL_WEIGHTS="$ckpt" \
                CUT3R_EVAL_NUM_PROCESSES=1 \
                accelerate launch \
                    --num_processes 1 \
                    --main_process_port 29517 \
                    eval/video_depth/launch.py \
                    --weights "$ckpt" \
                    --output_dir "$out" \
                    --eval_dataset "$data" \
                    --size 512 2>&1 | tail -10
                local inference_exit=$?
                set -e
                if [[ $inference_exit -ne 0 ]]; then
                    echo "   [WARN] video_depth $data inference failed (exit $inference_exit)"
                else
                    echo "   [scoring] $data depth metrics..."
                    python eval/video_depth/eval_depth.py \
                        --output_dir "$out" \
                        --eval_dataset "$data" \
                        --align "scale&shift" 2>&1 | tail -3
                fi
            done
        )
    fi
}

# ── re-score predictions already on disk, no inference ───────────────────────
# Folded in from the former eval_score_depth.sh. That script hardcoded four
# stage2_ep7 paths; this discovers whatever is actually present instead.
score_existing_depth() {
    hdr "RE-SCORING EXISTING video_depth PREDICTIONS"
    local root="eval_results/video_depth"
    [[ -d "$root" ]] || die "$root not found — nothing to score"

    local found=0
    for out in "$root"/*/; do
        [[ -d "$out" ]] || continue
        out="${out%/}"
        # dir names are <dataset>_<label>, so the dataset is the leading field
        local dataset="${out##*/}"
        dataset="${dataset%%_*}"
        echo "→ ${dataset}  (${out})"
        python eval/video_depth/eval_depth.py \
            --output_dir "$out" \
            --eval_dataset "$dataset" \
            --align "scale&shift" 2>&1 | tail -3
        found=$((found + 1))
    done
    [[ $found -gt 0 ]] || die "no prediction directories under $root"
    echo ""
    echo "Re-scored ${found} prediction directory/ies."
}

# ── run both models ──────────────────────────────────────────────────────────
if [[ "$SCORE_ONLY" == true ]]; then
    score_existing_depth
else
    run_all_evals "$OURS"  "$OURS_LABEL"  "$OURS_SIZE" "false"
    run_all_evals "$CUT3R" "$CUT3R_LABEL" "$CUT3R_SIZE" "true"
fi

# ── summary table ────────────────────────────────────────────────────────────
sep
echo ""
echo "  RESULTS SUMMARY"
echo ""

print_relpose_table() {
    local label="$1"
    local safe="${label//[^a-zA-Z0-9_]/_}"
    echo "  [$label]"
    for data in scannet tum sintel; do
        local result_dir="eval_results/relpose/${data}_${safe}"
        local metric_files
        mapfile -t metric_files < <(find "$result_dir" -name "*_eval_metric.txt" 2>/dev/null | sort)
        local n=${#metric_files[@]}
        if [[ $n -eq 0 ]]; then
            printf "    %-12s  (no results)\n" "$data"
            continue
        fi
        local sum_ape=0 sum_rpe_rot=0 sum_rpe_trans=0 count=0
        for f in "${metric_files[@]}"; do
            local ape rpe_r rpe_t
            ape="$(   awk '/APE w.r.t. translation/{found=1} found && /mean/{print $2; exit}' "$f")"
            rpe_r="$( awk '/RPE w.r.t. rotation/{found=1}    found && /mean/{print $2; exit}' "$f")"
            rpe_t="$( awk '/RPE w.r.t. translation/{found=1} found && /mean/{print $2; exit}' "$f")"
            if [[ -n "$ape" && -n "$rpe_r" && -n "$rpe_t" ]]; then
                sum_ape=$(python3 -c "print($sum_ape + $ape)")
                sum_rpe_rot=$(python3 -c "print($sum_rpe_rot + $rpe_r)")
                sum_rpe_trans=$(python3 -c "print($sum_rpe_trans + $rpe_t)")
                count=$((count + 1))
            fi
        done
        if [[ $count -gt 0 ]]; then
            local avg_ape avg_rot avg_trans
            avg_ape=$(  python3 -c "print(f'{$sum_ape/$count:.4f}')")
            avg_rot=$(  python3 -c "print(f'{$sum_rpe_rot/$count:.2f}')")
            avg_trans=$(python3 -c "print(f'{$sum_rpe_trans/$count:.4f}')")
            printf "    %-12s  APE=%-8s  RPE_rot=%-8s  RPE_t=%-8s  [%d scenes]\n" \
                "$data" "$avg_ape" "$avg_rot" "$avg_trans" "$count"
        else
            printf "    %-12s  (parse error)\n" "$data"
        fi
    done
    echo ""
}

print_depth_table() {
    local label="$1"
    local safe="${label//[^a-zA-Z0-9_]/_}"
    echo "  [$label]"
    for data in sintel bonn; do
        # prefer scale&shift result (most comparable across models)
        local resfile
        resfile="$(find "eval_results/video_depth/${data}_${safe}" \
            -name "result_scale*shift.json" 2>/dev/null | head -1)"
        if [[ -z "$resfile" ]]; then
            resfile="$(find "eval_results/video_depth/${data}_${safe}" \
                -name "*.json" 2>/dev/null | head -1)"
        fi
        if [[ -n "$resfile" && -f "$resfile" ]]; then
            local abs_rel rmse delta1
            # keys are "Abs Rel", "RMSE", "δ < 1.25"
            abs_rel="$(python3 -c "import json; d=json.load(open('$resfile')); print(f\"{d.get('Abs Rel', d.get('AbsRel', 'N/A')):.4f}\")" 2>/dev/null || echo 'N/A')"
            rmse="$(   python3 -c "import json; d=json.load(open('$resfile')); print(f\"{d.get('RMSE',   d.get('SiRMSE', 'N/A')):.4f}\")" 2>/dev/null || echo 'N/A')"
            delta1="$( python3 -c "import json; d=json.load(open('$resfile')); print(f\"{d.get('δ < 1.25', d.get('δ < 1.25', 'N/A')):.4f}\")" 2>/dev/null || echo 'N/A')"
            printf "    %-12s  AbsRel=%-8s  RMSE=%-8s  δ<1.25=%s\n" \
                "$data" "$abs_rel" "$rmse" "$delta1"
        else
            printf "    %-12s  (no results)\n" "$data"
        fi
    done
    echo ""
}

echo "  ── Camera Pose Estimation (relpose) ──"
print_relpose_table "$OURS_LABEL"
print_relpose_table "$CUT3R_LABEL"

echo "  ── Video Depth (scale+shift aligned) ──"
print_depth_table "$OURS_LABEL"
print_depth_table "$CUT3R_LABEL"

sep
echo ""
echo "Full logs: eval_results/"
