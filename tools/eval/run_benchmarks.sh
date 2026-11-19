#!/usr/bin/env bash
# Camera pose and video depth benchmarks for one checkpoint, with the protocol
# used for the reported results:
#   relpose      ScanNet, TUM-dynamics, Sintel   (--size 512 --full_seq)
#   video_depth  Sintel, Bonn                    (--size 512, scale&shift alignment)
#
# Usage, from the repository root:
#   bash tools/eval/run_benchmarks.sh LABEL CHECKPOINT [REPO]
#
#   LABEL       name for the results, e.g. LoopCUT3R or CUT3R
#   CHECKPOINT  a training checkpoint (it carries the model string in ck["args"])
#   REPO        source tree that builds the model (default: this repository)
#
# A checkpoint must be run by the source tree it was trained with.  To evaluate
# CUT3R, pass a CUT3R checkout as REPO; loading CUT3R weights from this tree
# would build LoopCUT3R's encoder and silently drop CUT3R's 24 encoder blocks.
# Check a pairing first with:  python tools/eval/check_weights.py REPO CHECKPOINT
#
# Results go to eval_results/{relpose,video_depth}/<dataset>_<LABEL>.  Then:
#   python tools/eval/pose_table.py CUT3R LoopCUT3R
#
# Environment: SIZE (default 512), TASKS (default "relpose video_depth").
# relpose needs evo 1.30.6:  pip install --no-deps evo==1.30.6
set -uo pipefail

LABEL=${1:?usage: run_benchmarks.sh LABEL CHECKPOINT [REPO]}
CKPT=$(realpath "${2:?usage: run_benchmarks.sh LABEL CHECKPOINT [REPO]}")
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
REPO=$(realpath "${3:-$HERE}")
SIZE=${SIZE:-512}
TASKS=${TASKS:-relpose video_depth}
RES=$HERE/eval_results
SHIMS=$HERE/tools/eval/shims
LOGS=$RES/logs
mkdir -p "$LOGS"
export CUT3R_EVAL_NUM_PROCESSES=1

relpose () {
  for data in scannet tum sintel; do
    out="$RES/relpose/${data}_${LABEL}"
    echo "--- relpose $LABEL / $data"
    (cd "$REPO" && python "$SHIMS/run_relpose.py" eval/relpose/launch.py \
        --weights "$CKPT" --output_dir "$out" --eval_dataset "$data" \
        --size "$SIZE" --full_seq) > "$LOGS/relpose_${data}_${LABEL}.log" 2>&1
    echo "    exit=$?  scored: $(find "$out" -name '*_eval_metric.txt' 2>/dev/null | wc -l)"
  done
}

video_depth () {
  for data in sintel bonn; do
    out="$RES/video_depth/${data}_${LABEL}"
    echo "--- video_depth $LABEL / $data"
    (cd "$REPO" && python "$SHIMS/run_video_depth.py" eval/video_depth/launch.py \
        --weights "$CKPT" --output_dir "$out" --eval_dataset "$data" \
        --size "$SIZE") > "$LOGS/video_depth_${data}_${LABEL}.log" 2>&1
    echo "    exit=$?"
    (cd "$HERE" && python eval/video_depth/eval_depth.py --output_dir "$out" \
        --eval_dataset "$data" --align "scale&shift" 2>&1 | tail -2)
  done
}

for task in $TASKS; do
  $task
done
