#!/bin/bash
# Run the relpose and video_depth benchmarks for CUT3R and LoopCUT3R, each model
# inside its OWN source tree.
#
# This is the whole point of the script.  eval_compare.sh runs both checkpoints
# from the LoopCUT3R tree, and add_ckpt_path.py:add_path_to_dust3r() ignores the
# checkpoint it is handed -- it always force-inserts the local src/.  A CUT3R
# checkpoint loaded that way gets built into LoopCUT3R's ARCroco3DStereo, whose
# encoder defaults to num_blocks=1 and uses different parameter names, so all 24
# CUT3R encoder blocks are discarded and the encoder runs at random init.
# load_model() only *prints* the load report, so nothing fails.  See
# docs/benchmark_audit.md.
#
# Verify any load first:
#   python tools/ab/probe_load.py <repo> <ckpt>     # wants 0 missing / 0 unexpected
#
# Usage:
#   bash tools/ab/run_baseline_eval.sh                # both models, both tasks
#   TASKS=relpose bash tools/ab/run_baseline_eval.sh  # relpose only
#
# Requires evo 1.30.6 for relpose.  Newer evo does NOT work: 1.38 dropped the
# est_name argument from ape()/rpe() and rewrote sync.matching_time_indices into
# a form that raises "object too deep for desired array" on these trajectories.
# Install without touching numpy (torch is built against 1.26.4):
#   pip install --no-deps evo==1.30.6 seaborn numexpr natsort argcomplete \
#                         rosbags apsw lz4
set -u

ROOT=${ROOT:-/app/recurrentTransformer}
LOOP=${LOOP:-$ROOT/LoopCUT3R}
CUT3R=${CUT3R:-$ROOT/CUT3R}
RES=${RES:-$LOOP/eval_results}
TASKS=${TASKS:-relpose video_depth}
SHIMS=$LOOP/tools/ab/shims

LOOP_CKPT=${LOOP_CKPT:-checkpoints/recurrent_stage3b/checkpoints/checkpoint-last.pth}
CUT3R_CKPT=${CUT3R_CKPT:-checkpoint/cut3r_512_dpt_4_64.pth}
LOOP_LABEL=${LOOP_LABEL:-LoopCUT3R_stage3b}
CUT3R_LABEL=${CUT3R_LABEL:-CUT3R_fixed}

# checkpoint-final.pth is a bare state_dict with no embedded args and the eval
# launchers have no --model_cfg, so they need a training checkpoint.  For
# stage3b, checkpoint-last.pth holds weights bitwise identical to
# checkpoint-final.pth, so nothing is lost by using it.

relpose () {   # repo  ckpt  label
  cd "$1" || return 1
  for data in scannet tum sintel; do
    out="$RES/relpose/${data}_$3"
    echo "--- relpose $3 / $data  $(date +%H:%M:%S)"
    python "$SHIMS/run_relpose.py" eval/relpose/launch.py \
      --weights "$2" --output_dir "$out" \
      --eval_dataset "$data" --size 512 --full_seq \
      > "/tmp/rp_$3_$data.log" 2>&1
    echo "    exit=$?  scored: $(find "$out" -name '*_eval_metric.txt' 2>/dev/null | wc -l)"
    grep -ciE "error in sequence" "/tmp/rp_$3_$data.log" | sed 's/^/    sequence errors: /'
  done
}

video_depth () {   # repo  ckpt  label
  cd "$1" || return 1
  for data in sintel bonn; do
    out="$RES/video_depth/${data}_$3"
    echo "--- video_depth $3 / $data  $(date +%H:%M:%S)"
    python "$SHIMS/run_video_depth.py" eval/video_depth/launch.py \
      --weights "$2" --output_dir "$out" --eval_dataset "$data" --size 512 \
      > "/tmp/vd_$3_$data.log" 2>&1
    echo "    exit=$?"
    python eval/video_depth/eval_depth.py --output_dir "$out" \
      --eval_dataset "$data" --align "scale&shift" 2>&1 | tail -2
  done
}

for task in $TASKS; do
  echo "######## $task : CUT3R (own tree)"
  $task "$CUT3R" "$CUT3R_CKPT" "$CUT3R_LABEL"
  echo "######## $task : LoopCUT3R (own tree)"
  $task "$LOOP" "$LOOP_CKPT" "$LOOP_LABEL"
done

echo "######## done $(date +%H:%M:%S)"
echo "table:  python tools/ab/benchmark_table.py"
