set -uo pipefail
cd /app/recurrentTransformer/LoopCUT3R
export CUT3R_EVAL_NUM_PROCESSES=1
BIG=/app/recurrentTransformer/dataset/disk14tb/eval_tesis/video_depth
corre () {
  LABEL="$1"; CK="$2"
  for data in sintel bonn; do
    out="$BIG/${data}_${LABEL}"; rm -rf "$out"; mkdir -p "$out"
    CUT3R_EVAL_WEIGHTS="$CK" accelerate launch --num_processes 1 --main_process_port 29521 \
      eval/video_depth/launch.py --weights "$CK" --output_dir "$out" --eval_dataset "$data" --size 224 2>&1 | tail -1
    echo "RES $data $LABEL $(python eval/video_depth/eval_depth.py --output_dir "$out" --eval_dataset "$data" --align "scale&shift" 2>&1 | grep -oE "\{.*\}" | tail -1)"
  done
}
corre LoopCUT3R_F1_stage1r3_ep12_224 checkpoints/recurrent_stage1_r3/checkpoints/checkpoint-last.pth
corre LoopCUT3R_F2_stage2_final_224  checkpoints/recurrent_stage2/checkpoints/checkpoint-final.pth
echo "=== LISTO ==="
