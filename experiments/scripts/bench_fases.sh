# Celdas que faltan para la tabla de progresion por fase del capitulo 5.
#   F1  stage1_r3 checkpoint-last (epoca 12 de 15: el final quedo en el M.2 retirado)
#   F2  stage2 checkpoint-final   (epoca 14)
# Mismo protocolo que el resto: --size 512 --full_seq, scale&shift.
# La profundidad se escribe en el disco de 14 TB y se enlaza, porque /home esta lleno.
set -uo pipefail
cd /app/recurrentTransformer/LoopCUT3R
export CUT3R_EVAL_NUM_PROCESSES=1
BIG=/app/recurrentTransformer/dataset/disk14tb/eval_tesis/video_depth
corre () {
  LABEL="$1"; CK="$2"
  for data in scannet tum sintel; do
    out="$PWD/eval_results/relpose/${data}_${LABEL}"
    CUT3R_EVAL_WEIGHTS="$CK" accelerate launch --num_processes 1 --main_process_port 29563 \
      eval/relpose/launch.py --weights "$CK" --output_dir "$out" --eval_dataset "$data" --size 512 --full_seq 2>&1 | tail -1
    echo "### RELPOSE $data $LABEL HECHO"
  done
  for data in sintel bonn; do
    mkdir -p "$BIG/${data}_${LABEL}"
    ln -sfn "$BIG/${data}_${LABEL}" "$PWD/eval_results/video_depth/${data}_${LABEL}"
    out="$PWD/eval_results/video_depth/${data}_${LABEL}"
    CUT3R_EVAL_WEIGHTS="$CK" accelerate launch --num_processes 1 --main_process_port 29521 \
      eval/video_depth/launch.py --weights "$CK" --output_dir "$out" --eval_dataset "$data" --size 512 2>&1 | tail -1
    python eval/video_depth/eval_depth.py --output_dir "$out" --eval_dataset "$data" --align "scale&shift" 2>&1 | grep -oE "\{.*\}" | tail -1
    echo "### DEPTH $data $LABEL HECHO"
  done
}
corre LoopCUT3R_F1_stage1r3_ep12 checkpoints/recurrent_stage1_r3/checkpoints/checkpoint-last.pth
corre LoopCUT3R_F2_stage2_final  checkpoints/recurrent_stage2/checkpoints/checkpoint-final.pth
echo "=== LISTO ==="
