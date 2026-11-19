# Las 5 celdas del benchmark para los dos pesos que faltan:
#   s3f      stage3f epoca 0, enc_rope=True entrenado (el experimento)
#   cons_ep2 stage3e epoca 2, el cabo suelto: mejor pose en la sonda,
#            descartado por la sonda ANTES de saber que la sonda no predice
#            el benchmark (seccion 5d). Nunca se midio aqui.
#
# Se juzga por esto, no por el loss ni por la sonda.
# A batir, stage3b sin tocar:
#   relpose  RPE_rot/ATE   scannet 54,807/0,8895  tum 42,693/0,7798  sintel 2,891/0,2368
#   AbsRel                 sintel 0,5808   bonn 0,1104
set -uo pipefail
cd /app/recurrentTransformer/LoopCUT3R
export CUT3R_EVAL_NUM_PROCESSES=1

prep () {  # $1 = checkpoint de origen (con args), $2 = destino
  python - "$1" "$2" <<'PY'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
assert "args" in ck and "model" in ck, f"{src} no trae args+model"
for k in ("optimizer", "scaler"):
    ck.pop(k, None)
torch.save(ck, dst)
m = ck["args"].model
print("  ", dst, "epoch", ck.get("epoch"), "|",
      [t.strip() for t in m.split(",") if "num_steps=" in t or "enc_rope" in t], flush=True)
PY
}

corre () {  # $1 = etiqueta, $2 = ruta del checkpoint de eval
  LABEL="$1"; CK="$2"
  for data in scannet tum sintel; do
    out="$PWD/eval_results/relpose/${data}_${LABEL}"
    echo "### RELPOSE $data $LABEL"
    CUT3R_EVAL_WEIGHTS="$CK" accelerate launch --num_processes 1 \
      --main_process_port 29563 eval/relpose/launch.py --weights "$CK" \
      --output_dir "$out" --eval_dataset "$data" --size 512 --full_seq 2>&1 | tail -2
    echo "### RELPOSE $data $LABEL HECHO"
  done
  for data in sintel bonn; do
    out="$PWD/eval_results/video_depth/${data}_${LABEL}"
    echo "### DEPTH $data $LABEL"
    CUT3R_EVAL_WEIGHTS="$CK" accelerate launch --num_processes 1 \
      --main_process_port 29521 eval/video_depth/launch.py --weights "$CK" \
      --output_dir "$out" --eval_dataset "$data" --size 512 2>&1 | tail -3
    python eval/video_depth/eval_depth.py --output_dir "$out" --eval_dataset "$data" \
      --align "scale&shift" 2>&1 | grep -oE "\{.*\}" | tail -1
    echo "### DEPTH $data $LABEL HECHO"
    df -h /home 2>/dev/null | tail -1 || true
  done
}

echo "=== preparando checkpoints de eval ==="
prep checkpoints/recurrent_stage3f_encrope_real/checkpoints/checkpoint-best.pth .scratch/s3f.pth
corre LoopCUT3R_stage3f .scratch/s3f.pth
rm -f .scratch/s3f.pth

prep checkpoints/recurrent_stage3e_consistency/checkpoints/checkpoint-3.pth .scratch/cons_ep2.pth
corre LoopCUT3R_cons_ep2 .scratch/cons_ep2.pth
rm -f .scratch/cons_ep2.pth

echo "=== LISTO ==="
