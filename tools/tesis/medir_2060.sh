#!/usr/bin/env bash
# Mide VRAM pico y FPS de CUT3R y Loop-CUT3R en la GPU de 6 GB (RTX 2060).
# Cada modelo corre con su propio codigo y la carga de pesos se verifica.
#
# Hace falta en la maquina de la 2060:
#   - los dos repositorios (CUT3R y LoopCUT3R) y su entorno de Python con PyTorch + CUDA
#   - cut3r_512_dpt_4_64.pth                        (3,0 GB)
#   - loopcut3r_fase3_final.pth                     (1,4 GB; esta en
#       dataset/disk14tb/eval_tesis/para_2060/ de la maquina de entrenamiento)
#
# Uso (ajustar las cuatro rutas):
#   CUT3R_REPO=~/CUT3R LOOP_REPO=~/LoopCUT3R \
#   CUT3R_CKPT=~/pesos/cut3r_512_dpt_4_64.pth LOOP_CKPT=~/pesos/loopcut3r_fase3_final.pth \
#   bash tools/tesis/medir_2060.sh
#
# Deja resultados_2060/*.json y resultados_2060/*.txt. Tarda unos minutos.
# Si una corrida no cabe en memoria se anota como OOM y sigue con la siguiente.
set -uo pipefail
: "${CUT3R_REPO:?falta CUT3R_REPO}"; : "${LOOP_REPO:?falta LOOP_REPO}"
: "${CUT3R_CKPT:?falta CUT3R_CKPT}"; : "${LOOP_CKPT:?falta LOOP_CKPT}"
CUT3R_REPO=$(realpath "$CUT3R_REPO"); LOOP_REPO=$(realpath "$LOOP_REPO")
CUT3R_CKPT=$(realpath "$CUT3R_CKPT"); LOOP_CKPT=$(realpath "$LOOP_CKPT")
OUT=${OUT:-$PWD/resultados_2060}; mkdir -p "$OUT"
IMGS="$LOOP_REPO/examples/001"
EF="$LOOP_REPO/tools/eval/efficiency.py"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"

for size in 512 224; do
  FR="13 90 300"; [ "$size" = 224 ] && FR="13 90"
  python "$EF" --repo "$CUT3R_REPO" --label cut3r --model_path "$CUT3R_CKPT" --img_dir "$IMGS" \
      --size $size --frames $FR --out "$OUT/cut3r_$size.json" 2>&1 | grep -E '^\[|ABORT|Error' | tee "$OUT/cut3r_$size.txt"
  python "$EF" --repo "$LOOP_REPO"  --label loop  --model_path "$LOOP_CKPT"  --img_dir "$IMGS" \
      --size $size --frames $FR --out "$OUT/loop_$size.json"  2>&1 | grep -E '^\[|ABORT|Error' | tee "$OUT/loop_$size.txt"
done
echo "=== listo: enviar la carpeta $OUT ==="
