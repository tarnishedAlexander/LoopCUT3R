# Corre las 5 celdas del benchmark (relpose scannet/tum/sintel + video_depth
# sintel/bonn) para UNA configuracion de inferencia de stage3b.
#
# El eval construye el modelo desde `ck["args"].model`, asi que la configuracion
# se graba ahi y los scripts de eval no se tocan. Editar el bloque de python de
# abajo para cambiar num_steps / enc_rope / prime_state.
#
# Resultados: tools/ab/relpose_table.py para la pose, y el Abs Rel que imprime
# eval/video_depth/eval_depth.py para la profundidad.
# Ver docs/attractor_mechanism.md seccion 5d.
set -uo pipefail
cd /app/recurrentTransformer/LoopCUT3R
export CUT3R_EVAL_NUM_PROCESSES=1
# LA medida que falta: 6 pasos SOLOS, sin rope y sin priming. Pesos de stage3b
# tal cual (layout sin rope), para que la UNICA diferencia con stage3b_rerun sea
# el recuento de aplicaciones del encoder.
python - <<'PY'
import re, torch
sd = torch.load("checkpoints/recurrent_stage3b/checkpoints/checkpoint-final.pth",
                map_location="cpu", weights_only=False)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
base = torch.load("checkpoints/recurrent_stage3b/checkpoints/checkpoint-best.pth",
                  map_location="cpu", weights_only=False)
s = base["args"].model
def set_kv(s, k, v):
    pat = re.compile(r"(?<![A-Za-z_])" + re.escape(k) + r"=[^,)]+")
    h = pat.findall(s)
    if len(h) == 1: return pat.sub(f"{k}={v}", s)
    assert not h and s.endswith("))")
    return s[:-2] + f", {k}={v}))"
for k, v in (("num_steps", "6"), ("enc_rope", "False"), ("prime_state", "False")):
    s = set_kv(s, k, v)
base["args"].model = s; base["model"] = sd
for k in ("optimizer", "scaler"): base.pop(k, None)
torch.save(base, ".scratch/solo6.pth")
print("  6 pasos solos:", s[-70:], flush=True)
PY
for data in scannet tum sintel; do
  out="$PWD/eval_results/relpose/${data}_ABL_solo6"
  echo "### RELPOSE $data solo6"
  CUT3R_EVAL_WEIGHTS=.scratch/solo6.pth accelerate launch --num_processes 1 \
    --main_process_port 29563 eval/relpose/launch.py --weights .scratch/solo6.pth \
    --output_dir "$out" --eval_dataset "$data" --size 512 --full_seq 2>&1 | tail -2
  echo "### RELPOSE $data solo6 HECHO"
done
for data in sintel bonn; do
  out="$PWD/eval_results/video_depth/${data}_ABL_solo6"
  echo "### DEPTH $data solo6"
  CUT3R_EVAL_WEIGHTS=.scratch/solo6.pth accelerate launch --num_processes 1 \
    --main_process_port 29521 eval/video_depth/launch.py --weights .scratch/solo6.pth \
    --output_dir "$out" --eval_dataset "$data" --size 512 2>&1 | tail -3
  python eval/video_depth/eval_depth.py --output_dir "$out" --eval_dataset "$data" \
    --align "scale&shift" 2>&1 | grep -oE "\{.*\}" | tail -1
  echo "### DEPTH $data solo6 HECHO"
  df -h /home 2>/dev/null | tail -1 || true
done
rm -f .scratch/solo6.pth
echo "=== LISTO ==="
