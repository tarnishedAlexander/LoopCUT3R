# Celdas de profundidad del punto de partida de stage3f (rope encendido, sin
# entrenar). Existian para relpose (ABL_rope_solo) pero no para video_depth.
set -uo pipefail
cd /app/recurrentTransformer/LoopCUT3R
export CUT3R_EVAL_NUM_PROCESSES=1
python - <<'PY'
import torch
b = torch.load("checkpoints/recurrent_stage3f_encrope_real/checkpoints/checkpoint-best.pth", map_location="cpu", weights_only=False)
sd = torch.load("checkpoints/loop3b_encrope_init.pth", map_location="cpu", weights_only=False)
assert set(sd) == set(b["model"]), "layout distinto"
b["model"] = sd
for k in ("optimizer", "scaler"): b.pop(k, None)
torch.save(b, ".scratch/rope_solo.pth")
print("ok", b["args"].model[-40:], flush=True)
PY
for data in sintel bonn; do
  out="$PWD/eval_results/video_depth/${data}_ABL_rope_solo"
  CUT3R_EVAL_WEIGHTS=.scratch/rope_solo.pth accelerate launch --num_processes 1 \
    --main_process_port 29521 eval/video_depth/launch.py --weights .scratch/rope_solo.pth \
    --output_dir "$out" --eval_dataset "$data" --size 512 2>&1 | tail -1
  echo "### DEPTH $data HECHO"
done
rm -f .scratch/rope_solo.pth
echo "=== LISTO ==="
