#!/usr/bin/env python3
"""Lista los checkpoints que NO tienen su .npz, y como sondearlos.

Por que existe: un .npz pesa 50 MB y un checkpoint 4,3 GB. Mientras el .npz
exista, se puede recalcular cualquier metrica y rehacer cualquier figura sin el
checkpoint. Borrar un checkpoint SIN sondearlo primero si pierde informacion —
paso con la epoca 1 de stage3e, borrada y nunca medida.

    nunca borrar un checkpoint que no se haya sondeado

    python tools/ab/cubrir_npz.py            # solo lista
"""
import glob, os, re, torch

RECON = ".scratch/ab/recon"
# etiqueta ya sondeada -> de que checkpoint salio
CONOCIDOS = {
    "checkpoints/recurrent_stage3b/checkpoints/checkpoint-final.pth": "loop3b",
    "checkpoints/recurrent_stage3d_encrope/checkpoints/checkpoint-1.pth": "rope_ep0",
    "checkpoints/recurrent_stage3d_encrope/checkpoints/checkpoint-2.pth": "rope_ep1",
    "checkpoints/recurrent_stage3e_consistency/checkpoints/checkpoint-1.pth": "cons_ep0",
    "checkpoints/recurrent_stage3e_consistency/checkpoints/checkpoint-3.pth": "cons_ep2",
}

def info(p):
    try:
        ck = torch.load(p, map_location="meta" if False else "cpu", weights_only=False,
                        mmap=True)
    except Exception:
        return None, None
    if isinstance(ck, dict) and "args" in ck:
        s = ck["args"].model
        m = re.search(r"img_size=\((\d+)", s)
        return "args", int(m.group(1)) if m else None
    return "desnudo", None

filas = []
for p in sorted(glob.glob("checkpoints/recurrent_*/checkpoints/*.pth")):
    gb = os.path.getsize(p) / 1e9
    tipo, res = info(p)
    etq = CONOCIDOS.get(p)
    filas.append((p, gb, tipo, res, etq))

print(f"  {'checkpoint':<58} {'GB':>5} {'args':>8} {'res':>5}  estado")
falta = []
for p, gb, tipo, res, etq in filas:
    if etq:
        est = f"sondeado -> {etq}"
    elif res and res < 512:
        est = "224px: la sonda corre a 512, no comparable"
    else:
        est = "SIN SONDEAR"
        falta.append((p, tipo))
    print(f"  {p:<58} {gb:>5.1f} {tipo or '?':>8} {str(res or '-'):>5}  {est}")

print(f"\n  sin sondear: {len(falta)}")
for p, tipo in falta:
    cfg = "" if tipo == "args" else "  (state dict desnudo: necesita --model_cfg)"
    print(f"    {p}{cfg}")
