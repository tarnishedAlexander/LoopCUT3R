#!/usr/bin/env python3
"""Verifica que un checkpoint de swap carga COMPLETO: 0 claves que falten, 0 que sobren.

Hace falta porque `load_model` usa `load_state_dict(strict=False)`, que no falla
cuando el checkpoint esta mal formado: simplemente deja el modelo con los pesos
de inicializacion y la sonda mide ruido. Paso de verdad al preparar N2.

    python tools/ab/check_swap_load.py .scratch/ab/swap_EE.pth .scratch/cfg_swap_EE.txt
"""
import sys, torch
sys.path.insert(0, ".")
from add_ckpt_path import add_path_to_dust3r

ck_path, cfg_path = sys.argv[1], sys.argv[2]
add_path_to_dust3r(ck_path)
from src.dust3r.model import ARCroco3DStereo

cfg = open(cfg_path).read().strip()
model = ARCroco3DStereo.from_pretrained(ck_path, model_cfg_str=cfg)
sd_ck = torch.load(ck_path, map_location="cpu", weights_only=False)
sd_ck = sd_ck.get("model", sd_ck) if isinstance(sd_ck, dict) and "model" in sd_ck else sd_ck
sd_mo = model.state_dict()

falta = sorted(set(sd_mo) - set(sd_ck))
sobra = sorted(set(sd_ck) - set(sd_mo))
forma = [(k, tuple(sd_mo[k].shape), tuple(sd_ck[k].shape))
         for k in sorted(set(sd_mo) & set(sd_ck)) if sd_mo[k].shape != sd_ck[k].shape]

print(f"  modelo {len(sd_mo)} tensores, checkpoint {len(sd_ck)}")
print(f"  faltan {len(falta)}   sobran {len(sobra)}   formas distintas {len(forma)}")
for n, l in (("FALTAN", falta), ("SOBRAN", sobra)):
    if l:
        print(f"  {n}: {l[:8]}{' ...' if len(l) > 8 else ''}")
if forma:
    print(f"  FORMAS: {forma[:5]}")
ok = not falta and not sobra and not forma
print("VEREDICTO:", "carga completa" if ok else "CARGA INCOMPLETA, no medir con esto")
raise SystemExit(0 if ok else 1)
