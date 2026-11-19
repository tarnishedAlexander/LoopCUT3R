#!/usr/bin/env python3
"""Control del arreglo de posicion del encoder (docs/attractor_mechanism.md 1b).

Tres comprobaciones, en este orden, porque la tercera no significa nada si las
dos primeras no pasan:

  1. REMAPEO EXACTO.  El modelo con `enc_rope=True` y los pesos remapeados,
     pero con el rope anulado a mano, tiene que dar la MISMA salida bit a bit
     que el modelo original.  Si falla, el renombrado in_proj<->qkv no es
     equivalente y cualquier medida posterior esta contaminada.
  2. LA POSICION YA IMPORTA.  Con el rope puesto, mentir en xpos tiene que
     cambiar la salida.  Antes daba exactamente 0.
  3. MAGNITUD DEL CHOQUE.  Cuanto cambia la salida del encoder al encender el
     rope sobre pesos que nunca lo vieron.  Esto es lo que dice si la
     re-adaptacion es barata o cara.

    python tools/ab/check_enc_rope.py --ckpt CKPT --ckpt_rope CKPT_REMAPEADO \\
        --cfg_file config_model_string.txt
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/app/recurrentTransformer/LoopCUT3R")
    ap.add_argument("--ckpt", required=True, help="checkpoint original (sin rope)")
    ap.add_argument("--ckpt_rope", required=True,
                    help="el mismo, pasado por tools/ab/remap_enc_attn.py --to rope")
    ap.add_argument("--cfg_file", required=True,
                    help="fichero con el model string del YAML de entrenamiento")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.ckpt)
    from src.dust3r.model import ARCroco3DStereo

    cfg = open(args.cfg_file).read().strip()
    if "enc_rope" in cfg:
        raise SystemExit("el model string ya trae enc_rope: pasa el original")
    cfg_rope = cfg.replace("))", ", enc_rope=True))")

    print("cargando el original ...", flush=True)
    a = ARCroco3DStereo.from_pretrained(args.ckpt, model_cfg_str=cfg).to(args.device).eval()
    print("cargando el remapeado con enc_rope=True ...", flush=True)
    b = ARCroco3DStereo.from_pretrained(args.ckpt_rope, model_cfg_str=cfg_rope).to(args.device).eval()

    print(f"\n  original  : attn = {type(a.enc_blocks[0].attn).__name__}"
          f"   usa_pos = {getattr(a.enc_blocks[0], 'attn_uses_pos', False)}")
    print(f"  remapeado : attn = {type(b.enc_blocks[0].attn).__name__}"
          f"   usa_pos = {getattr(b.enc_blocks[0], 'attn_uses_pos', False)}")
    if not getattr(b.enc_blocks[0], "attn_uses_pos", False):
        raise SystemExit("el modelo remapeado NO quedo con rope: enc_rope no llego")

    torch.manual_seed(0)
    img = torch.randn(1, 3, 224, 224, device=args.device)

    def encode(m, pos_override=None, rope_off=False):
        guardado = []
        if rope_off:
            for blk in m.enc_blocks:
                guardado.append(blk.attn.rope)
                blk.attn.rope = None
        with torch.no_grad():
            x, p = m.patch_embed(img)
            if pos_override is not None:
                p = pos_override
            y = x
            for _ in range(m.num_steps):
                for blk in m.enc_blocks:
                    y = blk(y, p)
            y = m.enc_norm(y)
        if rope_off:
            for blk, r in zip(m.enc_blocks, guardado):
                blk.attn.rope = r
        return y, p

    base, pos = encode(a)
    sin_rope, _ = encode(b, rope_off=True)
    con_rope, _ = encode(b)
    N = base.shape[1]
    perm = torch.randperm(N)
    pos_falsa = pos[:, perm]
    con_rope_falsa, _ = encode(b, pos_override=pos_falsa)

    d1 = (sin_rope - base).abs().max().item()
    d2 = (con_rope_falsa - con_rope).abs().max().item()
    rel = ((con_rope - base).norm() / base.norm()).item()

    print(f"\n  tokens N = {N}")
    print("=" * 70)
    print(f"  1) remapeado con rope anulado vs original : max|dif| = {d1:.3e}")
    print(f"       -> {'EXACTO' if d1 < 1e-5 else 'NO EQUIVALENTE, parar aqui'}")
    print(f"  2) rope puesto, posiciones mentirosas     : max|dif| = {d2:.3e}")
    print(f"       -> {'la posicion YA importa' if d2 > 0 else 'sigue ciego'}")
    print(f"  3) choque de encender el rope             : ||dif||/||base|| = {rel:.4f}"
          f"  ({100*rel:.1f}%)")
    print("=" * 70)
    ok = d1 < 1e-5 and d2 > 0
    print("\nVEREDICTO:", "control pasado, el arreglo esta bien cableado"
          if ok else "CONTROL FALLIDO, no seguir")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
