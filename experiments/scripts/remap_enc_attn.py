#!/usr/bin/env python3
"""Convierte los pesos de atencion del encoder recurrente entre los dos layouts.

`RecurrentBlock` sin rope usa `nn.MultiheadAttention`; con rope usa la misma
`Attention` que los bloques de CUT3R.  Los dos guardan la proyeccion QKV
fusionada con las mismas formas, asi que el cambio es un RENOMBRADO exacto:

    attn.in_proj_weight   <->  attn.qkv.weight     (3*dim, dim)
    attn.in_proj_bias     <->  attn.qkv.bias       (3*dim,)
    attn.out_proj.weight  <->  attn.proj.weight    (dim, dim)
    attn.out_proj.bias    <->  attn.proj.bias      (dim,)

Por que es exacto y no una aproximacion: las dos parten el eje de salida de
3*dim en tres trozos contiguos q, k, v y dentro de cada trozo parten dim en
num_heads cabezas contiguas; las dos escalan por head_dim**-0.5.  El control de
`tools/ab/check_enc_rope.py` lo verifica bit a bit apagando el rope.

    python tools/ab/remap_enc_attn.py ENTRADA.pth SALIDA.pth --to rope
"""
import argparse
import torch

PARES = [
    ("attn.in_proj_weight",  "attn.qkv.weight"),
    ("attn.in_proj_bias",    "attn.qkv.bias"),
    ("attn.out_proj.weight", "attn.proj.weight"),
    ("attn.out_proj.bias",   "attn.proj.bias"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("salida")
    ap.add_argument("--to", choices=["rope", "norope"], default="rope",
                    help="rope: in_proj/out_proj -> qkv/proj.  norope: al contrario.")
    ap.add_argument("--prefix", default="enc_blocks",
                    help="solo se renombran las claves bajo este prefijo, para no "
                         "tocar el decoder ni enc_blocks_ray_map")
    args = ap.parse_args()

    ck = torch.load(args.entrada, map_location="cpu", weights_only=False)
    clave_sd = "model" if "model" in ck else ("state_dict" if "state_dict" in ck else None)
    sd = ck[clave_sd] if clave_sd else ck

    mapa = {a: b for a, b in PARES} if args.to == "rope" else {b: a for a, b in PARES}
    nuevo, n = {}, 0
    for k, v in sd.items():
        base = k[7:] if k.startswith("module.") else k
        pref = k[:7] if k.startswith("module.") else ""
        hecho = False
        if base.startswith(args.prefix + "."):
            for viejo, nvo in mapa.items():
                if base.endswith("." + viejo):
                    nuevo[pref + base[: -len(viejo)] + nvo] = v
                    n += 1
                    hecho = True
                    break
        if not hecho:
            nuevo[k] = v

    if n == 0:
        raise SystemExit(
            f"no se renombro ninguna clave bajo '{args.prefix}' con destino "
            f"'{args.to}': el checkpoint ya esta en ese layout, o el prefijo no existe")

    if clave_sd:
        ck[clave_sd] = nuevo
    else:
        ck = nuevo
    torch.save(ck, args.salida)
    print(f"{n} claves renombradas ({args.to}), {len(nuevo)} en total -> {args.salida}")


if __name__ == "__main__":
    main()
