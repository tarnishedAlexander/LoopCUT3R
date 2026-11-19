#!/usr/bin/env python3
"""Volcar las activaciones del encoder, capa por capa, para UN modelo.

Se ejecuta una vez por modelo, cada uno desde SU PROPIO arbol de fuentes (la
regla de tools/ab/README.md: un checkpoint lo ejecuta el arbol en el que se
entreno).  Luego `cross_encoder_cka.py` compara los dos volcados.

Por que hooks y no `_encode_views_all_layers`: ese metodo solo existe en
LoopCUT3R, y ademas su bucle de encoder appendea una salida POR PASO
(croco.py:522: `for step: for blk in enc_blocks: ...; out.append(x)`), asi que
expone 8 representaciones, no 24.  Un hook sobre `enc_blocks` dispara una vez
por aplicacion de bloque, y eso da 24 en los dos modelos:

    CUT3R      24 bloques distintos x 1 pasada   = 24
    LoopCUT3R   3 bloques x 8 pasos              = 24

Las muestras para el CKA son TOKENS, no imagenes promediadas: una imagen da
~576 tokens, asi que 32 imagenes dan miles de muestras en vez de 32.  Los
indices de token se eligen con una semilla fija a partir del numero de tokens,
de modo que ambos modelos se comparan sobre LOS MISMOS parches.

    python tools/ab/dump_encoder_layers.py --repo /app/recurrentTransformer/CUT3R \
        --model_path .../cut3r_512_dpt_4_64.pth --out /tmp/enc_cut3r.npz \
        --images examples/001/*.jpg examples/002/*.jpg
"""
import argparse
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="raiz del repo que ejecuta el ckpt")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens", type=int, default=32,
                    help="tokens muestreados por imagen (semilla fija)")
    ap.add_argument("--model_cfg", default=None)
    args = ap.parse_args()

    # Las rutas de imagen se resuelven ANTES del chdir al repo del modelo.
    imgs = [os.path.abspath(p) for p in args.images]
    imgs = [p for p in imgs if os.path.isfile(p)]
    if not imgs:
        raise SystemExit("ninguna imagen valida")

    sys.path.insert(0, args.repo)
    os.chdir(args.repo)
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(args.model_path)

    import demo as D
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.inference import inference_recurrent

    print(f"cargando {args.model_path}", flush=True)
    kw = {"model_cfg_str": args.model_cfg} if args.model_cfg else {}
    model = ARCroco3DStereo.from_pretrained(args.model_path, **kw).to(args.device)
    model.eval()

    n_blocks = len(model.enc_blocks)
    n_steps = int(getattr(model, "num_steps", 1) or 1)
    esperadas = n_blocks * n_steps
    print(f"enc_blocks={n_blocks}  num_steps={n_steps}  "
          f"-> {esperadas} capturas esperadas por imagen", flush=True)

    capturas = []

    def hook(_mod, _inp, out):
        x = out[0] if isinstance(out, (tuple, list)) else out
        capturas.append(x.detach().float().cpu())

    handles = [b.register_forward_hook(hook) for b in model.enc_blocks]

    bancos = None          # lista de longitud L, cada uno acumula [n, D]
    idx_tok = None
    N_ref = None           # nº de tokens de la primera imagen aceptada
    usadas = 0
    try:
        for p in imgs:
            capturas.clear()
            views = D.prepare_input([p], [True], args.size)
            with torch.no_grad():
                inference_recurrent(views, model, args.device, verbose=False)

            if len(capturas) != esperadas:
                print(f"  !! {os.path.basename(p)}: {len(capturas)} capturas "
                      f"(esperaba {esperadas}) -- saltada", flush=True)
                continue

            # cada captura: [B, N_tokens, D] con B=1
            n_tok = capturas[0].shape[1]
            if idx_tok is None:
                # N_ref se fija con la PRIMERA imagen y no se vuelve a tocar.
                # (Antes reasignaba N en cada vuelta y lo comparaba contra si
                # mismo, asi que la comprobacion no disparaba nunca y entraban
                # imagenes con otra rejilla de parches.)
                N_ref = n_tok
                rng = np.random.default_rng(N_ref)   # semilla = nº de tokens
                idx_tok = rng.choice(N_ref, size=min(args.tokens, N_ref),
                                     replace=False)
                idx_tok.sort()
                bancos = [[] for _ in range(esperadas)]
                print(f"  {N_ref} tokens por imagen, muestreando {len(idx_tok)}",
                      flush=True)
            elif n_tok != N_ref:
                # Una resolucion distinta da otra rejilla de parches: los mismos
                # indices ya no son los mismos sitios de la imagen.
                print(f"  !! {os.path.basename(p)}: {n_tok} tokens != {N_ref} "
                      f"-- saltada (otra resolucion)", flush=True)
                continue

            for li, c in enumerate(capturas):
                bancos[li].append(c[0, idx_tok, :].numpy())
            usadas += 1
            print(f"  ok {os.path.basename(p)}", flush=True)
    finally:
        for h in handles:
            h.remove()

    if not usadas:
        raise SystemExit("ninguna imagen produjo capturas utiles")

    banco = np.stack([np.concatenate(b, axis=0) for b in bancos], axis=0)
    print(f"\nbanco: {banco.shape}  (capas, muestras, dim)  de {usadas} imagenes")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        banco=banco.astype(np.float32),
        n_blocks=n_blocks, n_steps=n_steps, n_imagenes=usadas,
        idx_tok=idx_tok, model_path=np.array(args.model_path),
        repo=np.array(args.repo), imagenes=np.array([os.path.basename(p) for p in imgs]),
    )
    print(f"guardado en {args.out}  ({os.path.getsize(args.out)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
