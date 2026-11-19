#!/usr/bin/env python3
"""Compone un checkpoint mezclando el ENCODER de un modelo y el DECODER de otro.

Para el experimento N2 (swap 2x2): atribuir el 3x de GHOST entre el encoder
recurrente y el decoder recurrente. Ver docs/attractor_mechanism.md seccion 6.

Los dos modelos comparten la clase de bloque en el DECODER (verificado: mismas
claves y mismas formas en `dec_blocks` y `dec_blocks_state`), asi que ese brazo
se copia tal cual. En el ENCODER no: CUT3R usa `Block` (`attn.qkv`, `mlp.fc1/
fc2`, sin LayerScale) y LoopCUT3R usa `RecurrentBlock` (`attn.in_proj_*`/
`out_proj` o `attn.qkv` segun rope, `mlp.0/2`, con `ls1`/`ls2`). La conversion
es EXACTA, no una aproximacion:

    attn.qkv.*   <-> attn.in_proj_*        (misma forma, mismo orden q,k,v)
    attn.proj.*  <-> attn.out_proj.*
    mlp.fc1.*    <-> mlp.0.*
    mlp.fc2.*    <-> mlp.2.*
    ls1.gamma = ls2.gamma = 1             (LayerScale identidad == sin LayerScale)

Con `enc_rope=True` el `RecurrentBlock` usa la MISMA clase `Attention` que
`Block`, y con gamma=1 su forward queda `x + attn(norm1(x), xpos)` y
`x + mlp(norm2(x))`: identico a `Block.forward` con `drop_path=Identity`.

**El control que lo demuestra** es reconstruir CUT3R entero en el layout
recurrente (`--enc cut3r --dec cut3r`) y comprobar que la sonda da los mismos
numeros que CUT3R. Si no los da, ninguna celda del 2x2 significa nada.

    python tools/ab/build_swap.py --enc cut3r --dec loop --out .scratch/ab/swap_EL.pth
"""
import argparse, os, torch

DONANTES = {
    "cut3r": "checkpoints/cut3r_512_dpt_4_64.pth",
    # stage3b con la atencion ya remapeada al layout rope: el encoder recurrente
    # CON posicion. Obligatorio para N2: si el brazo loop fuera sin posicion, el
    # 2x2 compararia "con rope" contra "sin rope" y reportaria la diferencia como
    # coste del weight sharing. Es el confound que bloqueaba N2 (seccion 1b).
    "loop":  "checkpoints/loop3b_encrope_init.pth",
}

# A que brazo pertenece cada familia de claves de primer nivel.
ENC = ("patch_embed", "patch_embed_ray_map", "enc_blocks", "enc_blocks_ray_map",
       "enc_norm", "enc_norm_ray_map", "masked_img_token", "masked_ray_map_token")
DEC = ("decoder_embed", "decoder_embed_state", "dec_blocks", "dec_blocks_state",
       "dec_norm", "dec_norm_state", "downstream_head", "pose_retriever",
       "pose_token", "register_tokens", "mask_token")

# La ATENCION no se renombra: con `enc_rope=True` el `RecurrentBlock` construye
# la misma clase `Attention` que `Block`, asi que las claves ya coinciden
# (`attn.qkv.*`, `attn.proj.*`). El renombrado in_proj/out_proj <-> qkv/proj
# vive en tools/ab/remap_enc_attn.py y hace falta en el otro sentido, para pasar
# un checkpoint loop SIN rope al layout con rope.
PARES_MLP = [("mlp.fc1.", "mlp.0."), ("mlp.fc2.", "mlp.2.")]


def cargar(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck)) if isinstance(ck, dict) else ck
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def familia(k):
    return k.split(".")[0]


# Los bloques del ray-map se construyen SIN rope (model.py:348 no le pasa
# `rope`), asi que usan `nn.MultiheadAttention` y esperan in_proj/out_proj
# aunque `enc_blocks` espere qkv/proj. Convertirlos pierde la posicion, y eso
# NO afecta a ninguna medida de este experimento: con entrada de imagenes
# `selected_ray_maps.size(0) == 0` (model.py:910) y la rama nunca corre; en su
# lugar se suma `masked_ray_map_token`. Se convierten solo para que el
# checkpoint cargue completo y el control de carga pueda ser estricto.
PARES_ATTN_SIN_ROPE = [("attn.qkv.weight",  "attn.in_proj_weight"),
                       ("attn.qkv.bias",    "attn.in_proj_bias"),
                       ("attn.proj.weight", "attn.out_proj.weight"),
                       ("attn.proj.bias",   "attn.out_proj.bias")]


def a_recurrente(sd):
    """Claves de `Block` -> claves de `RecurrentBlock`. Anade gamma=1.

    `enc_blocks` va al layout CON rope (misma clase `Attention`, las claves de
    atencion ya coinciden); `enc_blocks_ray_map` al layout SIN rope.
    """
    prefijos = ("enc_blocks.", "enc_blocks_ray_map.")
    out, convertidas = {}, 0
    for k, v in sd.items():
        if not any(k.startswith(p) for p in prefijos):
            out[k] = v
            continue
        nk = k
        for mlp_a, mlp_b in PARES_MLP:
            if mlp_a in nk:
                nk = nk.replace(mlp_a, mlp_b)
                convertidas += 1
        if nk.startswith("enc_blocks_ray_map."):
            for a, b in PARES_ATTN_SIN_ROPE:
                if nk.endswith(a):
                    nk = nk[: -len(a)] + b
                    convertidas += 1
        out[nk] = v
    # LayerScale identidad en cada bloque convertido
    dims = {}
    for k, v in out.items():
        if k.endswith("norm1.weight") and any(k.startswith(p) for p in prefijos):
            dims[".".join(k.split(".")[:2])] = v.shape[0]
    anadidas = 0
    for b, d in dims.items():
        for ls in ("ls1.gamma", "ls2.gamma"):
            if f"{b}.{ls}" not in out:
                out[f"{b}.{ls}"] = torch.ones(d)
                anadidas += 1
    return out, convertidas, anadidas


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enc", choices=sorted(DONANTES), required=True)
    ap.add_argument("--dec", choices=sorted(DONANTES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enc_sd = cargar(DONANTES[args.enc])
    dec_sd = cargar(DONANTES[args.dec])

    fuera = {}
    # el brazo del encoder, convertido al layout recurrente si viene de CUT3R
    if args.enc == "cut3r":
        enc_sd, conv, anad = a_recurrente(enc_sd)
        print(f"  encoder cut3r -> layout recurrente: {conv} claves mlp renombradas, "
              f"{anad} gamma=1 anadidas")
    for k, v in enc_sd.items():
        if familia(k) in ENC:
            fuera[k] = v
    n_enc = len(fuera)
    for k, v in dec_sd.items():
        if familia(k) in DEC:
            assert k not in fuera, f"clave en los dos brazos: {k}"
            fuera[k] = v
    n_dec = len(fuera) - n_enc

    sin_asignar = sorted({familia(k) for k in list(enc_sd) + list(dec_sd)}
                         - set(ENC) - set(DEC))
    assert not sin_asignar, f"familias sin brazo asignado: {sin_asignar}"

    # CUT3R no guarda `mask_token`; la clase si lo define. Solo se usa en la rama
    # de pre-entrenamiento con mascaras (croco.py:570, `if masks1 is None` toma
    # el camino de inferencia), asi que su valor no entra en ninguna medida.
    if "mask_token" not in fuera:
        emb = fuera["pose_token"].shape[-1] if "pose_token" in fuera else 768
        fuera["mask_token"] = torch.zeros(1, 1, emb)
        print(f"  mask_token a cero: {args.dec} no lo trae y la rama con mascaras "
              f"no corre en inferencia")

    nb = len([k for k in fuera if k.endswith("norm1.weight") and k.startswith("enc_blocks.")])
    print(f"  encoder: {n_enc} tensores de {args.enc} ({nb} bloques)")
    print(f"  decoder: {n_dec} tensores de {args.dec}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # SE GUARDA EL STATE DICT DESNUDO, no {"model": ...}: load_model
    # (src/dust3r/model.py:74) solo mira dentro de ckpt["model"] cuando el
    # checkpoint trae "args", y aqui no puede traerlos porque la geometria es
    # una mezcla nueva. En la otra rama hace `state_dict = ckpt` tal cual, asi
    # que envolverlo deja un dict de UNA clave y el load_state_dict(strict=False)
    # no carga NADA sin fallar. Paso de verdad: la primera version envuelta dio
    # las 1300 claves en missing_keys y una sonda que medía pesos aleatorios.
    torch.save(fuera, args.out)
    print(f"  escrito {args.out}  ({os.path.getsize(args.out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
