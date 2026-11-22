#!/usr/bin/env python3
"""Build a starting checkpoint: LoopCUT3R's recurrent encoder + CUT3R for the rest.

LoopCUT3R's decoder, heads, pose memory and state tokens were trained from
random initialisation (cosine similarity to CUT3R's weights ~0), while CUT3R's
came from DUSt3R/CroCo pretraining plus ~185 epochs.  This script separates
that confound from depth recurrence.

    python experiments/scripts/init_from_cut3r.py \
        --loop  checkpoints/loopcut3r_512_dpt.pth \
        --cut3r checkpoints/cut3r_512_dpt_4_64.pth \
        --decoder full   --out checkpoints/init_hybrid_full.pth
    python experiments/scripts/init_from_cut3r.py ... \
        --decoder shared --dec_src 0,4,8 --out checkpoints/init_loop_cut3rdec.pth

--decoder full    CUT3R's 12 independent decoder blocks per branch (hybrid:
                  only the encoder is recurrent).
--decoder shared  LoopCUT3R's 3 shared blocks x 4 steps, initialised from the
                  CUT3R blocks listed in --dec_src.
Encoder-side tensors (patch embeddings, recurrent image and ray-map encoders,
their norms, mask tokens) come from --loop; everything else from --cut3r.
--set key=value overrides a field of the model config (e.g. enc_rope=False).
"""
import argparse, argparse as _ap, re, sys, os

import torch

ENCODER_PREFIXES = (
    "patch_embed.", "patch_embed_ray_map.",
    "enc_blocks.", "enc_blocks_ray_map.",
    "enc_norm.", "enc_norm_ray_map.",
    "masked_img_token", "masked_ray_map_token",
)


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.removeprefix("module."): v for k, v in ck.get("model", ck).items()}
    return ck, sd


def set_field(cfg, key, value):
    pat = re.compile(rf"\b{key}=([^,()]+|\([^()]*\))")
    if pat.search(cfg):
        return pat.sub(f"{key}={value}", cfg, count=1)
    return cfg[:-2] + f", {key}={value}))"


ap = argparse.ArgumentParser()
ap.add_argument("--loop", required=True)
ap.add_argument("--cut3r", required=True)
ap.add_argument("--decoder", choices=["full", "shared"], required=True)
ap.add_argument("--dec_src", default="0,4,8")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--loop_cfg", default=None, help="training YAML whose `model:` line describes --loop, "
                "for checkpoints saved without args")
ap.add_argument("--out", required=True)
a = ap.parse_args()

loop_ck, loop_sd = load(a.loop)
_, cut_sd = load(a.cut3r)
if "args" in loop_ck:
    cfg = loop_ck["args"].model
elif a.loop_cfg:
    cfg = next(l.split(":", 1)[1].strip().strip('"') for l in open(a.loop_cfg) if l.startswith("model:"))
else:
    sys.exit(f"{a.loop} has no embedded config: pass --loop_cfg config/recurrent_stage3b.yaml")

if a.decoder == "full":
    for k in ("dec_num_blocks_img", "dec_num_steps_img", "dec_num_blocks_state", "dec_num_steps_state"):
        cfg = set_field(cfg, k, 12)
    cfg = set_field(cfg, "dec_use_tbptl", False)
for kv in a.set:
    k, v = kv.split("=", 1)
    cfg = set_field(cfg, k, v)

src = [int(x) for x in a.dec_src.split(",")]
out, origin, kept = {}, {"loop": 0, "cut3r": 0}, []
for k, v in loop_sd.items():  # loop_sd fixes the key set for --decoder shared
    if k.startswith(ENCODER_PREFIXES):
        out[k] = v; origin["loop"] += v.numel()
if a.decoder == "full":
    for k, v in cut_sd.items():
        if not k.startswith(ENCODER_PREFIXES):
            out[k] = v; origin["cut3r"] += v.numel()
    for k, v in loop_sd.items():
        if k not in out and not k.startswith(("dec_blocks.", "dec_blocks_state.")):
            kept.append(k); out[k] = v; origin["loop"] += v.numel()
else:
    for k, v in loop_sd.items():
        if k.startswith(ENCODER_PREFIXES):
            continue
        m = re.match(r"(dec_blocks(?:_state)?)\.(\d+)\.(.*)", k)
        ck = f"{m.group(1)}.{src[int(m.group(2))]}.{m.group(3)}" if m else k
        if ck not in cut_sd or cut_sd[ck].shape != v.shape:
            kept.append(k); out[k] = v; origin["loop"] += v.numel()  # e.g. mask_token, not in CUT3R
            continue
        out[k] = cut_sd[ck]; origin["cut3r"] += v.numel()

# The recurrent encoder stores its attention as nn.MultiheadAttention without
# RoPE and as CroCo's Attention with it; same tensors, different names
# (experiments/scripts/remap_enc_attn.py).  Match the layout to the final config.
PAIRS = [("attn.in_proj_weight", "attn.qkv.weight"), ("attn.in_proj_bias", "attn.qkv.bias"),
         ("attn.out_proj.weight", "attn.proj.weight"), ("attn.out_proj.bias", "attn.proj.bias")]
rope = re.search(r"\benc_rope=True", cfg) is not None
ren = {b: a for a, b in PAIRS} if not rope else dict(PAIRS)
n_ren = 0
for k in [k for k in out if k.startswith("enc_blocks.")]:
    for old_s, new_s in ren.items():
        if k.endswith("." + old_s):
            out[k[: -len(old_s)] + new_s] = out.pop(k); n_ren += 1
            break
if n_ren:
    print(f"renamed {n_ren} encoder attention tensors to the {'rope' if rope else 'no-rope'} layout")

torch.save({"model": out, "args": _ap.Namespace(model=cfg), "epoch": -1,
            "init": {"loop": os.path.basename(a.loop), "cut3r": os.path.basename(a.cut3r),
                     "decoder": a.decoder, "dec_src": src if a.decoder == "shared" else None}},
           a.out)
print(f"wrote {a.out}: {len(out)} tensors, "
      f"{origin['loop'] / 1e6:.2f} M from LoopCUT3R, {origin['cut3r'] / 1e6:.2f} M from CUT3R")
print("kept from LoopCUT3R (no CUT3R counterpart):", kept)
print("config:", cfg)
