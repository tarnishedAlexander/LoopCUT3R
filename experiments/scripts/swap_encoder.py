#!/usr/bin/env python3
"""CUT3R's 24-block encoder + another checkpoint's decoder, heads and pose memory.

Diagnostic for the attribution result: attr_hybrid_full (recurrent encoder +
CUT3R's full decoder) still fails on ScanNet.  Putting CUT3R's own encoder back
in front of that decoder tells whether the decoder side is still sound (then the
gap is the recurrent encoder) or was damaged by fine-tuning on its features.

    python experiments/scripts/swap_encoder.py \
        --cut3r checkpoints/cut3r_512_dpt_4_64.pth \
        --other checkpoints/attr_hybrid_full/checkpoints/checkpoint-final.pth \
        --out   checkpoints/cut3renc_hybriddec.pth

The result keeps CUT3R's config and is evaluated from a CUT3R checkout:
    bash tools/eval/run_benchmarks.sh SWAP checkpoints/cut3renc_hybriddec.pth /path/to/CUT3R
Only works when --other's decoder has CUT3R's layout (12 + 12 independent
blocks, as in attr_hybrid_full); it refuses anything else.
"""
import argparse, os, sys
import torch

ENCODER_PREFIXES = (
    "patch_embed.", "patch_embed_ray_map.",
    "enc_blocks.", "enc_blocks_ray_map.",
    "enc_norm.", "enc_norm_ray_map.",
    "masked_img_token", "masked_ray_map_token",
)

ap = argparse.ArgumentParser()
ap.add_argument("--cut3r", required=True)
ap.add_argument("--other", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

cut = torch.load(a.cut3r, map_location="cpu", weights_only=False)
oth = torch.load(a.other, map_location="cpu", weights_only=False)
strip = lambda sd: {k.removeprefix("module."): v for k, v in sd.items()}
csd, osd = strip(cut.get("model", cut)), strip(oth.get("model", oth))

out, swapped, bad = {}, 0, []
for k, v in csd.items():
    if k.startswith(ENCODER_PREFIXES):
        out[k] = v
        continue
    if k not in osd or osd[k].shape != v.shape:
        bad.append(k)
        continue
    out[k] = osd[k]; swapped += v.numel()
extra = [k for k in osd if k not in csd and not k.startswith(ENCODER_PREFIXES)]
if bad:
    sys.exit(f"{len(bad)} non-encoder CUT3R tensors have no counterpart in --other, e.g. {bad[:5]}")
print(f"{swapped / 1e6:.2f} M non-encoder parameters taken from {os.path.basename(a.other)}")
print("tensors only in --other (dropped):", extra)
ck = dict(cut); ck["model"] = out
ck["swap"] = {"encoder": os.path.basename(a.cut3r), "rest": a.other}
torch.save(ck, a.out)
print("wrote", a.out)
