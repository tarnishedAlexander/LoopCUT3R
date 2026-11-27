#!/usr/bin/env python3
"""CUT3R's own weights as a checkpoint this tree can run (24 blocks x 1 pass).

The recurrent models are evaluated from this tree and CUT3R from its own; this
builds CUT3R inside this tree so both go through the same inference and
evaluation code.  CUT3R's encoder blocks become rope RecurrentBlocks with
LayerScale fixed at 1 (same function); everything else is copied unchanged.

    python experiments/scripts/cut3r_in_loop_tree.py \
        --cut3r checkpoints/cut3r_512_dpt_4_64.pth \
        --like checkpoints/init_distilled_full.pth --out checkpoints/cut3r_looptree.pth
"""
import argparse, argparse as _ap, re
import torch

CUT2REC = [("norm1.", "norm1."), ("attn.qkv.", "attn.qkv."), ("attn.proj.", "attn.proj."),
           ("norm2.", "norm2."), ("mlp.fc1.", "mlp.0."), ("mlp.fc2.", "mlp.2.")]


def set_field(cfg, key, value):
    pat = re.compile(rf"\b{key}=([^,()]+|\([^()]*\))")
    return pat.sub(f"{key}={value}", cfg, count=1) if pat.search(cfg) else cfg[:-2] + f", {key}={value}))"


ap = argparse.ArgumentParser()
ap.add_argument("--cut3r", required=True)
ap.add_argument("--like", required=True, help="a full checkpoint of this tree with a 12-block decoder and enc_rope")
ap.add_argument("--out", required=True)
a = ap.parse_args()
cut = torch.load(a.cut3r, map_location="cpu", weights_only=False)
like = torch.load(a.like, map_location="cpu", weights_only=False)
csd = {k.removeprefix("module."): v for k, v in cut["model"].items()}
lsd = like["model"]
cfg = like["args"].model
for k, v in (("num_blocks", 24), ("num_steps", 1), ("use_tbptl", False), ("enc_rope", True)):
    cfg = set_field(cfg, k, v)

out = {}
for k, v in csd.items():
    m = re.match(r"enc_blocks\.(\d+)\.(.*)", k)
    if not m:
        out[k] = v
        continue
    for c, r in CUT2REC:
        if m.group(2).startswith(c):
            out[f"enc_blocks.{m.group(1)}.{r}{m.group(2)[len(c):]}"] = v
for i in range(24):
    out[f"enc_blocks.{i}.ls1.gamma"] = torch.ones(1024)
    out[f"enc_blocks.{i}.ls2.gamma"] = torch.ones(1024)
for k, v in lsd.items():  # tensors CUT3R does not have (mask_token)
    if k not in out and not k.startswith("enc_blocks."):
        out[k] = v
torch.save({"model": out, "args": _ap.Namespace(model=cfg), "epoch": -1}, a.out)
print(f"wrote {a.out}: {len(out)} tensors\nconfig: {cfg[:200]}...")
