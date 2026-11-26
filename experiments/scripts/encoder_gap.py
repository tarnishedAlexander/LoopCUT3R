#!/usr/bin/env python3
"""How far is a recurrent encoder from CUT3R's encoder, per pass and per dataset?

The two models live in different source trees with the same package names, so
each runs in its own process and writes its activations to disk:

    # 1. images, preprocessed once (any tree)
    python experiments/scripts/encoder_gap.py prep --out OUT/images.pt
    # 2. activations, one call per model, from that model's own tree
    python experiments/scripts/encoder_gap.py dump --repo /path/to/CUT3R \
        --ckpt cut3r.pth --images OUT/images.pt --out OUT/cut3r.pt
    python experiments/scripts/encoder_gap.py dump --repo . \
        --ckpt ckpt.pth --images OUT/images.pt --out OUT/loop.pt
    # 3. compare
    python experiments/scripts/encoder_gap.py compare --ref OUT/cut3r.pt OUT/loop.pt ...

CUT3R's 24 blocks are sampled every 24 / passes blocks, so recurrent pass k is
compared with CUT3R block k * 24 / passes; the last entry of both is the output
after the final norm, which is what the decoder receives.  Per image and depth:
mean token cosine similarity, relative L2 error and linear CKA over tokens.
"""
import argparse, glob, os, sys

import torch

SETS = {  # dataset: (glob of sequence folders, frames per sequence)
    "scannet": ("data/scannetv2/scene*/color_90", 3),
    "tum": ("data/tum/rgbd_dataset_*/rgb_90", 3),
    "sintel": ("data/sintel/training/final/*", 3),
}


def setup_tree(repo, ckpt):
    repo = os.path.abspath(repo)
    sys.path.insert(0, repo); os.chdir(repo)
    from add_ckpt_path import add_path_to_dust3r
    add_path_to_dust3r(ckpt)


def prep(a):
    setup_tree(a.repo, "")
    from src.dust3r.utils.image import load_images_for_eval
    items = []
    for name, (pat, n) in SETS.items():
        for seq in sorted(glob.glob(pat)):
            if not os.path.isdir(seq):
                continue
            files = sorted(f for f in glob.glob(os.path.join(seq, "*")) if f.lower().endswith((".jpg", ".png")))
            if len(files) < n:
                continue
            pick = [files[round(i * (len(files) - 1) / max(1, n - 1))] for i in range(n)]
            for v in load_images_for_eval(pick, size=512, verbose=False, crop=True):
                items.append({"set": name, "seq": os.path.basename(os.path.dirname(seq)) if name != "sintel" else os.path.basename(seq),
                              "img": v["img"], "true_shape": torch.as_tensor(v["true_shape"])})
    torch.save(items, a.out)
    print({s: sum(i["set"] == s for i in items) for s in SETS}, "images ->", a.out)


@torch.no_grad()
def dump(a):
    ckpt = os.path.abspath(a.ckpt); images = os.path.abspath(a.images); out = os.path.abspath(a.out)
    setup_tree(a.repo, ckpt)
    from src.dust3r.model import ARCroco3DStereo
    m = ARCroco3DStereo.from_pretrained(ckpt).cuda().eval()
    steps = getattr(m, "num_steps", None)
    recurrent = steps is not None and len(m.enc_blocks) * int(steps) == 24 and len(m.enc_blocks) < 24
    acts = []
    for it in torch.load(images, weights_only=False):
        img = it["img"].cuda(); ts = it["true_shape"].reshape(1, 2).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x, pos = m.patch_embed(img, true_shape=ts)
            layers = []
            if recurrent:
                for _ in range(int(steps)):
                    for blk in m.enc_blocks:
                        x = blk(x, pos)
                    layers.append(x)
            else:
                for i, blk in enumerate(m.enc_blocks):
                    x = blk(x, pos)
                    if (i + 1) % 3 == 0:
                        layers.append(x)
            layers.append(m.enc_norm(x))
        acts.append(torch.stack([l[0].float().cpu().half() for l in layers]))  # [depths+1, tokens, C]
    torch.save({"acts": acts, "ckpt": ckpt, "recurrent": recurrent}, out)
    print(f"{len(acts)} images, {acts[0].shape[0]} depths, recurrent={recurrent} -> {out}")


def cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    xy = (x.T @ y).norm() ** 2
    return (xy / ((x.T @ x).norm() * (y.T @ y).norm())).item()


def compare(a):
    imgs = torch.load(a.images, weights_only=False)
    ref = torch.load(a.ref, weights_only=False)["acts"]
    for path in a.models:
        acts = torch.load(path, weights_only=False)["acts"]
        print(f"\n== {os.path.basename(path)} vs {os.path.basename(a.ref)}")
        rows = {}
        for it, r, x in zip(imgs, ref, acts):
            for d in range(r.shape[0]):
                rr, xx = r[d].float(), x[d].float()
                cos = torch.nn.functional.cosine_similarity(rr, xx, dim=-1).mean().item()
                rel = ((xx - rr).norm() / rr.norm()).item()
                rows.setdefault((it["set"], d), []).append((cos, rel, cka(rr, xx)))
        n = r.shape[0]
        print(f"{'set':8s}" + "".join(f"  pass{d + 1:<2d}cos/CKA" if d < n - 1 else "   final cos / relL2 / CKA" for d in range(n)))
        for s in SETS:
            if (s, 0) not in rows:
                continue
            cells = []
            for d in range(n):
                v = torch.tensor(rows[(s, d)]).mean(0)
                cells.append(f"  {v[0]:.2f}/{v[2]:.2f}    " if d < n - 1 else f"   {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f}")
            print(f"{s:8s}" + "".join(cells))


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("prep"); p.add_argument("--repo", default="."); p.add_argument("--out", required=True)
p = sub.add_parser("dump"); p.add_argument("--repo", required=True); p.add_argument("--ckpt", required=True)
p.add_argument("--images", required=True); p.add_argument("--out", required=True)
p = sub.add_parser("compare"); p.add_argument("--ref", required=True); p.add_argument("--images", required=True)
p.add_argument("models", nargs="+")
a = ap.parse_args()
{"prep": prep, "dump": dump, "compare": compare}[a.cmd](a)
