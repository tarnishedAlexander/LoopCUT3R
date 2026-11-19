#!/usr/bin/env python3
"""
Experiment B, done over a FIXED pixel set.

The first pass compared median |pts_cam| per frame under each frame's own
confidence mask, so a change in which pixels are confident could masquerade as
a change in depth.  Here every quantity for step k is computed only over
mask_0 AND mask_k, and the depth-scale ratio is a per-pixel median of
|pts_cam[k]| / |pts_cam[0]| rather than a ratio of two medians.

  scale alpha   per-pixel median depth ratio to frame 0   (1.0 = no change)
  disp          median |pts_cam[k] - pts_cam[0]| / depth_0
  resid         the same after dividing frame 0 out by alpha: the part of the
                change that is NOT a global scale, i.e. shape
"""
import argparse
import os

import numpy as np


def load(p):
    d = np.load(p, allow_pickle=True)
    r = {k: d[k] for k in ("pts_cam", "pts_posed", "pts_world", "conf_cam", "pose_c2w")}
    r["mask"] = (r["conf_cam"] > ARGS.conf) & np.isfinite(r["pts_cam"]).all(-1) \
                                       & np.isfinite(r["pts_posed"]).all(-1)
    return r


def rot_deg(A, B):
    R = A[:3, :3] @ B[:3, :3].T
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def table(seq):
    print(f"\n--- {seq}   (the same image fed 8 times)")
    for tag, label in MODELS:
        p = f"{ARGS.recon_dir}/{tag}_{seq}.npz"
        if not os.path.exists(p):
            print(f"  {label}: no file"); continue
        r = load(p)
        n = len(r["pts_cam"])
        print(f"  {label}")
        print("    step   conf frac   shared px   scale alpha      disp     resid"
              "    cam t   cam rot")
        for k in range(n):
            mk = r["mask"][0] & r["mask"][k]
            frac = float(r["mask"][k].mean())
            if mk.sum() < 100:
                print(f"    {k:>4}      {frac:6.3f}      {int(mk.sum()):7d}"
                      "          --        --        --       --        --")
                continue
            d0 = np.linalg.norm(r["pts_cam"][0][mk], axis=-1)
            dk = np.linalg.norm(r["pts_cam"][k][mk], axis=-1)
            s0 = float(np.median(d0))
            alpha = float(np.median(dk / np.maximum(d0, 1e-8)))
            disp = float(np.median(np.linalg.norm(
                r["pts_cam"][k][mk] - r["pts_cam"][0][mk], axis=-1))) / s0
            resid = float(np.median(np.linalg.norm(
                r["pts_cam"][k][mk] - alpha * r["pts_cam"][0][mk], axis=-1))) / s0
            t = float(np.linalg.norm(
                r["pose_c2w"][k][:3, 3] - r["pose_c2w"][0][:3, 3])) / s0
            print(f"    {k:>4}      {frac:6.3f}      {int(mk.sum()):7d}"
                  f"      {alpha:7.4f}   {disp:7.4f}   {resid:7.4f}"
                  f"  {t:7.4f}   {rot_deg(r['pose_c2w'][k], r['pose_c2w'][0]):7.3f}")


def _args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir", help="directory of .npz dumps from extract_multi.py")
    ap.add_argument("--seqs", nargs="+", required=True,
                    help="repeat-sequence names, e.g. rep_002_01_x8")
    ap.add_argument("--models", nargs="+", required=True,
                    help="prefix=label pairs, e.g. cut3r=CUT3R loop3b=LoopCUT3R")
    ap.add_argument("--conf", type=float, default=1.5)
    return ap.parse_args()


if __name__ == "__main__":
    ARGS = _args()
    MODELS = [tuple(m.split("=", 1)) if "=" in m else (m, m) for m in ARGS.models]
    print("=" * 92)
    print("EXPERIMENT B (fixed pixel set)  --  identical input, so every row should read")
    print("                                   alpha 1.0000, disp 0, resid 0, t 0, rot 0")
    print("=" * 92)
    for s in ARGS.seqs:
        table(s)
