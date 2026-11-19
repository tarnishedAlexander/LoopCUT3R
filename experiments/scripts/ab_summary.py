#!/usr/bin/env python3
"""
Aggregate every cut3r_<seq>.npz / loop_<seq>.npz pair in a directory into one table.

All lengths are divided by that model's own scene radius, so the two models
being at different global scales does not affect anything.

    python ab_summary.py /app/recon
"""
import argparse
import glob
import os

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_KD = True
except ImportError:
    HAVE_KD = False


def nn_med(a, b, m, rng):
    sub = lambda x: x if len(x) <= m else x[rng.choice(len(x), m, replace=False)]
    a, b = sub(a), sub(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if HAVE_KD:
        return float(np.median(cKDTree(b).query(a, k=1)[0]))
    out = []
    for i in range(0, len(a), 2000):
        out.append(np.sqrt(((a[i:i+2000, None] - b[None]) ** 2).sum(-1)).min(1))
    return float(np.median(np.concatenate(out)))


def analyse(path, conf=1.5, m=20000, seed=0):
    d = np.load(path, allow_pickle=True)
    rng = np.random.default_rng(seed)
    P, W, C = d["pts_posed"], d["pts_world"], d["conf_cam"]
    mask = (C > conf) & np.isfinite(P).all(-1) & np.isfinite(W).all(-1)
    pts = P[mask]
    R = float(np.median(np.linalg.norm(pts - pts.mean(0), axis=-1)))

    ghost = np.array([nn_med(P[i][mask[i]], P[i-1][mask[i-1]], m, rng) / R
                      for i in range(1, len(P))])

    heads, hscale, conf_med = [], [], []
    for i in range(len(P)):
        mk = mask[i]
        if mk.sum() == 0:
            heads.append(np.nan); hscale.append(np.nan); conf_med.append(np.nan); continue
        a, b = P[i][mk], W[i][mk]
        heads.append(float(np.median(np.linalg.norm(a - b, axis=-1))) / R)
        hscale.append(float((a * b).sum() / (b * b).sum()))   # best-fit posed/world scale
        conf_med.append(float(np.median(C[i][mk])))
    heads, hscale, conf_med = map(np.array, (heads, hscale, conf_med))

    pos = d["pose_c2w"][:, :3, 3]
    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / R

    return dict(
        n=len(P), R=R, ghost=ghost, heads=heads, hscale=hscale,
        conf=conf_med, step=step, path=float(step.sum()),
        stopped=int((step < 0.1 * np.median(step)).sum()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="/app/recon")
    ap.add_argument("--conf", type=float, default=1.5)
    ap.add_argument("--points", type=int, default=20000)
    a = ap.parse_args()

    seqs = sorted(
        os.path.basename(p)[6:-4] for p in glob.glob(os.path.join(a.dir, "cut3r_*.npz"))
        if os.path.exists(os.path.join(a.dir, "loop_" + os.path.basename(p)[6:]))
    )
    if not seqs:
        raise SystemExit(f"no cut3r_*.npz / loop_*.npz pairs in {a.dir}")
    if not HAVE_KD:
        print("note: scipy missing, slow brute-force nearest neighbour\n")

    med = np.nanmedian
    res = {}
    for s in seqs:
        res[s] = (analyse(os.path.join(a.dir, f"cut3r_{s}.npz"), a.conf, a.points),
                  analyse(os.path.join(a.dir, f"loop_{s}.npz"), a.conf, a.points))

    hdr = (f"{'sequence':<14}{'n':>4}"
           f"{'GHOST c3r':>11}{'loop':>9}{'x':>7}"
           f"{'HEADS c3r':>11}{'loop':>9}{'x':>7}"
           f"{'path c3r':>10}{'loop':>8}{'x':>7}"
           f"{'scale drift':>13}{'stopped':>9}")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for s in seqs:
        c, l = res[s]
        gc, gl = med(c["ghost"]), med(l["ghost"])
        hc, hl = med(c["heads"]), med(l["heads"])
        drift = np.nanmax(np.abs(l["hscale"] - 1.0))
        print(f"{s:<14}{c['n']:>4}"
              f"{gc:>11.4f}{gl:>9.4f}{gl/gc:>6.1f}x"
              f"{hc:>11.4f}{hl:>9.4f}{hl/hc:>6.1f}x"
              f"{c['path']:>10.2f}{l['path']:>8.2f}{l['path']/c['path']:>6.2f}x"
              f"{drift:>13.2f}{str(c['stopped'])+'/'+str(l['stopped']):>9}")
    print("=" * len(hdr))
    print("GHOST = nearest-neighbour distance to the previous frame / scene radius")
    print("HEADS = |pose o pts_self  -  pts_world| / scene radius  (needs no reference)")
    print("path  = camera path length / scene radius")
    print("scale drift = max |posed/world scale - 1| over frames, LoopCUT3R")
    print("stopped = frames whose camera step is under 10% of the median (c3r/loop)")

    for s in seqs:
        c, l = res[s]
        if l["n"] < 8:
            continue
        print(f"\n--- {s}: per-frame posed/world scale (1.00 = heads agree) ---")
        print("  CUT3R      " + " ".join(f"{v:5.2f}" for v in c["hscale"]))
        print("  LoopCUT3R  " + " ".join(f"{v:5.2f}" for v in l["hscale"]))
        print(f"--- {s}: per-frame camera step / R (0.00 = camera stopped) ---")
        print("  CUT3R      " + " ".join(f"{v:5.2f}" for v in c["step"]))
        print("  LoopCUT3R  " + " ".join(f"{v:5.2f}" for v in l["step"]))


if __name__ == "__main__":
    main()
