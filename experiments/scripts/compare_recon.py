#!/usr/bin/env python3
"""
Compare reconstructions dumped by extract_recon.py.

Every length is divided by that model's own scene radius, so two models that
predict the scene at different global scales are still directly comparable.

    python compare_recon.py cut3r_002.npz loop_002.npz
    python compare_recon.py one.npz                      # single model
"""
import argparse
import sys

import numpy as np

try:
    from scipy.spatial import cKDTree
    HAVE_KD = True
except ImportError:
    HAVE_KD = False


def load(path, conf_thr):
    d = np.load(path, allow_pickle=True)
    r = {k: d[k] for k in d.files}
    r["name"] = path
    r["mask"] = (r["conf_cam"] > conf_thr) & np.isfinite(r["pts_posed"]).all(-1)
    return r


def scene_radius(r):
    """Median distance of confident points from the scene centroid."""
    p = r["pts_posed"][r["mask"]]
    if len(p) == 0:
        return np.nan
    return float(np.median(np.linalg.norm(p - p.mean(0), axis=-1)))


def subsample(p, m, rng):
    return p if len(p) <= m else p[rng.choice(len(p), m, replace=False)]


def nn_median(a, b, m, rng):
    """Median nearest-neighbour distance from a to b."""
    a, b = subsample(a, m, rng), subsample(b, m, rng)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    if HAVE_KD:
        return float(np.median(cKDTree(b).query(a, k=1)[0]))
    out = []
    for i in range(0, len(a), 2000):
        out.append(np.sqrt(((a[i:i+2000, None, :] - b[None]) ** 2).sum(-1)).min(1))
    return float(np.median(np.concatenate(out)))


def analyse(r, m, seed=0):
    rng = np.random.default_rng(seed)
    n = len(r["pts_cam"])
    R = scene_radius(r)
    pos = r["pose_c2w"][:, :3, 3]

    step = np.linalg.norm(np.diff(pos, axis=0), axis=-1) / R      # per-frame camera motion
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=-1).sum() / R)

    agree, ghost, scale, conf = [], [], [], []
    for i in range(n):
        mk = r["mask"][i]
        if mk.sum() == 0:
            agree.append(np.nan); scale.append(np.nan); conf.append(np.nan); continue
        agree.append(float(np.median(
            np.linalg.norm(r["pts_posed"][i][mk] - r["pts_world"][i][mk], axis=-1)) / R))
        scale.append(float(np.median(np.linalg.norm(r["pts_cam"][i][mk], axis=-1))))
        conf.append(float(np.median(r["conf_cam"][i][mk])))

    for i in range(1, n):
        a, b = r["pts_posed"][i][r["mask"][i]], r["pts_posed"][i-1][r["mask"][i-1]]
        ghost.append(nn_median(a, b, m, rng) / R)

    sc = np.array(scale)
    return dict(
        n=n, R=R, path=path,
        step=step, agree=np.array(agree), ghost=np.array(ghost),
        scale=sc, scale_cv=float(np.nanstd(sc) / np.nanmean(sc)),
        conf=np.array(conf),
    )


def summary_rows(a):
    med = np.nanmedian
    return [
        ("frames",                    f"{a['n']}"),
        ("scene radius (model units)", f"{a['R']:.3f}"),
        ("camera path length / R",    f"{a['path']:.3f}"),
        ("camera step / R  median",   f"{med(a['step']):.4f}"),
        ("camera step / R  min",      f"{np.nanmin(a['step']):.4f}"),
        ("frames with step < 10% med", f"{int((a['step'] < 0.1*med(a['step'])).sum())}"),
        ("GHOST  consec NN dist / R", f"{med(a['ghost']):.4f}"),
        ("GHOST  last-quarter",       f"{med(a['ghost'][3*len(a['ghost'])//4:]):.4f}"),
        ("HEADS  |posed-world| / R",  f"{med(a['agree']):.4f}"),
        ("per-frame depth scale CV",  f"{a['scale_cv']:.4f}"),
        ("median confidence",         f"{med(a['conf']):.3f}"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="1 or 2 files; the first is the reference")
    ap.add_argument("--conf", type=float, default=1.5, help="confidence threshold")
    ap.add_argument("--points", type=int, default=20000, help="points sampled per frame")
    args = ap.parse_args()

    if not HAVE_KD:
        print("note: scipy missing, using a slower brute-force nearest neighbour\n")

    res = [(p, analyse(load(p, args.conf), args.points)) for p in args.npz]

    w = 30
    print("=" * (w + 24 * len(res)))
    print("metric".ljust(w) + "".join(p.split("/")[-1][:22].rjust(24) for p, _ in res))
    print("-" * (w + 24 * len(res)))
    rows = [summary_rows(a) for _, a in res]
    for i, (label, _) in enumerate(rows[0]):
        print(label.ljust(w) + "".join(r[i][1].rjust(24) for r in rows))
    print("=" * (w + 24 * len(res)))

    print("\nper-frame camera step / R   (0.0000 means the camera stopped moving)")
    for p, a in res:
        s = "  ".join(f"{v:.3f}" for v in a["step"])
        print(f"  {p.split('/')[-1][:22]:<24} {s}")

    print("\nper-frame ghosting: NN distance to previous frame / R")
    for p, a in res:
        s = "  ".join(f"{v:.3f}" for v in a["ghost"])
        print(f"  {p.split('/')[-1][:22]:<24} {s}")

    if len(res) == 2:
        (_, ref), (_, cmp_) = res
        k = min(len(ref["ghost"]), len(cmp_["ghost"]))
        print("\nrelative to the reference (first file):")
        print(f"  ghosting            {np.nanmedian(cmp_['ghost'][:k]) / np.nanmedian(ref['ghost'][:k]):.2f}x")
        k2 = min(len(ref["step"]), len(cmp_["step"]))
        print(f"  camera step         {np.nanmedian(cmp_['step'][:k2]) / np.nanmedian(ref['step'][:k2]):.2f}x")
        print(f"  head disagreement   {np.nanmedian(cmp_['agree']) / np.nanmedian(ref['agree']):.2f}x")


if __name__ == "__main__":
    main()
