#!/usr/bin/env python3
"""
Experiment A, restricted to the pairs where BOTH models produce confident
points, so the two columns are over the same set of pairs.

GHOST(k) = median nearest-neighbour distance from frame k's confident points to
frame k-1's, divided by frame k-1's median depth (not the scene radius: a
2-frame window sees less of the scene than a 15-frame one, so scene radius is
not comparable between the two runs, while median depth is a single-frame
quantity and cancels any global scale difference).
"""
import os
import numpy as np
from scipy.spatial import cKDTree

import argparse

CONF, MPTS = 1.5, 20000


def _args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir", help="directory of .npz dumps from extract_multi.py")
    ap.add_argument("--seq", default="002", help="base sequence name")
    ap.add_argument("--frames", type=int, default=15, help="frames in the full sequence")
    ap.add_argument("--ref", default="cut3r", help="prefix of the reference model")
    ap.add_argument("--cmp", default="loop3b", help="prefix of the model under test")
    ap.add_argument("--conf", type=float, default=CONF)
    return ap.parse_args()


def load(p):
    d = np.load(p, allow_pickle=True)
    r = {k: d[k] for k in ("pts_cam", "pts_posed", "pts_world", "conf_cam")}
    r["mask"] = (r["conf_cam"] > ARGS.conf) & np.isfinite(r["pts_posed"]).all(-1)
    return r


def nn(a, b, rng):
    if len(a) < 100 or len(b) < 100:
        return np.nan
    if len(a) > MPTS: a = a[rng.choice(len(a), MPTS, replace=False)]
    if len(b) > MPTS: b = b[rng.choice(len(b), MPTS, replace=False)]
    return float(np.median(cKDTree(b).query(a, k=1)[0]))


def ghost(r, i, rng):
    mi, mj = r["mask"][i], r["mask"][i - 1]
    if mj.sum() < 100:
        return np.nan
    s = float(np.median(np.linalg.norm(r["pts_cam"][i - 1][mj], axis=-1)))
    return nn(r["pts_posed"][i][mi], r["pts_posed"][i - 1][mj], rng) / s


def collect(tag):
    rng = np.random.default_rng(0)
    D, seq = ARGS.recon_dir, ARGS.seq
    full = load(f"{D}/{tag}_{seq}.npz")
    out = {}
    for i in range(1, ARGS.frames):
        p = f"{D}/{tag}_pair_{seq}_{i:02d}_{i+1:02d}.npz"
        pr = load(p)
        out[i] = (ghost(pr, 1, rng), ghost(full, i, rng),
                  float(pr["mask"][0].mean()), float(pr["mask"][1].mean()),
                  float(full["mask"][i].mean()))
    return out


if __name__ == "__main__":
    ARGS = _args()
    c, l = collect(ARGS.ref), collect(ARGS.cmp)
    ok = [i for i in c if np.isfinite(c[i][0]) and np.isfinite(c[i][1])
          and np.isfinite(l[i][0]) and np.isfinite(l[i][1])]
    print("=" * 100)
    print(f"EXPERIMENT A  --  one consecutive pair of {ARGS.seq} as its own 2-frame sequence")
    print(f"                  (clean initial state) vs the same pair inside the full {ARGS.frames}")
    print("=" * 100)
    print(f"                  {ARGS.ref:<28s} {ARGS.cmp}")
    print("  pair      alone    in full   ratio       alone    in full   ratio"
          "      loop / cut3r alone   conf frac pair (loop)")
    for i in sorted(c):
        ca, cf = c[i][0], c[i][1]
        la, lf = l[i][0], l[i][1]
        f = lambda v: "     nan" if not np.isfinite(v) else f"{v:8.4f}"
        rr = lambda a, b: "   nan " if not (np.isfinite(a) and np.isfinite(b)) else f"{a/b:5.2f}x"
        mark = "" if i in ok else "   (dropped)"
        print(f"  {i:02d}->{i+1:02d} {f(ca)} {f(cf)} {rr(ca,cf)}   {f(la)} {f(lf)} {rr(la,lf)}"
              f"        {rr(la,ca)}         {l[i][2]:.2f}/{l[i][3]:.2f}{mark}")
    med = lambda xs: float(np.median(xs))
    ca = med([c[i][0] for i in ok]); cf = med([c[i][1] for i in ok])
    la = med([l[i][0] for i in ok]); lf = med([l[i][1] for i in ok])
    print("-" * 100)
    # El veredicto nombra la referencia y el n de pares EN LA MISMA LINEA, a
    # proposito: el GHOST no es un numero suelto. `prime` da 3.07x contra
    # cut3rbi y 2.77x contra cut3r, y la mediana va sobre los pares que AMBOS
    # modelos resuelven, que son 10 u 11 segun el modelo. Un multiplicador
    # copiado sin esos dos datos no se puede comparar con otro.
    print(f"  median over the {len(ok)}/{ARGS.frames - 1} pairs both models handle: "
          f"{ARGS.ref} {ca:.4f} alone / {cf:.4f} in full ({ca/cf:.2f}x),  "
          f"{ARGS.cmp} {la:.4f} / {lf:.4f} ({la/lf:.2f}x)")
    print(f"  GHOST  {ARGS.cmp} / {ARGS.ref}  (n={len(ok)}):"
          f"   {la/ca:.2f}x in isolated pairs,"
          f"   {lf/cf:.2f}x inside the full sequence")
    if ARGS.ref == "cut3r":
        print("  nota: --ref cut3r es el CUT3R FRIO. Para un modelo calentado "
              "(prime/bi) la\n        comparacion justa es --ref cut3rbi: el "
              "burn-in tambien ayuda a CUT3R.")
