#!/usr/bin/env python3
"""Final benchmark table.  Every relpose average is over the scenes ALL
compared labels scored, so no row benefits from a scene another row dropped."""
import glob, json, os, re
import numpy as np

import argparse
import pathlib

_HERE = pathlib.Path(__file__).resolve()
_DEFAULT = _HERE.parents[2] / "eval_results"   # <repo>/eval_results

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--results-dir", default=str(_DEFAULT),
                 help="eval_results directory (default: %(default)s)")
R = _ap.parse_args().results_dir


def relpose_scenes(lab, d):
    """{scene: (ATE, RPE_rot, RPE_t)} from the per-scene evo metric files."""
    out = {}
    for f in sorted(glob.glob(f"{R}/relpose/{d}_{lab}/**/*_eval_metric.txt", recursive=True)):
        t = open(f).read()
        g = lambda s: (lambda m: float(m.group(1)) if m else None)(
            re.search(re.escape(s) + r".*?mean\s+([0-9.eE+-]+)", t, re.S))
        v = (g("APE w.r.t. translation"), g("RPE w.r.t. rotation"), g("RPE w.r.t. translation"))
        if None not in v:
            out[os.path.basename(f).replace("_eval_metric.txt", "")] = v
    return out


def depth(lab, d):
    p = f"{R}/video_depth/{d}_{lab}/result_scale&shift.json"
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return (j.get("Abs Rel"), j.get("RMSE"), j.get("δ < 1.25"))


LABELS = [
    ("CUT3R      broken  (Loop tree, random enc)", "CUT3R_cut3r_512_dpt_4_64"),
    ("CUT3R      correct (own tree)",              "CUT3R_fixed"),
    ("LoopCUT3R  stage3b (Oct, older evo)",        "LoopCUT3R_recurrent_stage3b_checkpoint_last"),
    ("LoopCUT3R  stage3b (re-run, evo 1.30.6)",    "LoopCUT3R_stage3b_rerun"),
]

print("=" * 100)
print("CAMERA POSE (relpose)   Sim(3)-aligned, mean over the scenes every label scored.")
print("                        Lower is better.")
print("=" * 100)
agg = {}
for d in ("scannet", "tum", "sintel"):
    per = {lab: relpose_scenes(lab, d) for _, lab in LABELS}
    common = set.intersection(*[set(v) for v in per.values() if v]) if any(per.values()) else set()
    print(f"\n-- {d}   ({len(common)} scenes common to all labels)")
    dropped = {lab: sorted(set(v) - common) for lab, v in per.items() if v}
    for lab, dd in dropped.items():
        if dd:
            print(f"   note: {lab} has extra scenes not scored by every label: {dd}")
    print(f"   {'model':44s} {'ATE':>9s} {'RPE_rot':>9s} {'RPE_t':>9s}")
    for name, lab in LABELS:
        if not per[lab]:
            print(f"   {name:44s}   (no results)")
            continue
        a = np.array([per[lab][s] for s in sorted(common)])
        m = a.mean(0)
        agg[(lab, d)] = m
        print(f"   {name:44s} {m[0]:9.4f} {m[1]:9.3f} {m[2]:9.4f}")

print()
print("=" * 100)
print("VIDEO DEPTH   scale&shift aligned, identical scenes and frame counts.")
print("              Lower Abs Rel / RMSE better, higher delta better.")
print("=" * 100)
dp = {}
for d in ("sintel", "bonn"):
    print(f"\n-- {d}")
    print(f"   {'model':44s} {'Abs Rel':>9s} {'RMSE':>9s} {'d<1.25':>9s}")
    for name, lab in LABELS:
        v = depth(lab, d)
        if v is None:
            print(f"   {name:44s}   (not run)")
            continue
        dp[(lab, d)] = v
        print(f"   {name:44s} {v[0]:9.4f} {v[1]:9.4f} {v[2]:9.4f}")

print()
print("=" * 100)
print("LoopCUT3R vs the CORRECT CUT3R baseline      (>1.00x = LoopCUT3R worse)")
print("=" * 100)
C = "CUT3R_fixed"
L = "LoopCUT3R_stage3b_rerun"
for d in ("scannet", "tum", "sintel"):
    if (C, d) in agg and (L, d) in agg:
        c, l = agg[(C, d)], agg[(L, d)]
        print(f"  pose  {d:9s} ATE {l[0]/c[0]:5.2f}x    RPE_rot {l[1]/c[1]:5.2f}x    RPE_t {l[2]/c[2]:5.2f}x")
Ld = "LoopCUT3R_recurrent_stage3b_checkpoint_last"
for d in ("sintel", "bonn"):
    if (C, d) in dp and (Ld, d) in dp:
        print(f"  depth {d:9s} Abs Rel {dp[Ld][d][0]/dp[(C,d)][0]:5.2f}x    RMSE {dp[(Ld,d)][1]/dp[(C,d)][1]:5.2f}x"
              if False else
              f"  depth {d:9s} Abs Rel {dp[(Ld,d)][0]/dp[(C,d)][0]:5.2f}x    RMSE {dp[(Ld,d)][1]/dp[(C,d)][1]:5.2f}x")
print()
print("  reconstruction harness, each model in its own tree, for comparison:")
print("    GHOST 002 full sequence        2.79x")
print("    GHOST isolated 2-frame pairs   3.88x")
