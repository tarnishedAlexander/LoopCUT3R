#!/usr/bin/env python3
"""
Camera pose table (ATE, RPE trans, RPE rot) with ONE protocol for all labels.

Re-scores the saved `pred_traj.txt` files instead of averaging what each run
reported: every scene is cut to its first K frames, with K the minimum over the
compared labels, and only scenes that every label scored are averaged.  The
models are causal, so cutting the prediction equals running the shorter
sequence.  This matters on Sintel, where a full-sequence run and a capped run
would otherwise be compared on different frames.

Per-scene values (for figures and paired tests) are written to --out.
Run from the repository root (the dataset paths in eval/ are relative).

    python tools/eval/pose_table.py --out docs/results/pose_by_scene.json \
        CUT3R LoopCUT3R
"""
import argparse, copy, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from eval.relpose.metadata import dataset_metadata
from eval.relpose.evo_utils import load_traj, make_traj
from evo.core import sync
from evo.core.metrics import PoseRelation, Unit
from evo.main_ape import ape
from evo.main_rpe import rpe

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("labels", nargs="+")
ap.add_argument("--datasets", nargs="+", default=["scannet", "tum", "sintel"])
ap.add_argument("--out", default=None)
ap.add_argument("--results-dir", default=os.path.join(ROOT, "eval_results"))
args = ap.parse_args()
root = os.path.join(args.results_dir, "relpose")


def score(data, label, seq, K):
    md = dataset_metadata[data]
    a = np.loadtxt(f"{root}/{data}_{label}/{seq}/pred_traj.txt")[:K]
    pred = make_traj([a[:, 1:], a[:, 0]])
    gt_file = md["gt_traj_func"](md["img_path"], md.get("anno_path"), seq)
    g = load_traj(gt_traj_file=gt_file) if data == "sintel" else \
        load_traj(gt_traj_file=gt_file, traj_format=md.get("traj_format"))
    gt = make_traj([g[0][:max(K, 0) if data != "sintel" else None], g[1][:max(K, 0) if data != "sintel" else None]])
    if pred.timestamps.shape[0] == gt.timestamps.shape[0]:
        pred.timestamps = gt.timestamps
    else:
        gt, pred = sync.associate_trajectories(gt, pred)
    k = dict(est_name="t", align=True, correct_scale=True)
    ate = ape(copy.deepcopy(gt), copy.deepcopy(pred), pose_relation=PoseRelation.translation_part, **k).stats["rmse"]
    r = dict(delta=1, delta_unit=Unit.frames, rel_delta_tol=0.01, all_pairs=True, **k)
    rt = rpe(copy.deepcopy(gt), copy.deepcopy(pred), pose_relation=PoseRelation.translation_part, **r).stats["rmse"]
    rr = rpe(copy.deepcopy(gt), copy.deepcopy(pred), pose_relation=PoseRelation.rotation_angle_deg, **r).stats["rmse"]
    return dict(ate=float(ate), rpe_t=float(rt), rpe_r=float(rr), frames=int(len(a)))


res = {}
for data in args.datasets:
    seqs = sorted(s for s in os.listdir(f"{root}/{data}_{args.labels[0]}")
                  if os.path.isdir(f"{root}/{data}_{args.labels[0]}/{s}"))
    common = [s for s in seqs if all(os.path.exists(f"{root}/{data}_{l}/{s}/pred_traj.txt") for l in args.labels)]
    table, dropped, nfr, errors = {l: {} for l in args.labels}, [], [], []
    for s in common:
        K = min(len(np.loadtxt(f"{root}/{data}_{l}/{s}/pred_traj.txt")) for l in args.labels)
        try:
            row = {l: score(data, l, s, K) for l in args.labels}
        except Exception as e:  # missing ground truth, unreadable trajectory
            dropped.append(s); errors.append(f"{s}: {e}"); continue
        nfr.append(K)
        for l in args.labels:
            table[l][s] = row[l]
    n = len(nfr)
    if n == 0:
        print(f"-- {data}: no scene could be scored" + (f" ({errors[0]})" if errors else ""))
        continue
    print(f"-- {data}: {n} scenes, {sum(nfr)} frames (min {min(nfr)}, max {max(nfr)})" + (f"  dropped: {dropped}" if dropped else ""))
    print(f"   {'label':<34}{'ATE':>9}{'RPE_t':>9}{'RPE_r':>9}")
    for l in args.labels:
        m = {k: np.mean([v[k] for v in table[l].values()]) for k in ("ate", "rpe_t", "rpe_r")}
        print(f"   {l:<34}{m['ate']:>9.4f}{m['rpe_t']:>9.4f}{m['rpe_r']:>9.3f}")
    res[data] = table

if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("wrote", args.out)
