#!/usr/bin/env python3
"""
Measure the distribution of inter-frame camera baseline the training sampler
actually produces, at one or more num_views settings.

The A/B showed the pose defect scales with camera motion: LoopCUT3R holds up on
examples/004 (steps ~0.04 of scene radius) and degrades on 001/002 (0.15-0.18,
peaks past 1.0). This answers whether the training distribution ever contains
the motions where it fails, and whether num_views=32 shifts it smaller than
num_views=4 -- which is the proposed reason the 32-view stages made the
high-motion clips worse.

Baselines are divided by the sequence's own scene radius, exactly as in
compare_recon.py, so the numbers are directly comparable to the A/B tables.

    python sampler_baseline.py --config config/recurrent_stage4b_32v.yaml --views 4 32
"""
import argparse
import os
import sys

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")


def relax_limits():
    """The h5/npz layer keeps file handles open, and this probe pulls samples
    far faster than training does, so it exhausts the fd limit at 32 views.
    Raise the soft limit to the hard one and stop torch sharing tensors by fd
    (otherwise the loader dies with 'received 0 items of ancdata')."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f"[limits] RLIMIT_NOFILE {soft} -> {hard}")
    except Exception as e:
        print(f"[limits] could not raise fd limit: {e}")
    try:
        import torch.multiprocessing as mp
        mp.set_sharing_strategy("file_system")
    except Exception as e:
        print(f"[limits] could not set sharing strategy: {e}")


# Measured from the demo clips (CUT3R poses), for reference.
REFERENCE = {"004 (model copes)": 0.04, "001 (degrades)": 0.15, "002 (worst)": 0.18}


def setup_path():
    """The repo's internals do `import dust3r`, so src/ must be on sys.path.
    Same thing add_ckpt_path.add_path_to_dust3r does for demo.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "src"), here):
        if p not in sys.path:
            sys.path.insert(0, p)


def build_loader(cfg_path, num_views, batch_workers, seed, datasets=None):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(cfg_path)
    OmegaConf.update(cfg, "num_views", num_views, force_add=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    if datasets:
        missing = [d for d in datasets if d not in cfg]
        if missing:
            raise SystemExit(f"not in config: {missing}")
        spec = " + ".join(f"10000 @ {cfg[d]}" for d in datasets)
    else:
        spec = cfg["train_dataset"]

    from dust3r.datasets import get_data_loader

    return get_data_loader(
        spec,
        batch_size=1,
        num_workers=batch_workers,
        shuffle=True,
        drop_last=True,
        pin_mem=False,
        accelerator=None,
    )


def sequence_steps(views, stats):
    """Consecutive camera baselines for one sequence, / scene radius.

    A single view with a missing pose (the base dataset writes nan) must not
    discard the sequence: at num_views=32 that throws away almost everything.
    Keep every consecutive pair whose two poses are both finite.
    """
    pos, ok, pts = [], [], []
    for v in views:
        cp = np.asarray(v["camera_pose"])[0]
        good = bool(np.isfinite(cp).all())
        ok.append(good)
        pos.append(cp[:3, 3] if good else np.zeros(3))
        p = np.asarray(v["pts3d"])[0]
        m = np.asarray(v["valid_mask"])[0].astype(bool)
        if m.sum():
            pts.append(p[m])
    stats["views"] += len(ok)
    stats["bad_pose"] += len(ok) - sum(ok)
    if not pts:
        stats["no_pts"] += 1
        return None
    allp = np.concatenate(pts, 0)
    if len(allp) > 200000:
        allp = allp[:: len(allp) // 200000 + 1]
    R = float(np.median(np.linalg.norm(allp - allp.mean(0), axis=-1)))
    if not np.isfinite(R) or R <= 0:
        stats["bad_radius"] += 1
        return None
    pos = np.stack(pos, 0)
    pair = np.array(ok[:-1]) & np.array(ok[1:])
    if not pair.any():
        stats["no_pair"] += 1
        return None
    d = np.linalg.norm(np.diff(pos, axis=0), axis=-1)[pair] / R
    d = d[np.isfinite(d)]
    if d.size == 0:
        stats["no_pair"] += 1
        return None
    return d


def set_epoch(loader, epoch):
    """The batched sampler refuses to iterate until an epoch is set.
    Mirrors what train.py does before each epoch."""
    ds = getattr(loader, "dataset", None)
    if ds is not None and hasattr(ds, "set_epoch"):
        ds.set_epoch(epoch)
    bs = getattr(loader, "batch_sampler", None)
    inner = getattr(bs, "batch_sampler", None)
    for obj in (inner, bs):
        if obj is not None and hasattr(obj, "set_epoch"):
            obj.set_epoch(epoch)
            break


def run(cfg_path, num_views, n_seq, workers, seed, datasets=None):
    loader = build_loader(cfg_path, num_views, workers, seed, datasets)
    set_epoch(loader, seed)
    stats = dict(views=0, bad_pose=0, no_pts=0, bad_radius=0, no_pair=0)
    steps, seq_med, used, skipped = [], [], 0, 0
    for batch in loader:
        s = sequence_steps(batch, stats)
        if s is None:
            skipped += 1
        else:
            steps.append(s); seq_med.append(float(np.median(s))); used += 1
        if used % 25 == 0 and used and s is not None:
            print(f"    {used}/{n_seq}", flush=True)
        if used >= n_seq:
            break
    print(f"    views={stats['views']} bad_pose={stats['bad_pose']} "
          f"no_pts={stats['no_pts']} bad_radius={stats['bad_radius']} "
          f"no_pair={stats['no_pair']} skipped={skipped}", flush=True)
    if not steps:
        print(f"NO USABLE SEQUENCES at num_views={num_views}; see counters above")
        return np.array([np.nan]), np.array([np.nan]), 0, skipped
    return np.concatenate(steps), np.array(seq_med), used, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/recurrent_stage4b_32v.yaml")
    ap.add_argument("--views", type=int, nargs="+", default=[4, 32])
    ap.add_argument("--sequences", type=int, default=150)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="config keys to use instead of the full train_dataset mix, "
                         "e.g. dataset1 dataset3 dataset5")
    a = ap.parse_args()

    setup_path()
    relax_limits()

    res = {}
    for nv in a.views:
        print(f"sampling {a.sequences} sequences at num_views={nv} ...", flush=True)
        res[nv] = run(a.config, nv, a.sequences, a.workers, a.seed, a.datasets)

    q = [10, 25, 50, 75, 90, 99]
    w = 26
    print("\n" + "=" * (w + 14 * len(a.views)))
    print("inter-frame camera baseline / scene radius".ljust(w)
          + "".join(f"num_views={nv}".rjust(14) for nv in a.views))
    print("-" * (w + 14 * len(a.views)))
    for p in q:
        print(f"p{p:<25}" + "".join(f"{np.percentile(res[nv][0], p):>14.4f}" for nv in a.views))
    print(f"{'mean':<26}" + "".join(f"{res[nv][0].mean():>14.4f}" for nv in a.views))
    print(f"{'sequences used':<26}" + "".join(f"{res[nv][2]:>14d}" for nv in a.views))
    print(f"{'sequences skipped':<26}" + "".join(f"{res[nv][3]:>14d}" for nv in a.views))
    print("-" * (w + 14 * len(a.views)))
    for label, v in REFERENCE.items():
        print(f"{'frac >= ' + label:<26}"
              + "".join(f"{(res[nv][0] >= v).mean():>13.1%}" for nv in a.views))
    print("=" * (w + 14 * len(a.views)))
    print("\nreference: median step/R measured from the demo clips (CUT3R poses)")
    for label, v in REFERENCE.items():
        print(f"  {label:<22}{v:.2f}")
    if len(a.views) == 2:
        lo, hi = a.views
        r = np.median(res[hi][0]) / np.median(res[lo][0])
        print(f"\nmedian baseline at num_views={hi} is {r:.2f}x the value at num_views={lo}")
        print("  < 1.00 supports the hypothesis: more views -> finer sampling -> smaller motions")


if __name__ == "__main__":
    main()
