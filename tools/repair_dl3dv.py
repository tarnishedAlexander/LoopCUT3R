#!/usr/bin/env python3
"""
Repair DL3DV processed data with missing depth (or other dense assets).

Default: remove RGB/cam/mask files that have no matching depth .npy
(so scenes stay usable without re-downloading terabytes).

Use --remove-scene if a scene has too many gaps (e.g. >50% missing depth).
"""
import argparse
import os
import os.path as osp
import shutil


def repair_scene(dense: str, rgb_dir: str, dry_run: bool, remove_scene_threshold: float):
    rgbs = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    orphan = []
    for rgb in rgbs:
        base = rgb[:-4]
        depth_path = osp.join(dense, "depth", base + ".npy")
        if not osp.isfile(depth_path):
            orphan.append(rgb)

    if not orphan:
        return {"removed_files": 0, "removed_scene": False}

    if len(orphan) / max(len(rgbs), 1) >= remove_scene_threshold:
        scene_dir = osp.dirname(dense)
        if dry_run:
            print(f"[dry-run] would remove entire scene: {scene_dir} ({len(orphan)}/{len(rgbs)} missing depth)")
        else:
            shutil.rmtree(scene_dir)
            print(f"Removed scene (too incomplete): {scene_dir}")
        return {"removed_files": len(rgbs), "removed_scene": True}

    removed = 0
    for rgb in orphan:
        base = rgb[:-4]
        paths = [
            osp.join(rgb_dir, rgb),
            osp.join(dense, "cam", base + ".npz"),
            osp.join(dense, "sky_mask", rgb),
            osp.join(dense, "outlier_mask", rgb),
        ]
        for p in paths:
            if osp.isfile(p):
                if dry_run:
                    print(f"[dry-run] would delete {p}")
                else:
                    os.remove(p)
                removed += 1
    return {"removed_files": removed, "removed_scene": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default="/app/recurrentTransformer/dataset/disk12tb/dl3dv/processed_dl3dv",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--remove-scene-threshold",
        type=float,
        default=0.5,
        help="If fraction of frames missing depth >= this, delete whole scene (default 0.5)",
    )
    args = p.parse_args()

    total_files = 0
    scenes_touched = 0
    scenes_removed = 0

    for tier in sorted(os.listdir(args.root)):
        tier_dir = osp.join(args.root, tier)
        if not osp.isdir(tier_dir):
            continue
        for scene_hash in os.listdir(tier_dir):
            dense = osp.join(tier_dir, scene_hash, "dense")
            rgb_dir = osp.join(dense, "rgb")
            if not osp.isdir(rgb_dir):
                continue
            out = repair_scene(dense, rgb_dir, args.dry_run, args.remove_scene_threshold)
            if out["removed_files"] or out["removed_scene"]:
                scenes_touched += 1
                total_files += out["removed_files"]
                if out["removed_scene"]:
                    scenes_removed += 1

    mode = "dry-run" if args.dry_run else "done"
    print(
        f"{mode}: touched {scenes_touched} scene(s), "
        f"removed {scenes_removed} whole scene(s), "
        f"{total_files} file(s) pruned"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
