#!/usr/bin/env python3
"""Audit DL3DV processed folders for missing dense assets."""
import argparse
import os
import os.path as osp
from collections import defaultdict


def iter_scenes(root: str):
    for tier in sorted(os.listdir(root)):
        tier_dir = osp.join(root, tier)
        if not osp.isdir(tier_dir):
            continue
        for scene_hash in os.listdir(tier_dir):
            dense = osp.join(tier_dir, scene_hash, "dense")
            rgb_dir = osp.join(dense, "rgb")
            if osp.isdir(rgb_dir):
                yield osp.join(tier, scene_hash), dense, rgb_dir


def audit_scene(dense: str, rgb_dir: str):
    missing_by_kind = defaultdict(list)
    for rgb in sorted(os.listdir(rgb_dir)):
        if not rgb.endswith(".png"):
            continue
        base = rgb[:-4]
        checks = {
            "depth": osp.join(dense, "depth", base + ".npy"),
            "cam": osp.join(dense, "cam", base + ".npz"),
            "sky": osp.join(dense, "sky_mask", rgb),
            "outlier": osp.join(dense, "outlier_mask", rgb),
        }
        for kind, path in checks.items():
            if not osp.isfile(path):
                missing_by_kind[kind].append(rgb)
    return missing_by_kind


def audit(root: str, max_examples: int = 20):
    examples = []
    scenes_ok = 0
    scenes_bad = []
    total_missing = 0

    for scene_id, dense, rgb_dir in iter_scenes(root):
        n_rgb = len([f for f in os.listdir(rgb_dir) if f.endswith(".png")])
        missing = audit_scene(dense, rgb_dir)
        n_miss = sum(len(v) for v in missing.values())
        if n_miss == 0:
            scenes_ok += 1
            continue
        scenes_bad.append((scene_id, n_rgb, missing))
        total_missing += n_miss
        for kind, frames in missing.items():
            for rgb in frames:
                if len(examples) < max_examples:
                    examples.append(
                        (scene_id, rgb, kind, osp.join(dense, kind, rgb))
                    )

    scenes_bad.sort(key=lambda x: sum(len(v) for v in x[2].values()), reverse=True)
    return {
        "scenes_ok": scenes_ok,
        "scenes_bad": scenes_bad,
        "total_missing": total_missing,
        "examples": examples,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        default="/app/recurrentTransformer/dataset/disk12tb/dl3dv/processed_dl3dv",
    )
    p.add_argument("--max-examples", type=int, default=20)
    p.add_argument(
        "--full",
        action="store_true",
        help="Print per-scene summary for all broken scenes",
    )
    args = p.parse_args()

    if not osp.isdir(args.root):
        print(f"ERROR: root does not exist: {args.root}")
        return 1

    stats = audit(args.root, args.max_examples)
    print(f"Root: {args.root}")
    print(f"Scenes OK: {stats['scenes_ok']}")
    print(f"Scenes with gaps: {len(stats['scenes_bad'])}")
    print(f"Total missing files: {stats['total_missing']}")

    if stats["scenes_bad"]:
        print("\nWorst scenes:")
        for scene_id, n_rgb, missing in stats["scenes_bad"][:10]:
            parts = ", ".join(f"{k}={len(v)}" for k, v in sorted(missing.items()))
            print(f"  {scene_id}: {n_rgb} rgb, missing ({parts})")

        if args.full:
            print("\nAll broken scenes:")
            for scene_id, n_rgb, missing in stats["scenes_bad"]:
                parts = ", ".join(f"{k}={len(v)}" for k, v in sorted(missing.items()))
                print(f"  {scene_id}: {parts}")

        if stats["examples"]:
            print(f"\nFirst {len(stats['examples'])} example(s):")
            for scene, rgb, kind, path in stats["examples"]:
                print(f"  {scene} / {rgb} missing {kind}: {path}")
    else:
        print("OK: every frame has rgb + depth + cam + masks.")

    return 0 if not stats["scenes_bad"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
