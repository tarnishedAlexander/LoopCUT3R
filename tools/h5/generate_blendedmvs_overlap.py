#!/usr/bin/env python3
"""
Generate new_overlap.h5 for a preprocessed BlendedMVS directory.

The BlendedMVS_Multi loader expects new_overlap.h5 with per-scene sparse
overlap matrices. When the precomputed pairs file is unavailable (e.g. for
BlendedMVS2 re-processed from raw), this script generates a uniform-overlap
file so all image pairs are treated as valid neighbors.

Usage (inside container):
    python scripts/generate_blendedmvs_overlap.py \
        /app/recurrentTransformer/dataset/datasetfaster/processed_BlendedMVS2_full
"""
import os
import sys
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm


def generate_overlap(processed_dir: str):
    root = Path(processed_dir)
    out_path = root / "new_overlap.h5"

    if out_path.exists():
        print(f"[overlap] {out_path} already exists — skipping")
        return

    # Scene dirs are 24-char hex strings
    scenes = sorted([d for d in root.iterdir() if d.is_dir() and len(d.name) == 24])
    print(f"[overlap] Found {len(scenes)} scenes in {root}")

    with h5py.File(str(out_path), "w") as f:
        for scene_dir in tqdm(scenes):
            basenames = sorted([
                p.stem for p in scene_dir.iterdir()
                if p.suffix == ".jpg"
            ])
            if not basenames:
                continue

            n = len(basenames)
            # Uniform overlap: all pairs score 1.0 (excluding self)
            rows, cols, vals = [], [], []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        rows.append(i)
                        cols.append(j)
                        vals.append(1.0)

            grp = f.create_group(scene_dir.name)
            grp.create_dataset("basenames", data=np.array(basenames, dtype="S"))
            grp.create_dataset("indices", data=np.array([rows, cols], dtype=np.int32))
            grp.create_dataset("values", data=np.array(vals, dtype=np.float32))
            grp.attrs["shape"] = (n, n)

    print(f"[overlap] Saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_blendedmvs_overlap.py <processed_dir>")
        sys.exit(1)
    generate_overlap(sys.argv[1])
