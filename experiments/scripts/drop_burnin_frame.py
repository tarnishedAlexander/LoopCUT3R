#!/usr/bin/env python3
"""Drop frame 0 from a dump, which is exactly what discarding the burn-in
prediction means.  Per-frame arrays are sliced; scalars are carried over."""
import numpy as np, pathlib, sys

PERFRAME = {"pts_cam", "pts_world", "pts_posed", "conf_cam", "conf_world",
            "pose_c2w", "rgb", "img_paths"}
for arg in sys.argv[1:]:
    src, dst = arg.split("=")
    d = np.load(src, allow_pickle=True)
    n_before = len(d["pts_cam"])
    out = {}
    for k in d.files:
        v = d[k]
        out[k] = v[1:] if (k in PERFRAME and v.ndim >= 1 and len(v) == n_before) else v
    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    print(f"  {pathlib.Path(src).name}  {n_before} -> {len(out['pts_cam'])}  "
          f"=> {pathlib.Path(dst).name}")
