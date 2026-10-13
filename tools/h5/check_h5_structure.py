"""Check H5 structure for ARKitScenesHighRes.
Run inside container: python tools/h5/check_h5_structure.py
"""
import h5py, sys

H5_PATH = "/app/recurrentTransformer/dataset/disk14tb/processed_arkitscenes_highres.h5"

with h5py.File(H5_PATH, "r") as f:
    top = list(f.keys())
    print("Top-level keys:", top)
    for k in top[:3]:
        sub = list(f[k].keys()) if isinstance(f[k], h5py.Group) else ["<dataset>"]
        print(f"  f[{k!r}] → {sub[:5]}{'...' if len(sub) > 5 else ''}")
        if sub and sub[0] != "<dataset>":
            k2 = sub[0]
            sub2 = list(f[k][k2].keys()) if isinstance(f[k][k2], h5py.Group) else ["<dataset>"]
            print(f"    f[{k!r}][{k2!r}] → {sub2[:5]}{'...' if len(sub2) > 5 else ''}")
