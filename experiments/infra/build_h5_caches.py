import sys
import os
import time

# Add LoopCUT3R/src to sys.path (this file now lives in LoopCUT3R/tools/h5/)
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
sys.path.append(src_dir)

from dust3r.datasets.utils.h5_vfs import H5VFSRegistry

h5_files = [
    "/app/recurrentTransformer/dataset/disk12tb/processed_dl3dv.h5",
    "/app/recurrentTransformer/dataset/disk12tb/processed_scannet.h5",
    "/app/recurrentTransformer/dataset/disk12tb/processed_TartanAir.h5"
]

print("Starting cache pre-building for H5 datasets inside docker container...")
for h5_path in h5_files:
    print(f"\n==================================================")
    print(f"Processing: {h5_path}")
    print(f"==================================================")
    if not os.path.exists(h5_path):
        print(f"WARNING: File {h5_path} not found in this environment context.")
        continue
    
    start = time.time()
    # Build the cache (this will visit items and write the .cache.pkl)
    cache = H5VFSRegistry._build_cache(h5_path)
    end = time.time()
    
    if cache is not None:
        print(f"SUCCESS: Built and saved cache for {h5_path} in {end-start:.2f} seconds.")
        print(f"Total files in cache: {len(cache['files'])}")
        print(f"Total directories in cache: {len(cache['dirs'])}")
    else:
        print(f"FAILED: Could not build cache for {h5_path}")
