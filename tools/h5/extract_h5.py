#!/usr/bin/env python3
"""
Extract H5VFS archive back to raw files on disk.

Usage:
    python extract_h5.py <h5_file> <output_dir>

Resumes automatically — already-existing files are skipped.
Run inside the container: docker exec -it <container> python ...
"""
import h5py
import numpy as np
import os
import sys
import pickle
from pathlib import Path


def extract_h5(h5_path: str, out_dir: str):
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written = skipped = errors = 0

    def process(name, obj):
        nonlocal written, skipped, errors

        # Skip children of .npz groups — handled atomically when we hit the group itself
        for part in Path(name).parts[:-1]:
            if part.endswith('.npz'):
                return

        out_path = out_root / name

        # ── Directory ────────────────────────────────────────────────────────
        if isinstance(obj, h5py.Group):
            if name.endswith('.npz'):
                # .npz files are stored as H5 groups (keys = array names)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if out_path.exists():
                    skipped += 1
                    return
                try:
                    data = {}
                    for k in obj.keys():
                        ds = obj[k]
                        if ds.dtype.kind == 'V':
                            arr = pickle.loads(ds[()].tobytes())
                        else:
                            arr = ds[()]  # [()] handles both scalar (ndim=0) and n-dim; [:] fails on scalars
                            if arr.dtype == np.float16:
                                arr = arr.astype(np.float32)
                            elif arr.dtype.kind == 'S':  # H5 stores strings as fixed-length bytes; restore to unicode
                                arr = arr.astype(str)
                        data[k] = arr
                    np.savez_compressed(str(out_path), **data)
                    written += 1
                except Exception as e:
                    print(f"[ERROR] {name}: {e}", flush=True)
                    errors += 1
            else:
                out_path.mkdir(parents=True, exist_ok=True)
            return

        if not isinstance(obj, h5py.Dataset):
            return

        # ── File ─────────────────────────────────────────────────────────────
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            skipped += 1
            return

        try:
            if obj.dtype.kind == 'V':
                # Void bytes — raw binary / pickled npy
                out_path.write_bytes(obj[()].tobytes())

            elif obj.ndim == 0:
                # Scalar blob
                out_path.write_bytes(obj[()].tobytes())

            elif obj.ndim == 1:
                # 1-D byte array — original JPEG / PNG bytes stored verbatim
                out_path.write_bytes(obj[:].tobytes())

            else:
                # Multi-dimensional array (depth map, mask, pre-decoded image)
                arr = obj[:]
                if arr.dtype == np.float16:
                    arr = arr.astype(np.float32)

                low = name.lower()
                if low.endswith(('.jpg', '.jpeg', '.png')):
                    import cv2
                    ext = '.' + low.rsplit('.', 1)[-1]
                    if arr.max() > 255:
                        # uint16 depth (cm values 0-65500) — imencode silently downcasts,
                        # use imwrite directly which correctly writes 16-bit PNG
                        encode_arr = np.clip(np.round(arr), 0, 65535).astype(np.uint16)
                        cv2.imwrite(str(out_path), encode_arr)
                    else:
                        if arr.max() <= 1.0 and arr.dtype in (np.float32, np.float64):
                            encode_arr = (arr * 255).astype(np.uint8)
                        else:
                            encode_arr = arr.astype(np.uint8)
                        params = [cv2.IMWRITE_JPEG_QUALITY, 95] if 'jp' in ext else []
                        ok, buf = cv2.imencode(ext, encode_arr, params)
                        if ok:
                            out_path.write_bytes(buf.tobytes())
                        else:
                            np.save(str(out_path) + '.npy', arr)
                else:
                    np.save(str(out_path), arr)

            written += 1

        except Exception as e:
            print(f"[ERROR] {name}: {e}", flush=True)
            errors += 1

        total = written + skipped
        if total % 5000 == 0 and total > 0:
            print(f"  progress: written={written}  skipped={skipped}  errors={errors}",
                  flush=True)

    print(f"[h5_extract] {h5_path}  →  {out_dir}")
    with h5py.File(h5_path, 'r', rdcc_nbytes=256 * 1024 * 1024,
                   rdcc_nslots=100003, rdcc_w0=0.0) as f:
        f.visititems(process)

    print(f"[h5_extract] done — written={written}  skipped={skipped}  errors={errors}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_h5.py <h5_file> <output_dir>")
        sys.exit(1)
    extract_h5(sys.argv[1], sys.argv[2])
