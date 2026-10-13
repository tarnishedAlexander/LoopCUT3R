#!/usr/bin/env python3
"""
Unified HDF5 Dataset Packer for LoopCUT3R.
Recursively compresses and packs any preprocessed dataset (RGB, Depths, Poses, Masks)
into a single process-safe H5 file.
Uses a multi-process producer-consumer pattern for high throughput.
"""
import os
import sys
import argparse
import time
import queue
from multiprocessing import Process, Queue, cpu_count
import h5py
import cv2
import numpy as np
from tqdm import tqdm

# Supported formats
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
DEPTH_EXTS = ('.npy', '.png')
NPZ_EXTS = ('.npz',)

def compress_worker(src_dir, task_queue, result_queue, lossless_rgb=False):
    """Worker process that reads and compresses files from disk."""
    while True:
        try:
            rel_path = task_queue.get(timeout=1.0)
        except queue.Empty:
            break
        
        if rel_path is None:
            break
        
        full_path = os.path.join(src_dir, rel_path)
        if not os.path.exists(full_path):
            continue
        try:
            data_type = None
            payload = None
            lower_path = rel_path.lower()
            
            # 1. Camera parameters (NPZ files)
            if lower_path.endswith(NPZ_EXTS):
                data_type = 'npz'
                with np.load(full_path, allow_pickle=True) as data:
                    payload = {}
                    for k in data.files:
                        arr = data[k]
                        if arr.dtype == np.float64:
                            arr = arr.astype(np.float32)
                        elif arr.dtype.kind == 'U':
                            # Convert unicode array to bytes array for HDF5 compatibility
                            arr = np.char.encode(arr, 'utf-8')
                        elif arr.dtype.kind == 'O':
                            # Object array, pickle it and store as void bytes
                            import pickle
                            arr = np.void(pickle.dumps(arr))
                        payload[k] = arr

            # 2. Depth maps (.npy files or .png in a depth folder)
            elif lower_path.endswith('.npy') or ('depth' in lower_path and lower_path.endswith('.png')):
                if lower_path.endswith('.npy'):
                    try:
                        arr = np.load(full_path, allow_pickle=True)
                        if np.issubdtype(arr.dtype, np.number) and arr.dtype.kind != 'O':
                            data_type = 'depth'
                            payload = arr.astype(np.float32)
                        else:
                            # It is string/object metadata. Store as raw bytes.
                            data_type = 'raw_npy'
                            with open(full_path, 'rb') as f:
                                payload = f.read()
                    except Exception:
                        data_type = 'raw_npy'
                        with open(full_path, 'rb') as f:
                            payload = f.read()
                else:
                    data_type = 'depth'
                    depth = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                    if depth is None:
                        raise ValueError(f"Could not load depth image: {full_path}")
                    payload = depth.astype(np.float32)

            # 3. Masks (sky_mask, outlier_mask, masks folders)
            elif 'mask' in lower_path and lower_path.endswith(IMAGE_EXTS):
                data_type = 'mask'
                mask = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if mask is None:
                    raise ValueError(f"Could not load mask: {full_path}")
                success, buf = cv2.imencode('.png', mask)
                if not success:
                    raise ValueError(f"Failed to encode mask: {rel_path}")
                payload = buf

            # 4. RGB Images
            elif lower_path.endswith(IMAGE_EXTS):
                data_type = 'rgb'
                img = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Could not load image: {full_path}")
                
                # Note: We keep BGR format (OpenCV default) for compression.
                # cv2.imencode automatically handles BGR order. The custom VFS cv2.imread wrapper 
                # will return BGR, matching the original cv2.imread output for downstream processing.
                if lossless_rgb and lower_path.endswith('.png'):
                    success, buf = cv2.imencode('.png', img)
                else:
                    success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                if not success:
                    raise ValueError(f"Failed to encode image: {rel_path}")
                payload = buf
            else:
                data_type = 'raw'
                with open(full_path, 'rb') as f:
                    payload = f.read()

            result_queue.put((rel_path, data_type, payload))
        except Exception as e:
            print(f"\n[Worker Error] Failed to process {rel_path}: {e}")
            result_queue.put((rel_path, 'error', str(e)))

def main():
    parser = argparse.ArgumentParser(description="Pack preprocessed datasets into compressed H5 containers.")
    parser.add_argument('--src', type=str, required=True, help="Source directory containing preprocessed dataset.")
    parser.add_argument('--dest', type=str, required=True, help="Destination H5 file path.")
    parser.add_argument('--max-scenes', type=int, default=None, help="Max scenes to process (for testing).")
    parser.add_argument('--max-frames', type=int, default=None, help="Max frames per scene to process (for testing).")
    parser.add_argument('--workers', type=int, default=max(1, cpu_count() - 2), help="Number of compression processes.")
    parser.add_argument('--lossless-rgb', action='store_true', help="Use lossless PNG for PNG source images.")
    args = parser.parse_args()

    src_dir = os.path.abspath(args.src)
    dest_path = os.path.abspath(args.dest)

    if not os.path.isdir(src_dir):
        print(f"Error: Source directory does not exist: {src_dir}")
        sys.exit(1)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Scanning directory: {src_dir} ...")
    
    all_files = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, src_dir)
            all_files.append(rel_path)
            
    print(f"Found {len(all_files)} total files.")

    if args.max_scenes is not None or args.max_frames is not None:
        print(f"Applying filters (max_scenes={args.max_scenes}, max_frames={args.max_frames})...")
        scene_files = {}
        for f in all_files:
            parts = f.split(os.sep)
            if len(parts) > 1:
                scene_key = parts[0] if parts[0] not in ('1K', '2K', '3K', '4K', '5K', '6K', '7K') else os.path.join(parts[0], parts[1])
                if scene_key not in scene_files:
                    scene_files[scene_key] = []
                scene_files[scene_key].append(f)
            else:
                if '' not in scene_files:
                    scene_files[''] = []
                scene_files[''].append(f)

        filtered_files = []
        if '' in scene_files:
            filtered_files.extend(scene_files[''])

        scenes_to_keep = [k for k in scene_files.keys() if k != '']
        scenes_to_keep.sort()
        if args.max_scenes is not None:
            scenes_to_keep = scenes_to_keep[:args.max_scenes]

        for s in scenes_to_keep:
            s_files = scene_files[s]
            if args.max_frames is not None:
                basenames = set()
                for f in s_files:
                    if f.lower().endswith(IMAGE_EXTS) and any(x in f.lower() for x in ('rgb', 'images', 'depth')):
                        basename = os.path.basename(f).split('.')[0]
                        basenames.add(basename)
                
                sorted_basenames = sorted(list(basenames))[:args.max_frames]
                
                for f in s_files:
                    basename = os.path.basename(f).split('.')[0]
                    # Keep metadata unconditionally so scene doesn't break
                    if basename in sorted_basenames or not any(x in f.lower() for x in ('rgb', 'depth', 'images', 'masks')):
                        filtered_files.append(f)
            else:
                filtered_files.extend(s_files)
                
        all_files = filtered_files
        print(f"Filtered to {len(all_files)} files.")

    # Filter out already packed scenes if appending
    if os.path.exists(dest_path):
        print(f"Destination file {dest_path} already exists. Checking for already packed scenes...")
        try:
            with h5py.File(dest_path, 'r') as h5_file:
                existing_scenes = set()
                for cat in h5_file.keys():
                    if isinstance(h5_file[cat], h5py.Group):
                        for scene in h5_file[cat].keys():
                            existing_scenes.add(os.path.join(cat, scene))
                
                print(f"Found {len(existing_scenes)} completed scenes in H5.")
                
                filtered_files = []
                skipped_scenes = set()
                for f in all_files:
                    parts = f.split(os.sep)
                    if len(parts) > 1:
                        scene_key = parts[0] if parts[0] not in ('1K', '2K', '3K', '4K', '5K', '6K', '7K') else os.path.join(parts[0], parts[1])
                        if scene_key in existing_scenes:
                            skipped_scenes.add(scene_key)
                            continue
                    filtered_files.append(f)
                
                print(f"Skipped {len(skipped_scenes)} already packed scenes. Remaining files: {len(filtered_files)}")
                all_files = filtered_files
        except Exception as e:
            print(f"Error checking existing H5 file: {e}. Starting fresh.")

    task_queue = Queue()
    result_queue = Queue(maxsize=1000)

    for f in all_files:
        task_queue.put(f)
    for _ in range(args.workers):
        task_queue.put(None)

    print(f"Launching {args.workers} compression workers...")
    workers = []
    for _ in range(args.workers):
        p = Process(target=compress_worker, args=(src_dir, task_queue, result_queue, args.lossless_rgb))
        p.daemon = True
        p.start()
        workers.append(p)

    print(f"Writing to H5 archive: {dest_path}...")
    start_time = time.time()
    
    with h5py.File(dest_path, 'a') as h5_file:
        progress_bar = tqdm(total=len(all_files), desc="Packing dataset")
        
        while True:
            try:
                rel_path, data_type, payload = result_queue.get(timeout=0.5)
            except queue.Empty:
                # Correctly monitor if workers are alive to avoid infinite loops
                if all(not w.is_alive() for w in workers) and result_queue.empty():
                    break
                continue
            
            if data_type == 'error':
                print(f"\n[Warning] Skipped {rel_path} due to error: {payload}")
                progress_bar.update(1)
                continue
            
            try:
                if rel_path in h5_file:
                    del h5_file[rel_path]
                
                if data_type == 'npz':
                    grp = h5_file.create_group(rel_path)
                    for k, arr in payload.items():
                        grp.create_dataset(k, data=arr)
                        
                elif data_type == 'depth':
                    # Enabled chunking along with gzip/lzf compression for substantial depth file space savings
                    h5_file.create_dataset(rel_path, data=payload, chunks=True, compression="lzf")
                    
                elif data_type in ('rgb', 'mask'):
                    h5_file.create_dataset(rel_path, data=payload)
                    
                elif data_type in ('raw', 'raw_npy'):
                    h5_file.create_dataset(rel_path, data=np.void(payload))
            except Exception as e:
                print(f"\n[Writer Error] Failed to write {rel_path} to H5: {e}")
            
            progress_bar.update(1)

    for p in workers:
        p.join()

    duration = time.time() - start_time
    print(f"\nSUCCESS: Packed dataset into H5 archive in {duration:.1f} seconds.")
    print(f"H5 File Size: {os.path.getsize(dest_path) / (1024**3):.2f} GB")

if __name__ == '__main__':
    main()
