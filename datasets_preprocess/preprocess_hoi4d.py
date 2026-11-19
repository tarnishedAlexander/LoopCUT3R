#!/usr/bin/env python3
"""
HOI4D Preprocessing Script (Video Version)

Handles HOI4D data downloaded as video files:
  align_rgb/image.mp4 + align_depth/depth_video.avi

Extracts paired RGB+depth frames, rescales to 640x480, saves as:
  out_dir/<scene_name>/rgb/frame_XXXXXX.png
  out_dir/<scene_name>/depth/frame_XXXXXX.npy   (float32 meters)
  out_dir/<scene_name>/cam/frame_XXXXXX.npz     (intrinsics)

Usage:
    PYTHONPATH=/path/to/LoopCUT3R:/path/to/LoopCUT3R/src \
    python datasets_preprocess/preprocess_hoi4d.py \
        --root_dir /path/to/HOI4D_release \
        --cam_root /path/to/camera_params \
        --out_dir /path/to/processed_hoi4d
"""

import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import argparse

import src.dust3r.datasets.utils.cropping as cropping


def parse_arguments():
    parser = argparse.ArgumentParser(description="Preprocess HOI4D (video format).")
    parser.add_argument("--root_dir", required=True,
                        help="Path to HOI4D_release (contains ZY2021* scene dirs).")
    parser.add_argument("--cam_root", required=True,
                        help="Path to camera_params dir (contains ZY2021*/intrin.npy).")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for processed scenes.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Parallel worker count (default: cpu_count // 2).")
    parser.add_argument("--fps_stride", type=int, default=3,
                        help="Sample every N-th frame (default 3 → ~10 fps from 30 fps video).")
    return parser.parse_args()


def decode_depth_mm(frame):
    """
    Convert a raw depth video frame to uint16 millimeters.
    HOI4D depth AVIs pack 16-bit depth across two 8-bit channels:
      depth_mm = low_byte | (high_byte << 8)   [little-endian, B=low, G=high]
    Falls back gracefully for true grayscale or single-channel frames.
    """
    if frame.ndim == 2:
        return frame.astype(np.uint16)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[:, :, 0].astype(np.uint16)
    # 3-channel: try little-endian (B = low byte, G = high byte)
    depth_le = frame[:, :, 0].astype(np.uint16) | (frame[:, :, 1].astype(np.uint16) << 8)
    med = float(np.median(depth_le[depth_le > 0])) if np.any(depth_le > 0) else 0.0
    if 100.0 < med < 15000.0:
        return depth_le
    # Try big-endian (B = high byte, G = low byte)
    depth_be = (frame[:, :, 0].astype(np.uint16) << 8) | frame[:, :, 1].astype(np.uint16)
    return depth_be


def process_scene(task):
    """
    Open the RGB and depth videos for one scene, extract every fps_stride-th frame,
    rescale to 640x480, and save outputs.  Returns number of frames written.
    """
    scene_dir, cam_file, out_rgb_dir, out_depth_dir, out_cam_dir, fps_stride = task

    intrinsics = np.load(cam_file)

    rgb_video   = os.path.join(scene_dir, "align_rgb",   "image.mp4")
    depth_video = os.path.join(scene_dir, "align_depth", "depth_video.avi")

    if not os.path.isfile(rgb_video) or not os.path.isfile(depth_video):
        return 0

    cap_rgb = cv2.VideoCapture(rgb_video)
    cap_dep = cv2.VideoCapture(depth_video)

    frame_idx = 0
    saved = 0

    while True:
        ret_r, frame_r = cap_rgb.read()
        ret_d, frame_d = cap_dep.read()
        if not ret_r or not ret_d:
            break

        if frame_idx % fps_stride == 0:
            out_img_path = os.path.join(out_rgb_dir,   f"frame_{frame_idx:06d}.png")
            out_dep_path = os.path.join(out_depth_dir, f"frame_{frame_idx:06d}.npy")
            out_cam_path = os.path.join(out_cam_dir,   f"frame_{frame_idx:06d}.npz")

            if not (os.path.exists(out_img_path) and
                    os.path.exists(out_dep_path) and
                    os.path.exists(out_cam_path)):
                try:
                    img = Image.fromarray(cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB))

                    depth_mm = decode_depth_mm(frame_d)
                    depth_m  = depth_mm.astype(np.float32) / 1000.0

                    img_r, depth_r, intr_r = cropping.rescale_image_depthmap(
                        img, depth_m, intrinsics.copy(), (640, 480)
                    )

                    img_r.save(out_img_path)
                    np.save(out_dep_path, depth_r)
                    np.savez(out_cam_path, intrinsics=intr_r)
                    saved += 1
                except Exception as e:
                    print(f"[WARN] frame {frame_idx} in {scene_dir}: {e}")

        frame_idx += 1

    cap_rgb.release()
    cap_dep.release()
    return saved


def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    pattern = os.path.join(args.root_dir, "ZY2021*", "H*", "C*", "N*", "S*", "s*", "T*")
    scene_dirs = sorted(glob.glob(pattern))
    print(f"Found {len(scene_dirs)} scenes.")

    tasks = []
    for scene_dir in tqdm(scene_dirs, desc="Collecting scenes"):
        scene_relpath = os.path.relpath(scene_dir, args.root_dir)
        scene_name    = "_".join(scene_relpath.split(os.sep))
        top_level     = scene_relpath.split(os.sep)[0]

        cam_file = os.path.join(args.cam_root, top_level, "intrin.npy")
        if not os.path.isfile(cam_file):
            continue

        # Skip scenes without the expected video files
        rgb_video   = os.path.join(scene_dir, "align_rgb",   "image.mp4")
        depth_video = os.path.join(scene_dir, "align_depth", "depth_video.avi")
        if not os.path.isfile(rgb_video) or not os.path.isfile(depth_video):
            continue

        out_rgb_dir   = os.path.join(args.out_dir, scene_name, "rgb")
        out_depth_dir = os.path.join(args.out_dir, scene_name, "depth")
        out_cam_dir   = os.path.join(args.out_dir, scene_name, "cam")
        os.makedirs(out_rgb_dir,   exist_ok=True)
        os.makedirs(out_depth_dir, exist_ok=True)
        os.makedirs(out_cam_dir,   exist_ok=True)

        tasks.append((scene_dir, cam_file, out_rgb_dir, out_depth_dir, out_cam_dir, args.fps_stride))

    print(f"Queued {len(tasks)} scenes with video data.")

    max_workers = args.max_workers or max(1, os.cpu_count() // 2)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_scene, tasks),
            total=len(tasks),
            desc="Processing scenes",
        ))

    print(f"Done. Total frames written: {sum(results)}")


if __name__ == "__main__":
    main()
