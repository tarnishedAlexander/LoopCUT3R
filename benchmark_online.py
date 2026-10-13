#!/usr/bin/env python3
"""
Benchmark online (streaming) inference speed of LoopCUT3R.
Simulates exactly what a ROS node would do: one frame at a time,
recurrent state carried forward, no global alignment.
"""

import os
import glob
import time
import argparse
import numpy as np
import torch

from add_ckpt_path import add_path_to_dust3r


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="checkpoints/recurrent_stage1_r3/checkpoints/checkpoint-last.pth")
    parser.add_argument("--seq_path", type=str,
                        default="demo_data/minicity")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup frames before timing (GPU JIT / caching)")
    parser.add_argument("--max_frames", type=int, default=100,
                        help="Total frames to benchmark")
    return parser.parse_args()


def load_model(model_path, device):
    add_path_to_dust3r(model_path)
    from src.mast3r.model import AsymmetricMASt3R
    from src.dust3r.model import AsymmetricCroCo3DStereo
    from src.ARCroco3DStereo import ARCroco3DStereo

    print(f"Loading model from {model_path} ...")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_args = getattr(ckpt.get("args"), "model", None)

    model = eval(model_args)
    model.load_state_dict(ckpt["model"], strict=False)
    model = model.to(device).eval()
    print("Model loaded.")
    return model


def prepare_frame(img_path, size, device):
    from src.dust3r.utils.image import load_images
    imgs = load_images([img_path], size=size)
    img = imgs[0]
    view = {
        "img": img["img"].to(device),
        "ray_map": torch.full(
            (img["img"].shape[0], 6, img["img"].shape[-2], img["img"].shape[-1]),
            torch.nan, device=device
        ),
        "true_shape": torch.from_numpy(img["true_shape"]).to(device),
        "idx": 0,
        "instance": "0",
        "camera_pose": torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0),
        "img_mask": torch.tensor(True, device=device).unsqueeze(0),
        "ray_mask": torch.tensor(False, device=device).unsqueeze(0),
        "update": torch.tensor(True, device=device).unsqueeze(0),
        "reset": torch.tensor(False, device=device).unsqueeze(0),
    }
    return view


def main():
    args = parse_args()

    img_paths = sorted(glob.glob(os.path.join(args.seq_path, "*.jpg")) +
                       glob.glob(os.path.join(args.seq_path, "*.png")))
    img_paths = img_paths[:args.max_frames]
    print(f"Found {len(img_paths)} frames, benchmarking {len(img_paths)} "
          f"(warmup: {args.warmup})")

    model = load_model(args.model_path, args.device)

    # Pre-load all frames to CPU to avoid disk I/O skewing results
    print("Pre-loading frames to CPU memory...")
    views = [prepare_frame(p, args.size, "cpu") for p in img_paths]
    print("Frames loaded.")

    state = None
    frame_times = []

    print(f"\nRunning online inference ({args.warmup} warmup + "
          f"{len(views) - args.warmup} timed frames)...\n")

    for i, view in enumerate(views):
        # Move to device just before inference (simulates ROS callback latency)
        v = {k: (val.to(args.device) if isinstance(val, torch.Tensor) else val)
             for k, val in view.items()}

        if state is not None:
            v["reset"] = torch.tensor(False, device=args.device).unsqueeze(0)
        else:
            v["reset"] = torch.tensor(True, device=args.device).unsqueeze(0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output, state = model([v], return_state=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0

        if i >= args.warmup:
            frame_times.append(elapsed)

        vram = torch.cuda.memory_allocated() / 1024**3

        if i % 10 == 0:
            fps_str = f"{1/elapsed:.1f} fps" if elapsed > 0 else "?"
            warmup_marker = " [warmup]" if i < args.warmup else ""
            print(f"  Frame {i:3d}: {elapsed*1000:.1f} ms  ({fps_str})  "
                  f"VRAM: {vram:.2f} GB{warmup_marker}")

    if not frame_times:
        print("No timed frames — increase max_frames or decrease warmup.")
        return

    arr = np.array(frame_times) * 1000  # ms
    print(f"\n{'='*50}")
    print(f"Results ({len(frame_times)} timed frames, {args.size}px):")
    print(f"  Mean latency : {arr.mean():.1f} ms  ({1000/arr.mean():.1f} fps)")
    print(f"  Median       : {np.median(arr):.1f} ms  ({1000/np.median(arr):.1f} fps)")
    print(f"  P95          : {np.percentile(arr, 95):.1f} ms")
    print(f"  Min / Max    : {arr.min():.1f} / {arr.max():.1f} ms")
    print(f"  VRAM (steady): {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"  Target 20fps : {'✓ YES' if 1000/arr.mean() >= 20 else '✗ NO'} "
          f"(need ≤50ms, got {arr.mean():.1f}ms)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
