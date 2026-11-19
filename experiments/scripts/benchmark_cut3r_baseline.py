#!/usr/bin/env python3
"""
LoopCUT3R Benchmark
=========================
Measures 8 metrics for paper/report comparison:
  1. Parameter count (total + breakdown)
  2. Peak VRAM at 512×288 (rescaled 1920x1080)
  3. FPS (frames per second)
  4. AbsRel (depth accuracy on Sintel)
  5. δ₁.₂₅  (depth threshold accuracy on Sintel)
  6. ATE    (absolute trajectory error on ScanNet/TUM)
  7. RPE translation (relative pose error, metres)
  8. RPE rotation    (relative pose error, degrees)

Usage:
  python benchmark_cut3r_baseline.py --ckpt checkpoints/recurrent_stage1/checkpoints/checkpoint-best.pth --depth_dataset sintel --pose_dataset scannet
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np

# ── Ensure LoopCUT3R src is importable ───────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))


# =========================================================
#  CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="LoopCUT3R Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", default="checkpoints/recurrent_stage1/checkpoints/checkpoint-best.pth",
                    help="Path to model checkpoint")
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=512,
                    help="Input resolution (224 or 512)")
    p.add_argument("--num_warmup", type=int, default=3,
                    help="Number of warm-up inference passes (not timed)")
    p.add_argument("--num_timed", type=int, default=10,
                    help="Number of timed inference passes for FPS/VRAM")
    p.add_argument("--img_dir", default="examples/001",
                    help="Directory of images for VRAM/FPS profiling")
    p.add_argument("--skip_depth_eval", action="store_true",
                    help="Skip depth evaluation")
    p.add_argument("--skip_pose_eval", action="store_true",
                    help="Skip pose evaluation")
    p.add_argument("--depth_dataset", default="sintel",
                    choices=["sintel", "bonn", "kitti"],
                    help="Dataset for depth evaluation")
    p.add_argument("--pose_dataset", default="scannet",
                    choices=["scannet", "tum", "sintel"],
                    help="Dataset for pose evaluation")
    p.add_argument("--depth_align", default="scale",
                    choices=["scale", "scale&shift", "metric"],
                    help="Depth alignment method")
    p.add_argument("--output_dir", default="eval_results",
                    help="Root directory for all outputs")
    return p.parse_args()


# =========================================================
#  Part 1: Parameter Count
# =========================================================
def count_parameters(model):
    """Count parameters with a breakdown by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ── Group by component ──
    groups = {
        "encoder":      [],   # patch_embed, enc_blocks, pos_embed, mask_token
        "decoder":      [],   # dec_blocks, dec_blocks_state
        "head":         [],   # downstream_head*, *_head*
        "state_memory": [],   # state*, mem*, pose_retriever, local_mem*
        "ray_encoder":  [],   # ray*
    }

    for name, p in model.named_parameters():
        nl = name.lower()
        if any(nl.startswith(pfx) for pfx in
               ("patch_embed", "enc_block", "enc_norm", "pos_embed", "mask_token")):
            groups["encoder"].append(p.numel())
        elif any(nl.startswith(pfx) for pfx in
                 ("dec_block", "dec_norm")):
            groups["decoder"].append(p.numel())
        elif "head" in nl or nl.startswith("downstream"):
            groups["head"].append(p.numel())
        elif any(kw in nl for kw in ("state", "mem", "pose_retriever", "local_mem")):
            groups["state_memory"].append(p.numel())
        elif nl.startswith("ray"):
            groups["ray_encoder"].append(p.numel())

    breakdown = {k: sum(v) for k, v in groups.items()}
    accounted = sum(breakdown.values())
    breakdown["other"] = total - accounted

    return {"total": total, "trainable": trainable, **breakdown}


# =========================================================
#  Part 2: VRAM & FPS Profiling
# =========================================================
def build_views(img_dir, size):
    """Load images and build the view-dict list expected by inference()."""
    from dust3r.utils.image import load_images

    # Gather image paths
    img_paths = sorted(
        [str(p) for p in Path(img_dir).glob("*.jpg")]
        + [str(p) for p in Path(img_dir).glob("*.png")]
    )
    if not img_paths:
        raise FileNotFoundError(f"No .jpg/.png images found in {img_dir}")

    images = load_images(img_paths, size=size)
    views = []
    for i, im in enumerate(images):
        h, w = im["true_shape"][0]
        views.append({
            "img":          im["img"],                                          # [1, 3, H, W]
            "ray_map":      torch.full((1, 6, int(h), int(w)), float("nan")),   # [1, 6, H, W]
            "true_shape":   torch.from_numpy(im["true_shape"]),                 # [1, 2]
            "idx":          i,
            "instance":     str(i),
            "camera_pose":  torch.eye(4, dtype=torch.float32).unsqueeze(0),     # [1, 4, 4]
            "img_mask":     torch.tensor([True]),
            "ray_mask":     torch.tensor([False]),
            "update":       torch.tensor([True]),
            "reset":        torch.tensor([i == 0]),
        })
    return views


def profile_inference(model, img_dir, size, device, num_warmup=3, num_timed=10):
    """Measure peak VRAM and FPS over repeated inference runs."""
    from dust3r.inference import inference

    views = build_views(img_dir, size)
    num_frames = len(views)
    h, w = views[0]["true_shape"][0].tolist()
    resolution = f"{int(w)}x{int(h)}"

    if device == "cpu":
        print("  [INFO] Device is CPU. Skipping GPU VRAM & FPS profiling.")
        return {
            "peak_vram_gb":       None,
            "fps":                None,
            "avg_time_per_seq_s": None,
            "num_frames":         num_frames,
            "resolution":         resolution,
            "num_timed_runs":     0,
        }

    # ── Warm-up ──
    print(f"  Warm-up: {num_warmup} pass(es) on {num_frames} frames @ {resolution} ...")
    for i in range(num_warmup):
        _ = inference(views, model, device, verbose=False)
        torch.cuda.synchronize(device)

    # ── Reset peak memory tracking ──
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # ── Timed runs ──
    print(f"  Timed:   {num_timed} pass(es) ...")
    times = []
    for r in range(num_timed):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = inference(views, model, device, verbose=False)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        times.append(elapsed)
        fps_this = num_frames / elapsed
        print(f"    Run {r + 1:>2}/{num_timed}: {elapsed:.3f}s  ({fps_this:.1f} FPS)")

    peak_vram_bytes = torch.cuda.max_memory_allocated(device)
    total_time = sum(times)
    fps = (num_timed * num_frames) / total_time

    return {
        "peak_vram_gb":       round(peak_vram_bytes / (1024 ** 3), 2),
        "fps":                round(fps, 2),
        "avg_time_per_seq_s": round(total_time / num_timed, 3),
        "num_frames":         num_frames,
        "resolution":         resolution,
        "num_timed_runs":     num_timed,
    }


# =========================================================
#  Part 3: Depth Accuracy
# =========================================================
DEPTH_DATA_DIRS = {
    "sintel": "data/sintel",
    "bonn":   "data/bonn",
    "kitti":  "data/kitti",
}


def eval_depth(ckpt_path, size, output_base, dataset="sintel", align="scale"):
    """Run the existing depth eval pipeline via subprocess."""
    data_dir = DEPTH_DATA_DIRS.get(dataset, f"data/{dataset}")
    if not os.path.exists(data_dir):
        print(f"  WARNING: {data_dir} not found — skipping depth evaluation.")
        return {"status": "DATASET_NOT_FOUND", "dataset": dataset}

    output_dir = f"{output_base}/video_depth/{dataset}_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Inference
    cmd_infer = [
        "accelerate", "launch", "--num_processes", "1",
        "eval/video_depth/launch.py",
        "--weights", ckpt_path,
        "--output_dir", output_dir,
        "--eval_dataset", dataset,
        "--size", str(size),
    ]
    print(f"  Inference cmd: {' '.join(cmd_infer)}")
    res = subprocess.run(cmd_infer, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    if res.returncode != 0:
        print(f"  Inference FAILED:\n{res.stderr[:800]}")
        return {"status": "INFERENCE_FAILED", "error": res.stderr[:500]}

    # Step 2: Evaluation
    results = {"status": "OK", "dataset": dataset}
    for al in [align, "scale&shift"] if align != "scale&shift" else ["scale&shift", "scale"]:
        cmd_eval = [
            sys.executable, "eval/video_depth/eval_depth.py",
            "--output_dir", output_dir,
            "--eval_dataset", dataset,
            "--align", al,
        ]
        print(f"  Eval cmd ({al}): {' '.join(cmd_eval)}")
        res = subprocess.run(cmd_eval, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
        if res.returncode != 0:
            print(f"  Eval ({al}) FAILED:\n{res.stderr[:400]}")
            continue

        result_file = os.path.join(output_dir, f"result_{al}.json")
        if os.path.exists(result_file):
            with open(result_file, "r") as f:
                metrics = json.load(f)
            tag = al.replace("&", "_and_")
            results[f"{tag}"] = {
                "abs_rel":     metrics.get("Abs Rel"),
                "delta_1.25":  metrics.get("δ < 1.25"),
                "delta_1.25²": metrics.get("δ < 1.25^2"),
                "delta_1.25³": metrics.get("δ < 1.25^3"),
                "sq_rel":      metrics.get("Sq Rel"),
                "rmse":        metrics.get("RMSE"),
                "log_rmse":    metrics.get("Log RMSE"),
            }
    return results


# =========================================================
#  Part 4: Pose Accuracy
# =========================================================
POSE_DATA_DIRS = {
    "scannet": "data/scannetv2",
    "tum":     "data/tum",
    "sintel":  "data/sintel",
}


def eval_pose(ckpt_path, size, output_base, dataset="scannet"):
    """Run the existing pose eval pipeline via subprocess."""
    data_dir = POSE_DATA_DIRS.get(dataset, f"data/{dataset}")
    if not os.path.exists(data_dir):
        print(f"  WARNING: {data_dir} not found — skipping pose evaluation.")
        return {"status": "DATASET_NOT_FOUND", "dataset": dataset}

    output_dir = f"{output_base}/relpose/{dataset}_benchmark"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "accelerate", "launch", "--num_processes", "1",
        "--main_process_port", "29558",
        "eval/relpose/launch.py",
        "--weights", ckpt_path,
        "--output_dir", output_dir,
        "--eval_dataset", dataset,
        "--size", str(size),
    ]
    print(f"  Pose cmd: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(SCRIPT_DIR))
    if res.returncode != 0:
        print(f"  Pose eval FAILED:\n{res.stderr[:800]}")
        return {"status": "EVAL_FAILED", "error": res.stderr[:500]}

    # Parse metric files using the existing evo_utils helpers
    sys.path.insert(0, str(SCRIPT_DIR / "eval" / "relpose"))
    from evo_utils import process_directory, calculate_averages

    seq_results = process_directory(output_dir)
    if not seq_results:
        return {"status": "NO_RESULTS_FOUND", "dataset": dataset}

    avg_ate, avg_rpe_t, avg_rpe_r = calculate_averages(seq_results)

    return {
        "status":       "OK",
        "dataset":      dataset,
        "ate_m":        round(avg_ate, 6),
        "rpe_trans_m":  round(avg_rpe_t, 6),
        "rpe_rot_deg":  round(avg_rpe_r, 4),
        "num_sequences": len(seq_results),
        "per_sequence": {
            r[0]: {"ate": round(r[1], 6),
                   "rpe_trans": round(r[2], 6),
                   "rpe_rot": round(r[3], 4)}
            for r in seq_results
        },
    }


# =========================================================
#  Part 5: Output Formatting
# =========================================================
def print_header(title, width=64):
    print(f"\n  {'─' * width}")
    print(f"  {title}")
    print(f"  {'─' * width}")


def format_table(results):
    """Print a formatted summary table to stdout."""
    W = 64
    model_name = results.get("model", "MODEL").upper()
    print("\n" + "=" * (W + 4))
    print(f"  {f'{model_name} BENCHMARK RESULTS':^{W}}")
    print("=" * (W + 4))

    # ── Parameters ──
    params = results.get("parameters", {})
    print_header("Model Parameters")
    print(f"    Total:         {params.get('total', 0) / 1e6:>10.2f} M")
    print(f"    Trainable:     {params.get('trainable', 0) / 1e6:>10.2f} M")
    print(f"    Encoder:       {params.get('encoder', 0) / 1e6:>10.2f} M")
    print(f"    Decoder:       {params.get('decoder', 0) / 1e6:>10.2f} M")
    print(f"    Head:          {params.get('head', 0) / 1e6:>10.2f} M")
    print(f"    State/Memory:  {params.get('state_memory', 0) / 1e6:>10.2f} M")
    print(f"    Ray encoder:   {params.get('ray_encoder', 0) / 1e6:>10.2f} M")
    if params.get("other", 0) > 0:
        print(f"    Other:         {params['other'] / 1e6:>10.2f} M")

    # ── Inference Profile ──
    inf = results.get("inference_profile")
    if inf:
        print_header("Inference Profile")
        print(f"    Resolution:    {inf['resolution']:>10}")
        print(f"    Num frames:    {inf['num_frames']:>10}")
        if inf['peak_vram_gb'] is not None:
            print(f"    Peak VRAM:     {inf['peak_vram_gb']:>10.2f} GB")
            print(f"    FPS:           {inf['fps']:>10.2f}")
            print(f"    Avg seq time:  {inf['avg_time_per_seq_s']:>10.3f} s")
        else:
            print("    Peak VRAM:            N/A (CPU)")
            print("    FPS:                  N/A (CPU)")

    # ── Depth Accuracy ──
    depth = results.get("depth_accuracy_sintel", {})
    if depth.get("status") == "OK":
        primary = depth.get("scale") or depth.get("scale_and_shift") or {}
        al_name = "scale" if depth.get("scale") else "scale&shift"
        ds = depth.get("dataset", "sintel")
        print_header(f"Depth Accuracy ({ds}, {al_name} alignment)")
        print(f"    AbsRel:        {primary.get('abs_rel', 'N/A')}")
        print(f"    δ < 1.25:      {primary.get('delta_1.25', 'N/A')}")
        print(f"    δ < 1.25²:     {primary.get('delta_1.25²', 'N/A')}")
        print(f"    δ < 1.25³:     {primary.get('delta_1.25³', 'N/A')}")
        print(f"    RMSE:          {primary.get('rmse', 'N/A')}")
        print(f"    Sq Rel:        {primary.get('sq_rel', 'N/A')}")
    elif depth.get("status") == "DATASET_NOT_FOUND":
        print_header("Depth Accuracy")
        print(f"    SKIPPED — {depth.get('dataset', 'sintel')} data not found")

    # ── Pose Accuracy ──
    pose = results.get("pose_accuracy_scannet", {})
    if pose.get("status") == "OK":
        ds = pose.get("dataset", "scannet")
        print_header(f"Pose Accuracy ({ds})")
        print(f"    ATE:           {pose['ate_m']:>10.6f} m")
        print(f"    RPE trans:     {pose['rpe_trans_m']:>10.6f} m")
        print(f"    RPE rot:       {pose['rpe_rot_deg']:>10.4f} °")
        print(f"    Num sequences: {pose['num_sequences']:>10}")
    elif pose.get("status") == "DATASET_NOT_FOUND":
        print_header("Pose Accuracy")
        print(f"    SKIPPED — {pose.get('dataset', 'scannet')} data not found")

    print("\n" + "=" * (W + 4))


# =========================================================
#  Main
# =========================================================
def main():
    args = parse_args()

    # Resolve checkpoint path relative to LoopCUT3R root
    ckpt_path = str(SCRIPT_DIR / args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    if not os.path.isfile(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    cuda_available = False
    gpu_name = "N/A (CPU)"
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "CPU Only"
    except Exception as e:
        print(f"  [WARNING] CUDA check failed: {e}")
        gpu_name = "CUDA init error (CPU fallback)"

    if device == "cuda" and not cuda_available:
        print("  [WARNING] CUDA not available. Falling back to CPU.")
        device = "cpu"

    results = OrderedDict()
    results["model"] = Path(ckpt_path).stem
    results["checkpoint_size_mb"] = round(os.path.getsize(ckpt_path) / (1024 ** 2), 1)
    results["input_size"] = args.size
    results["device"] = gpu_name

    # ─────────────────────────────────────────────────────
    # Part 1: Load model & count parameters
    # ─────────────────────────────────────────────────────
    print("\n[1/5] Loading model and counting parameters ...")
    from dust3r.model import ARCroco3DStereo
    print(f"  Loading model from {ckpt_path} on {device}...")
    
    model = ARCroco3DStereo.from_pretrained(ckpt_path)
    if device != "cpu":
        try:
            model = model.to(device)
        except Exception as e:
            print(f"  [WARNING] Transfer to {device} failed: {e}. Running on CPU instead.")
            device = "cpu"
    model.eval()

    results["parameters"] = count_parameters(model)
    total_m = results["parameters"]["total"] / 1e6
    print(f"  Total: {total_m:.2f}M parameters")

    # ─────────────────────────────────────────────────────
    # Part 2: VRAM & FPS profiling
    # ─────────────────────────────────────────────────────
    print(f"\n[2/5] Profiling VRAM & FPS (size={args.size}) ...")
    img_dir = str(SCRIPT_DIR / args.img_dir) if not os.path.isabs(args.img_dir) else args.img_dir
    profile = profile_inference(
        model, img_dir, args.size, device,
        num_warmup=args.num_warmup,
        num_timed=args.num_timed,
    )
    results["inference_profile"] = profile
    if profile["fps"] is not None:
        print(f"  Peak VRAM: {profile['peak_vram_gb']:.2f} GB | FPS: {profile['fps']:.2f}")

    # Free the model
    del model
    if cuda_available:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────
    # Part 3: Depth evaluation
    # ─────────────────────────────────────────────────────
    if not args.skip_depth_eval:
        print(f"\n[3/5] Depth evaluation ({args.depth_dataset}) ...")
        depth_results = eval_depth(
            ckpt_path, args.size, args.output_dir,
            dataset=args.depth_dataset, align=args.depth_align,
        )
        results["depth_accuracy_sintel"] = depth_results
    else:
        print("\n[3/5] Depth evaluation — SKIPPED (--skip_depth_eval)")
        results["depth_accuracy_sintel"] = {"status": "SKIPPED"}

    # ─────────────────────────────────────────────────────
    # Part 4: Pose evaluation
    # ─────────────────────────────────────────────────────
    if not args.skip_pose_eval:
        print(f"\n[4/5] Pose evaluation ({args.pose_dataset}) ...")
        pose_results = eval_pose(
            ckpt_path, args.size, args.output_dir,
            dataset=args.pose_dataset,
        )
        results["pose_accuracy_scannet"] = pose_results
    else:
        print("\n[4/5] Pose evaluation — SKIPPED (--skip_pose_eval)")
        results["pose_accuracy_scannet"] = {"status": "SKIPPED"}

    # ─────────────────────────────────────────────────────
    # Part 5: Write output
    # ─────────────────────────────────────────────────────
    print("\n[5/5] Writing results ...")
    out_path = os.path.join(args.output_dir, "baseline_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  JSON saved to {out_path}")

    format_table(results)


if __name__ == "__main__":
    main()
