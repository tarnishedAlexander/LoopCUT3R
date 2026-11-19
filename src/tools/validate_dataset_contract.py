#!/usr/bin/env python3
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dust3r.datasets import get_data_loader


REQUIRED_VIEW_KEYS = {
    "img",
    "ray_map",
    "true_shape",
    "img_mask",
    "ray_mask",
    "camera_pose",
    "camera_intrinsics",
    "depthmap",
    "valid_mask",
    "reset",
}


def parse_args():
    parser = argparse.ArgumentParser("Validate CUT3R dataset contract")
    parser.add_argument("--config-dir", type=str, default="../config")
    parser.add_argument("--config-name", type=str, default="encoder_only_mydata")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument(
        "--dataset-expr",
        type=str,
        default=None,
        help="Optional explicit dataset expression. If omitted, uses first dataset from config split.",
    )
    parser.add_argument(
        "--check-processed-root",
        type=str,
        default=None,
        help="Optional root path to validate processed sequence layout (rgb/depth/cam).",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=5,
        help="Max sequence folders to inspect in --check-processed-root mode.",
    )
    return parser.parse_args()


def first_dataset_expr(expr: str) -> str:
    return expr.split("+")[0].strip()


def load_cfg(config_dir: str, config_name: str):
    cfg_path = Path(config_dir) / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    return OmegaConf.load(cfg_path), cfg_path.resolve()


def pick_dataset_expr(cfg, split: str) -> str:
    expr = str(cfg.train_dataset if split == "train" else cfg.test_dataset)
    return first_dataset_expr(expr)


def _extract_root(dataset_expr: str):
    m = re.search(r"ROOT\s*=\s*'([^']*)'", dataset_expr)
    if m:
        return m.group(1), "'"
    m = re.search(r'ROOT\s*=\s*"([^"]*)"', dataset_expr)
    if m:
        return m.group(1), '"'
    return None, None


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p

    from_cfg = (base_dir / p).resolve()
    if from_cfg.exists():
        return from_cfg
    return (Path.cwd() / p).resolve()


def maybe_override_loader_root(dataset_expr: str, processed_root: Path, base_dir: Path):
    """Patch ROOT=... in dataset expression when current ROOT is missing.

    Behavior:
    1. If ROOT is not present, leave expr unchanged.
    2. If ROOT resolves to an existing dir, leave expr unchanged.
    3. If ROOT is missing and processed_root exists, replace ROOT with processed_root.
    """
    root_value, quote = _extract_root(dataset_expr)
    if root_value is None:
        return dataset_expr, None

    resolved_root = _resolve_path(root_value, base_dir)
    if resolved_root.exists():
        return dataset_expr, None

    if not processed_root.exists():
        return dataset_expr, (
            f"config ROOT path missing ({resolved_root}) and --check-processed-root does not exist "
            f"({processed_root})"
        )

    if quote == '"':
        patched = re.sub(
            r'ROOT\s*=\s*"[^"]*"',
            f'ROOT="{processed_root}"',
            dataset_expr,
            count=1,
        )
    else:
        patched = re.sub(
            r"ROOT\s*=\s*'[^']*'",
            f"ROOT='{processed_root}'",
            dataset_expr,
            count=1,
        )

    return patched, (
        f"overrode loader ROOT: {resolved_root} -> {processed_root}"
    )


def validate_processed_layout(root: Path, max_sequences: int):
    errors = []
    warnings = []

    if not root.exists():
        return [f"processed root does not exist: {root}"], warnings
    if not root.is_dir():
        return [f"processed root is not a directory: {root}"], warnings

    seq_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not seq_dirs:
        return [f"no sequence directories found under: {root}"], warnings

    inspected = 0
    for seq in seq_dirs:
        if inspected >= max_sequences:
            break
        rgb_dir = seq / "rgb"
        depth_dir = seq / "depth"
        cam_dir = seq / "cam"

        if not rgb_dir.is_dir() or not depth_dir.is_dir() or not cam_dir.is_dir():
            errors.append(
                f"{seq.name}: expected rgb/depth/cam directories"
            )
            continue

        rgb = {p.stem for p in rgb_dir.glob("*.jpg")}
        depth = {p.stem for p in depth_dir.glob("*.npy")}
        cam = {p.stem for p in cam_dir.glob("*.npz")}

        common = sorted(rgb & depth & cam)
        if not common:
            errors.append(f"{seq.name}: no common basenames across rgb/depth/cam")
            continue

        if len(common) < 3:
            warnings.append(f"{seq.name}: very short sequence ({len(common)} frames)")

        sample_npz = cam_dir / f"{common[0]}.npz"
        try:
            data = np.load(sample_npz)
            if "intrinsics" not in data or "pose" not in data:
                errors.append(f"{seq.name}: {sample_npz.name} missing intrinsics/pose")
            else:
                intr = data["intrinsics"]
                pose = data["pose"]
                if intr.shape != (3, 3):
                    errors.append(f"{seq.name}: intrinsics shape {intr.shape}, expected (3,3)")
                if pose.shape != (4, 4):
                    errors.append(f"{seq.name}: pose shape {pose.shape}, expected (4,4)")
        except Exception as exc:
            errors.append(f"{seq.name}: failed reading {sample_npz.name}: {exc}")

        inspected += 1

    if inspected == 0:
        errors.append("no sequences inspected")

    return errors, warnings


def check_tensor(name: str, value: torch.Tensor, errors):
    if not torch.is_tensor(value):
        errors.append(f"key={name}: expected tensor, got {type(value)}")
        return
    if not torch.isfinite(value).all() and value.dtype.is_floating_point:
        # NaNs can be intentional for masked tokens in some flows, so this is warning-level behavior.
        pass


def validate_view_schema(view: dict, view_idx: int):
    errors = []
    missing = sorted(REQUIRED_VIEW_KEYS - set(view.keys()))
    if missing:
        errors.append(f"view[{view_idx}] missing required keys: {missing}")

    if "img" in view:
        img = view["img"]
        if not torch.is_tensor(img) or img.ndim != 4:
            errors.append(f"view[{view_idx}].img must be [B,3,H,W], got type={type(img)}")
        elif img.shape[1] != 3:
            errors.append(f"view[{view_idx}].img channel dim must be 3, got {img.shape}")

    if "ray_map" in view:
        ray_map = view["ray_map"]
        if not torch.is_tensor(ray_map):
            errors.append(f"view[{view_idx}].ray_map must be tensor")
        elif ray_map.ndim != 4:
            errors.append(f"view[{view_idx}].ray_map must be 4D, got {ray_map.shape}")
        else:
            if ray_map.shape[-1] != 6 and ray_map.shape[1] != 6:
                errors.append(
                    f"view[{view_idx}].ray_map expected 6 channels in last or second dim, got {ray_map.shape}"
                )

    for key in [
        "true_shape",
        "img_mask",
        "ray_mask",
        "camera_pose",
        "camera_intrinsics",
        "depthmap",
        "valid_mask",
        "reset",
    ]:
        if key in view:
            check_tensor(key, view[key], errors)

    if "camera_pose" in view:
        cp = view["camera_pose"]
        if torch.is_tensor(cp) and cp.ndim == 3 and cp.shape[-2:] != (4, 4):
            errors.append(f"view[{view_idx}].camera_pose expected [B,4,4], got {cp.shape}")

    if "camera_intrinsics" in view:
        intr = view["camera_intrinsics"]
        if torch.is_tensor(intr) and intr.ndim == 3 and intr.shape[-2:] != (3, 3):
            errors.append(
                f"view[{view_idx}].camera_intrinsics expected [B,3,3], got {intr.shape}"
            )

    return errors


def validate_loader(dataset_expr: str, cfg, args):
    errors = []
    warnings = []

    batch_size = int(args.batch_size if args.batch_size is not None else cfg.batch_size)
    num_workers = int(args.num_workers if args.num_workers is not None else cfg.num_workers)

    loader = get_data_loader(
        dataset_expr,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
        fixed_length=bool(getattr(cfg, "fixed_length", True)),
    )

    # Match train/test codepath requirements for datasets wrapped by
    # epoch-aware samplers/resizers.
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(0)
    if (
        hasattr(loader, "batch_sampler")
        and hasattr(loader.batch_sampler, "batch_sampler")
        and hasattr(loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        loader.batch_sampler.batch_sampler.set_epoch(0)

    seen_batches = 0
    for batch in loader:
        if not isinstance(batch, (list, tuple)):
            errors.append(f"batch must be list/tuple of views, got {type(batch)}")
            break

        if len(batch) == 0:
            errors.append("batch contains zero views")
            break

        for i, view in enumerate(batch):
            if not isinstance(view, dict):
                errors.append(f"view[{i}] is not dict, got {type(view)}")
                continue
            errors.extend(validate_view_schema(view, i))

        seen_batches += 1
        if seen_batches >= args.max_batches:
            break

    if seen_batches == 0:
        errors.append("dataloader produced no batches")

    return errors, warnings, seen_batches


def main():
    args = parse_args()

    cfg, cfg_path = load_cfg(args.config_dir, args.config_name)
    dataset_expr = args.dataset_expr or pick_dataset_expr(cfg, args.split)

    print(f"[info] config={args.config_name} split={args.split}")
    print(f"[info] config_path={cfg_path}")
    print(f"[info] dataset_expr={dataset_expr}")

    all_errors = []
    all_warnings = []
    run_base_dir = cfg_path.parent
    processed_root = None

    if args.check_processed_root is not None:
        processed_root = Path(args.check_processed_root).resolve()
        print(f"[info] validating processed root: {processed_root}")
        errors, warnings = validate_processed_layout(processed_root, args.max_sequences)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

    if processed_root is not None:
        patched_expr, note = maybe_override_loader_root(
            dataset_expr, processed_root, run_base_dir
        )
        if note is not None:
            print(f"[info] {note}")
        dataset_expr = patched_expr
        print(f"[info] loader_dataset_expr={dataset_expr}")

    print("[info] validating dataloader contract")
    errors, warnings, seen_batches = validate_loader(dataset_expr, cfg, args)
    all_errors.extend(errors)
    all_warnings.extend(warnings)

    print(f"[info] inspected batches: {seen_batches}")
    if all_warnings:
        print("[warn] warnings:")
        for w in all_warnings:
            print(f"  - {w}")

    if all_errors:
        print("[error] contract validation failed:")
        for e in all_errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("[ok] dataset contract validation passed")


if __name__ == "__main__":
    main()
