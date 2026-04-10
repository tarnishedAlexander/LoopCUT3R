#!/usr/bin/env python3
import argparse
import datetime
import gc
import json
import os
import sys
from pathlib import Path

# Ensure `src/` is importable when this file is run as:
#   python tools/layer_similarity_encoder.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from dust3r.datasets import get_data_loader
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, inf, strip_module  # noqa: F401
import dust3r.utils.path_to_croco  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser("CUT3R encoder layer similarity")
    parser.add_argument("--config-dir", type=str, default="../config")
    parser.add_argument("--config-name", type=str, default="encoder_only_mydata_5gb")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-images", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Execution device. Use cpu to avoid CUDA OOM during analysis.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--with-cka", action="store_true", help="(obsolete, CKA is now always active)")
    parser.add_argument("--report-topk", type=int, default=8)
    
    args = parser.parse_args()
    args.with_cka = True  # Always compute CKA by default
    return args


def _first_dataset_expr(expr: str) -> str:
    return expr.split("+")[0].strip()


def _to_device_batch(batch, device):
    ignore_keys = {"depthmap", "dataset", "label", "instance", "idx", "rng"}
    for view in batch:
        for name in list(view.keys()):
            if name in ignore_keys:
                continue
            value = view[name]
            if torch.is_tensor(value):
                view[name] = value.to(device, non_blocking=True)
            elif isinstance(value, (list, tuple)):
                view[name] = [
                    x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                    for x in value
                ]
    return batch


def _save_heatmap(matrix: np.ndarray, title: str, save_path: Path):
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        im = ax.imshow(matrix, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Layer index")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] heatmap save skipped for {save_path.name}: {exc}")


def _save_drift_heatmap(matrix: np.ndarray, title: str, save_path: Path):
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(matrix, aspect="auto", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Temporal step t→t+1")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] drift heatmap save skipped for {save_path.name}: {exc}")


def _load_checkpoint_if_any(model, checkpoint_path, device):
    if checkpoint_path is None:
        return
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        print(f"[warn] checkpoint not found: {ckpt_path}")
        return
    print(f"Loading checkpoint: {ckpt_path}")
    # Always load checkpoint tensors on CPU first to avoid GPU OOM spikes.
    # CUT3R training checkpoints often store OmegaConf objects under `args`.
    # PyTorch >=2.6 defaults to weights_only=True, which rejects these objects.
    # We explicitly use weights_only=False for trusted local checkpoints.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        print("[warn] unsupported checkpoint format, skipping load")
        return
    msg = model.load_state_dict(strip_module(state_dict), strict=False)
    print(msg)
    del state_dict
    del ckpt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _update_stats(stats, feats):
    # feats: L x B x D
    B = feats.shape[1]
    feats_norm = F.normalize(feats, dim=-1, eps=1e-6)
    cosine_dist = 1.0 - torch.einsum("lbd,mbd->lmb", feats_norm, feats_norm).mean(
        dim=-1
    )
    mse_dist = (feats[:, None] - feats[None, :]).pow(2).mean(dim=-1).mean(dim=-1)
    norms = torch.linalg.norm(feats, ord=2, dim=-1).mean(dim=-1)

    if stats["cosine_acc"] is None:
        L = feats.shape[0]
        stats["cosine_acc"] = torch.zeros(L, L)
        stats["mse_acc"] = torch.zeros(L, L)
        stats["norm_acc"] = torch.zeros(L)

    stats["cosine_acc"] += cosine_dist.cpu() * B
    stats["mse_acc"] += mse_dist.cpu() * B
    stats["norm_acc"] += norms.cpu() * B
    stats["count"] += B


def _finalize_stats(stats):
    if stats["count"] == 0:
        return None
    denom = stats["count"]
    return {
        "cosine": (stats["cosine_acc"] / denom).numpy(),
        "mse": (stats["mse_acc"] / denom).numpy(),
        "norms": (stats["norm_acc"] / denom).numpy(),
        "count": int(denom),
    }


def _empty_stats():
    return {
        "cosine_acc": None,
        "mse_acc": None,
        "norm_acc": None,
        "count": 0,
    }


def _append_to_bank(bank, feats):
    # feats: L x B x D
    feats_cpu = feats.detach().cpu()
    if bank["layers"] is None:
        bank["layers"] = [[] for _ in range(feats_cpu.shape[0])]
    for l in range(feats_cpu.shape[0]):
        bank["layers"][l].append(feats_cpu[l])


def _finalize_bank(bank):
    if bank["layers"] is None:
        return None
    return [torch.cat(chunks, dim=0) for chunks in bank["layers"]]


def _center_gram(X):
    K = X @ X.T
    n = K.shape[0]
    one = torch.ones((n, n), device=K.device, dtype=K.dtype) / n
    return K - one @ K - K @ one + one @ K @ one


def _linear_cka(X, Y, eps=1e-8):
    K = _center_gram(X)
    L = _center_gram(Y)
    hsic = (K * L).sum()
    denom = torch.sqrt((K * K).sum() * (L * L).sum()).clamp_min(eps)
    return (hsic / denom).item()


def _cka_matrix(layer_bank):
    if layer_bank is None:
        return None
    L = len(layer_bank)
    out = np.zeros((L, L), dtype=np.float32)
    with torch.no_grad():
        for i in range(L):
            Xi = layer_bank[i].float()
            for j in range(L):
                Xj = layer_bank[j].float()
                out[i, j] = _linear_cka(Xi, Xj)
    return out


def _distance_top_pairs(matrix, topk=8):
    L = matrix.shape[0]
    pairs = []
    for i in range(L):
        for j in range(i + 1, L):
            pairs.append((i, j, float(matrix[i, j])))
    pairs.sort(key=lambda x: x[2])
    return pairs[:topk]


def _early_mid_late_trend(matrix):
    L = matrix.shape[0]
    if L < 3:
        return None
    s1 = max(1, L // 3)
    s2 = max(s1 + 1, (2 * L) // 3)
    early = matrix[:s1, :s1]
    mid = matrix[s1:s2, s1:s2]
    late = matrix[s2:, s2:]

    def offdiag_mean(M):
        if M.shape[0] <= 1:
            return float(np.nan)
        mask = ~np.eye(M.shape[0], dtype=bool)
        return float(M[mask].mean())

    return {
        "early": offdiag_mean(early),
        "mid": offdiag_mean(mid),
        "late": offdiag_mean(late),
    }


def _write_report(out_dir: Path, report_data, topk=8):
    lines = ["# CUT3R Layer Similarity Report", ""]
    lines.append(f"Generated: {datetime.datetime.now().isoformat()}")
    lines.append("")

    for name, block in report_data.items():
        if block is None:
            continue
        lines.append(f"## {name}")
        lines.append("")

        cosine = block.get("cosine")
        if cosine is not None:
            top_pairs = _distance_top_pairs(cosine, topk=topk)
            lines.append("### Top similar layer pairs (lowest cosine distance)")
            for i, j, v in top_pairs:
                lines.append(f"- layers ({i}, {j}): cosine_distance={v:.6f}")
            lines.append("")

            lines.append("### Block redundancy hints")
            redundant = [
                (i, i + 1, float(cosine[i, i + 1]))
                for i in range(cosine.shape[0] - 1)
                if cosine[i, i + 1] < 0.02
            ]
            if redundant:
                for i, j, v in redundant:
                    lines.append(f"- adjacent ({i}, {j}) near-redundant: {v:.6f}")
            else:
                lines.append("- no strong adjacent redundancy under threshold 0.02")
            lines.append("")

            trend = _early_mid_late_trend(cosine)
            lines.append("### Progression trend (early/mid/late)")
            if trend is None:
                lines.append("- not enough layers")
            else:
                lines.append(
                    f"- mean off-diagonal cosine distance: early={trend['early']:.6f}, mid={trend['mid']:.6f}, late={trend['late']:.6f}"
                )
                if np.isfinite(trend["early"]) and np.isfinite(trend["late"]):
                    if trend["late"] < trend["early"]:
                        lines.append("- hint: late-stage collapse/compaction increases")
                    else:
                        lines.append("- hint: no late-stage collapse increase detected")
            lines.append("")

        if block.get("cka") is not None:
            lines.append("### CKA summary")
            cka = block["cka"]
            cka_pairs = []
            for i in range(cka.shape[0]):
                for j in range(i + 1, cka.shape[1]):
                    cka_pairs.append((i, j, float(cka[i, j])))
            cka_pairs.sort(key=lambda x: x[2], reverse=True)
            for i, j, v in cka_pairs[:topk]:
                lines.append(f"- layers ({i}, {j}): cka={v:.6f}")
            lines.append("")

    with open(out_dir / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def compute_layer_similarity(model, loader, device, max_images=512, use_amp=True):
    model.eval()
    image_global = _empty_stats()
    ray_global = _empty_stats()
    merged_global = _empty_stats()
    image_by_view = {}
    ray_by_view = {}
    merged_by_view = {}
    image_bank = {"layers": None}
    ray_bank = {"layers": None}
    merged_bank = {"layers": None}
    drift_mse_sum = None
    drift_cos_sum = None
    drift_count = 0
    budget_images = 0

    amp_enabled = use_amp and (device.type == "cuda")

    with torch.no_grad():
        for batch in loader:
            batch = _to_device_batch(batch, device)
            num_views = len(batch)
            anchor_views = sorted(set([0, num_views // 2, num_views - 1]))

            # Merged representation (after _encode_views assembly)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                _, merged_layers_chunked, _ = model._encode_views_all_layers(batch)

            merged_feats_per_view = []
            for t in range(num_views):
                feats_t = torch.stack(
                    [layer_chunk[t].mean(dim=1).float() for layer_chunk in merged_layers_chunked],
                    dim=0,
                )
                merged_feats_per_view.append(feats_t)
                _update_stats(merged_global, feats_t)
                if t in anchor_views:
                    if t not in merged_by_view:
                        merged_by_view[t] = _empty_stats()
                    _update_stats(merged_by_view[t], feats_t)
                _append_to_bank(merged_bank, feats_t)

            if len(merged_feats_per_view) > 1:
                L = merged_feats_per_view[0].shape[0]
                if drift_mse_sum is None:
                    drift_mse_sum = torch.zeros(num_views - 1, L)
                    drift_cos_sum = torch.zeros(num_views - 1, L)
                for t in range(num_views - 1):
                    a = merged_feats_per_view[t]
                    b = merged_feats_per_view[t + 1]
                    mse_l = ((a - b).pow(2)).mean(dim=-1).mean(dim=-1)  # [L]
                    cos_l = 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-6).mean(
                        dim=-1
                    )
                    drift_mse_sum[t] += mse_l.cpu()
                    drift_cos_sum[t] += cos_l.cpu()
                drift_count += 1

            # batch is a list of views, each view has a batch dimension.
            for view_idx, view in enumerate(batch):
                img = view["img"]
                if "img_mask" in view:
                    img_mask = view["img_mask"].bool()
                else:
                    img_mask = torch.ones(
                        img.shape[0], dtype=torch.bool, device=img.device
                    )
                if "true_shape" in view:
                    true_shape_full = view["true_shape"]
                else:
                    h, w = img.shape[-2:]
                    true_shape_full = torch.tensor(
                        [[h, w]], device=img.device, dtype=torch.int64
                    ).repeat(img.shape[0], 1)

                # Image path stats
                if img_mask.sum() > 0:
                    img_sel = img[img_mask]
                    true_shape = true_shape_full[img_mask]

                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                    ):
                        layer_acts, _, _ = model._encode_image_all_layers(
                            img_sel, true_shape
                        )

                    feats = torch.stack(
                        [act.mean(dim=1).float() for act in layer_acts], dim=0
                    )
                    _update_stats(image_global, feats)
                    if view_idx in anchor_views:
                        if view_idx not in image_by_view:
                            image_by_view[view_idx] = _empty_stats()
                        _update_stats(image_by_view[view_idx], feats)
                    _append_to_bank(image_bank, feats)

                    budget_images += int(img_sel.shape[0])

                # Ray-map path stats
                ray_mask = (
                    view["ray_mask"].bool()
                    if "ray_mask" in view
                    else torch.zeros(img.shape[0], dtype=torch.bool, device=img.device)
                )
                if ray_mask.sum() > 0:
                    ray_sel = view["ray_map"][ray_mask].permute(0, 3, 1, 2)
                    true_shape_ray = true_shape_full[ray_mask]

                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                    ):
                        ray_acts, _, _ = model._encode_ray_map_all_layers(
                            ray_sel, true_shape_ray
                        )

                    ray_feats = torch.stack(
                        [act.mean(dim=1).float() for act in ray_acts], dim=0
                    )
                    _update_stats(ray_global, ray_feats)
                    if view_idx in anchor_views:
                        if view_idx not in ray_by_view:
                            ray_by_view[view_idx] = _empty_stats()
                        _update_stats(ray_by_view[view_idx], ray_feats)
                    _append_to_bank(ray_bank, ray_feats)

                if budget_images >= max_images:
                    break

            if budget_images >= max_images:
                break

    image_global_final = _finalize_stats(image_global)
    ray_global_final = _finalize_stats(ray_global)
    merged_global_final = _finalize_stats(merged_global)
    if image_global_final is None and ray_global_final is None and merged_global_final is None:
        raise RuntimeError("No valid images were processed for layer similarity.")

    image_by_view_final = {}
    for k, v in image_by_view.items():
        fin = _finalize_stats(v)
        if fin is not None:
            image_by_view_final[f"view_{k}"] = fin

    ray_by_view_final = {}
    for k, v in ray_by_view.items():
        fin = _finalize_stats(v)
        if fin is not None:
            ray_by_view_final[f"view_{k}"] = fin

    merged_by_view_final = {}
    for k, v in merged_by_view.items():
        fin = _finalize_stats(v)
        if fin is not None:
            merged_by_view_final[f"view_{k}"] = fin

    drift = None
    if drift_mse_sum is not None and drift_count > 0:
        drift = {
            "mse": (drift_mse_sum / drift_count).numpy(),
            "cosine": (drift_cos_sum / drift_count).numpy(),
            "count": int(drift_count),
        }

    return {
        "image_global": image_global_final,
        "ray_global": ray_global_final,
        "merged_global": merged_global_final,
        "image_by_view": image_by_view_final,
        "ray_by_view": ray_by_view_final,
        "merged_by_view": merged_by_view_final,
        "image_bank": _finalize_bank(image_bank),
        "ray_bank": _finalize_bank(ray_bank),
        "merged_bank": _finalize_bank(merged_bank),
        "drift": drift,
        "budget_images": int(budget_images),
    }


def _save_result_block(out_dir: Path, prefix: str, result):
    if result is None:
        return
    np.save(out_dir / f"{prefix}_cosine_distance.npy", result["cosine"])
    np.save(out_dir / f"{prefix}_mse_distance.npy", result["mse"])
    np.save(out_dir / f"{prefix}_layer_norms.npy", result["norms"])
    _save_heatmap(
        result["cosine"],
        f"{prefix}: Cosine Distance",
        out_dir / f"{prefix}_cosine_distance.png",
    )
    _save_heatmap(
        result["mse"],
        f"{prefix}: MSE Distance",
        out_dir / f"{prefix}_mse_distance.png",
    )


def main():
    args = parse_args()

    cfg_path = Path(args.config_dir) / f"{args.config_name}.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.resolve(cfg)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = eval(cfg.model)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidate = Path(str(cfg.output_dir)) / "checkpoint-last.pth"
        if candidate.exists():
            checkpoint_path = str(candidate)
    if checkpoint_path is None or (not Path(checkpoint_path).exists()):
        print(
            "[error] No trained checkpoint found. "
            "You have not trained any model yet (or checkpoint path is wrong).\n"
            "Please train first, then rerun with --checkpoint /path/to/checkpoint.pth"
        )
        return
    _load_checkpoint_if_any(model, checkpoint_path, device)
    model.to(device)

    dataset_expr = str(cfg.test_dataset if args.split == "test" else cfg.train_dataset)
    dataset_expr = _first_dataset_expr(dataset_expr)

    batch_size = args.batch_size if args.batch_size is not None else int(cfg.batch_size)
    num_workers = (
        args.num_workers if args.num_workers is not None else int(cfg.num_workers)
    )

    loader = get_data_loader(
        dataset_expr,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=False,
        drop_last=False,
        accelerator=None,
        fixed_length=True,
    )

    # Some CUT3R datasets require setting epoch before __getitem__.
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(0)
    if (
        hasattr(loader, "batch_sampler")
        and hasattr(loader.batch_sampler, "batch_sampler")
        and hasattr(loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        loader.batch_sampler.batch_sampler.set_epoch(0)

    results = compute_layer_similarity(
        model,
        loader,
        device,
        max_images=args.max_images,
        use_amp=(not args.no_amp),
    )

    default_out = Path(str(cfg.output_dir)) / "interpretability" / f"layer_similarity_{args.split}"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    _save_result_block(images_dir, "image_global", results["image_global"])
    _save_result_block(images_dir, "ray_global", results["ray_global"])
    _save_result_block(images_dir, "merged_global", results["merged_global"])
    for view_name, view_result in results["image_by_view"].items():
        _save_result_block(images_dir, f"image_{view_name}", view_result)
    for view_name, view_result in results["ray_by_view"].items():
        _save_result_block(images_dir, f"ray_{view_name}", view_result)
    for view_name, view_result in results["merged_by_view"].items():
        _save_result_block(images_dir, f"merged_{view_name}", view_result)

    if args.with_cka:
        image_cka = _cka_matrix(results["image_bank"])
        ray_cka = _cka_matrix(results["ray_bank"])
        merged_cka = _cka_matrix(results["merged_bank"])
        if image_cka is not None:
            np.save(out_dir / "image_global_cka.npy", image_cka)
            _save_heatmap(image_cka, "image_global: CKA", out_dir / "image_global_cka.png")
        if ray_cka is not None:
            np.save(out_dir / "ray_global_cka.npy", ray_cka)
            _save_heatmap(ray_cka, "ray_global: CKA", out_dir / "ray_global_cka.png")
        if merged_cka is not None:
            np.save(out_dir / "merged_global_cka.npy", merged_cka)
            _save_heatmap(merged_cka, "merged_global: CKA", out_dir / "merged_global_cka.png")
    else:
        image_cka = ray_cka = merged_cka = None

    if results["drift"] is not None:
        np.save(images_dir / "temporal_drift_mse.npy", results["drift"]["mse"])
        np.save(images_dir / "temporal_drift_cosine.npy", results["drift"]["cosine"])
        _save_drift_heatmap(
            results["drift"]["mse"],
            "Temporal Drift (MSE): layer l at view t vs t+1",
            images_dir / "temporal_drift_mse.png",
        )
        _save_drift_heatmap(
            results["drift"]["cosine"],
            "Temporal Drift (Cosine): layer l at view t vs t+1",
            images_dir / "temporal_drift_cosine.png",
        )

    image_global_layers = (
        int(results["image_global"]["cosine"].shape[0])
        if results["image_global"] is not None
        else None
    )
    ray_global_layers = (
        int(results["ray_global"]["cosine"].shape[0])
        if results["ray_global"] is not None
        else None
    )
    merged_global_layers = (
        int(results["merged_global"]["cosine"].shape[0])
        if results["merged_global"] is not None
        else None
    )

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": args.config_name,
        "config_path": str(cfg_path),
        "split": args.split,
        "dataset_expr": dataset_expr,
        "checkpoint": checkpoint_path,
        "device": str(device),
        "budget_images": int(results["budget_images"]),
        "image_global_count": (
            int(results["image_global"]["count"])
            if results["image_global"] is not None
            else 0
        ),
        "ray_global_count": (
            int(results["ray_global"]["count"]) if results["ray_global"] is not None else 0
        ),
        "image_num_layers": image_global_layers,
        "ray_num_layers": ray_global_layers,
        "merged_num_layers": merged_global_layers,
        "image_views": list(results["image_by_view"].keys()),
        "ray_views": list(results["ray_by_view"].keys()),
        "merged_views": list(results["merged_by_view"].keys()),
        "cka": bool(args.with_cka),
        "drift_count": int(results["drift"]["count"]) if results["drift"] is not None else 0,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _write_report(
        out_dir,
        {
            "image_global": {
                "cosine": results["image_global"]["cosine"] if results["image_global"] is not None else None,
                "cka": image_cka,
            }
            if results["image_global"] is not None
            else None,
            "ray_global": {
                "cosine": results["ray_global"]["cosine"] if results["ray_global"] is not None else None,
                "cka": ray_cka,
            }
            if results["ray_global"] is not None
            else None,
            "merged_global": {
                "cosine": results["merged_global"]["cosine"] if results["merged_global"] is not None else None,
                "cka": merged_cka,
            }
            if results["merged_global"] is not None
            else None,
        },
        topk=args.report_topk,
    )

    print(f"Saved results to: {out_dir}")
    print(
        f"budget_images={results['budget_images']}, "
        f"image_count={meta['image_global_count']}, ray_count={meta['ray_global_count']}"
    )


if __name__ == "__main__":
    main()
