#!/usr/bin/env python3
import argparse
import datetime
import gc
import json
import os
import sys
from pathlib import Path

# Ensure `src/` is importable when this file is run as:
#   python tools/layer_similarity_decoder.py
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
    parser = argparse.ArgumentParser("CUT3R decoder layer similarity")
    parser.add_argument("--config-dir", type=str, default="../config")
    parser.add_argument("--config-name", type=str, default="stage4")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--max-images", type=int, default=256)
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
    args.with_cka = True
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
    lines = ["# CUT3R Decoder Layer Similarity Report", ""]
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

            trend = _early_mid_late_trend(cosine)
            lines.append("### Progression trend (early/mid/late)")
            if trend is None:
                lines.append("- not enough layers")
            else:
                lines.append(
                    f"- mean off-diagonal cosine distance: early={trend['early']:.6f}, mid={trend['mid']:.6f}, late={trend['late']:.6f}"
                )
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


def _pool_decoder_layers(model, decoder_layers, base_tokens):
    image_layers = decoder_layers["image_layers"]
    state_layers = decoder_layers["state_layers"]

    pooled_img = []
    for layer in image_layers:
        feat = layer
        if feat.shape[1] == base_tokens + 1:
            feat = feat[:, 1:]
        if feat.shape[-1] != model.dec_embed_dim:
            feat = model.decoder_embed(feat)
        pooled_img.append(feat.mean(dim=1).float())

    pooled_state = [layer.mean(dim=1).float() for layer in state_layers]

    return torch.stack(pooled_img, dim=0), torch.stack(pooled_state, dim=0)


def compute_decoder_similarity(model, loader, device, max_images=256, use_amp=True):
    model.eval()

    dec_image_global = _empty_stats()
    dec_state_global = _empty_stats()
    dec_image_by_view = {}
    dec_state_by_view = {}
    dec_image_bank = {"layers": None}
    dec_state_bank = {"layers": None}

    drift_img_mse_sum = None
    drift_img_cos_sum = None
    drift_state_mse_sum = None
    drift_state_cos_sum = None
    drift_count = 0

    budget_images = 0
    amp_enabled = use_amp and (device.type == "cuda")

    with torch.no_grad():
        for batch in loader:
            batch = _to_device_batch(batch, device)
            num_views = len(batch)
            anchor_views = sorted(set([0, num_views // 2, num_views - 1]))

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                (feat, pos, shape), (
                    init_state_feat,
                    init_mem,
                    state_feat,
                    state_pos,
                    mem,
                ) = model._forward_encoder(batch)

            dec_img_per_view = []
            dec_state_per_view = []

            for i in range(num_views):
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    _, (state_feat, mem), dec_layers = model._forward_decoder_step(
                        batch,
                        i,
                        feat_i=feat[i],
                        pos_i=pos[i],
                        shape_i=shape[i],
                        init_state_feat=init_state_feat,
                        init_mem=init_mem,
                        state_feat=state_feat,
                        state_pos=state_pos,
                        mem=mem,
                        return_decoder_layers=True,
                    )

                dec_img_feats, dec_state_feats = _pool_decoder_layers(
                    model, dec_layers, base_tokens=feat[i].shape[1]
                )

                dec_img_per_view.append(dec_img_feats)
                dec_state_per_view.append(dec_state_feats)

                _update_stats(dec_image_global, dec_img_feats)
                _update_stats(dec_state_global, dec_state_feats)
                _append_to_bank(dec_image_bank, dec_img_feats)
                _append_to_bank(dec_state_bank, dec_state_feats)

                if i in anchor_views:
                    if i not in dec_image_by_view:
                        dec_image_by_view[i] = _empty_stats()
                    if i not in dec_state_by_view:
                        dec_state_by_view[i] = _empty_stats()
                    _update_stats(dec_image_by_view[i], dec_img_feats)
                    _update_stats(dec_state_by_view[i], dec_state_feats)

            if len(dec_img_per_view) > 1:
                L_img = dec_img_per_view[0].shape[0]
                L_state = dec_state_per_view[0].shape[0]
                if drift_img_mse_sum is None:
                    drift_img_mse_sum = torch.zeros(num_views - 1, L_img)
                    drift_img_cos_sum = torch.zeros(num_views - 1, L_img)
                    drift_state_mse_sum = torch.zeros(num_views - 1, L_state)
                    drift_state_cos_sum = torch.zeros(num_views - 1, L_state)

                for t in range(num_views - 1):
                    a_img = dec_img_per_view[t]
                    b_img = dec_img_per_view[t + 1]
                    img_mse = ((a_img - b_img).pow(2)).mean(dim=-1).mean(dim=-1)
                    img_cos = 1.0 - F.cosine_similarity(a_img, b_img, dim=-1, eps=1e-6).mean(dim=-1)
                    drift_img_mse_sum[t] += img_mse.cpu()
                    drift_img_cos_sum[t] += img_cos.cpu()

                    a_state = dec_state_per_view[t]
                    b_state = dec_state_per_view[t + 1]
                    state_mse = ((a_state - b_state).pow(2)).mean(dim=-1).mean(dim=-1)
                    state_cos = 1.0 - F.cosine_similarity(a_state, b_state, dim=-1, eps=1e-6).mean(dim=-1)
                    drift_state_mse_sum[t] += state_mse.cpu()
                    drift_state_cos_sum[t] += state_cos.cpu()

                drift_count += 1

            budget_images += int(batch[0]["img"].shape[0])
            if budget_images >= max_images:
                break

    dec_image_global_final = _finalize_stats(dec_image_global)
    dec_state_global_final = _finalize_stats(dec_state_global)
    if dec_image_global_final is None and dec_state_global_final is None:
        raise RuntimeError("No valid samples were processed for decoder layer similarity.")

    dec_image_by_view_final = {}
    for k, v in dec_image_by_view.items():
        fin = _finalize_stats(v)
        if fin is not None:
            dec_image_by_view_final[f"view_{k}"] = fin

    dec_state_by_view_final = {}
    for k, v in dec_state_by_view.items():
        fin = _finalize_stats(v)
        if fin is not None:
            dec_state_by_view_final[f"view_{k}"] = fin

    drift = None
    if drift_img_mse_sum is not None and drift_count > 0:
        drift = {
            "image_mse": (drift_img_mse_sum / drift_count).numpy(),
            "image_cosine": (drift_img_cos_sum / drift_count).numpy(),
            "state_mse": (drift_state_mse_sum / drift_count).numpy(),
            "state_cosine": (drift_state_cos_sum / drift_count).numpy(),
            "count": int(drift_count),
        }

    return {
        "decoder_image_global": dec_image_global_final,
        "decoder_state_global": dec_state_global_final,
        "decoder_image_by_view": dec_image_by_view_final,
        "decoder_state_by_view": dec_state_by_view_final,
        "decoder_image_bank": _finalize_bank(dec_image_bank),
        "decoder_state_bank": _finalize_bank(dec_state_bank),
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

    if hasattr(loader, "dataset") and hasattr(loader.dataset, "set_epoch"):
        loader.dataset.set_epoch(0)
    if (
        hasattr(loader, "batch_sampler")
        and hasattr(loader.batch_sampler, "batch_sampler")
        and hasattr(loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        loader.batch_sampler.batch_sampler.set_epoch(0)

    results = compute_decoder_similarity(
        model,
        loader,
        device,
        max_images=args.max_images,
        use_amp=(not args.no_amp),
    )

    default_out = Path(str(cfg.output_dir)) / "interpretability" / f"layer_similarity_decoder_{args.split}"
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    _save_result_block(images_dir, "decoder_image_global", results["decoder_image_global"])
    _save_result_block(images_dir, "decoder_state_global", results["decoder_state_global"])
    for view_name, view_result in results["decoder_image_by_view"].items():
        _save_result_block(images_dir, f"decoder_image_{view_name}", view_result)
    for view_name, view_result in results["decoder_state_by_view"].items():
        _save_result_block(images_dir, f"decoder_state_{view_name}", view_result)

    if args.with_cka:
        dec_img_cka = _cka_matrix(results["decoder_image_bank"])
        dec_state_cka = _cka_matrix(results["decoder_state_bank"])
        if dec_img_cka is not None:
            np.save(out_dir / "decoder_image_global_cka.npy", dec_img_cka)
            _save_heatmap(dec_img_cka, "decoder_image_global: CKA", out_dir / "decoder_image_global_cka.png")
        if dec_state_cka is not None:
            np.save(out_dir / "decoder_state_global_cka.npy", dec_state_cka)
            _save_heatmap(dec_state_cka, "decoder_state_global: CKA", out_dir / "decoder_state_global_cka.png")
    else:
        dec_img_cka = dec_state_cka = None

    if results["drift"] is not None:
        np.save(images_dir / "decoder_temporal_drift_image_mse.npy", results["drift"]["image_mse"])
        np.save(images_dir / "decoder_temporal_drift_image_cosine.npy", results["drift"]["image_cosine"])
        np.save(images_dir / "decoder_temporal_drift_state_mse.npy", results["drift"]["state_mse"])
        np.save(images_dir / "decoder_temporal_drift_state_cosine.npy", results["drift"]["state_cosine"])
        _save_drift_heatmap(
            results["drift"]["image_mse"],
            "Decoder Temporal Drift (Image, MSE): layer l at view t vs t+1",
            images_dir / "decoder_temporal_drift_image_mse.png",
        )
        _save_drift_heatmap(
            results["drift"]["image_cosine"],
            "Decoder Temporal Drift (Image, Cosine): layer l at view t vs t+1",
            images_dir / "decoder_temporal_drift_image_cosine.png",
        )
        _save_drift_heatmap(
            results["drift"]["state_mse"],
            "Decoder Temporal Drift (State, MSE): layer l at view t vs t+1",
            images_dir / "decoder_temporal_drift_state_mse.png",
        )
        _save_drift_heatmap(
            results["drift"]["state_cosine"],
            "Decoder Temporal Drift (State, Cosine): layer l at view t vs t+1",
            images_dir / "decoder_temporal_drift_state_cosine.png",
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
        "decoder_image_global_count": (
            int(results["decoder_image_global"]["count"])
            if results["decoder_image_global"] is not None
            else 0
        ),
        "decoder_state_global_count": (
            int(results["decoder_state_global"]["count"])
            if results["decoder_state_global"] is not None
            else 0
        ),
        "decoder_image_num_layers": (
            int(results["decoder_image_global"]["cosine"].shape[0])
            if results["decoder_image_global"] is not None
            else None
        ),
        "decoder_state_num_layers": (
            int(results["decoder_state_global"]["cosine"].shape[0])
            if results["decoder_state_global"] is not None
            else None
        ),
        "decoder_image_views": list(results["decoder_image_by_view"].keys()),
        "decoder_state_views": list(results["decoder_state_by_view"].keys()),
        "cka": bool(args.with_cka),
        "drift_count": int(results["drift"]["count"]) if results["drift"] is not None else 0,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _write_report(
        out_dir,
        {
            "decoder_image_global": {
                "cosine": results["decoder_image_global"]["cosine"] if results["decoder_image_global"] is not None else None,
                "cka": dec_img_cka,
            }
            if results["decoder_image_global"] is not None
            else None,
            "decoder_state_global": {
                "cosine": results["decoder_state_global"]["cosine"] if results["decoder_state_global"] is not None else None,
                "cka": dec_state_cka,
            }
            if results["decoder_state_global"] is not None
            else None,
        },
        topk=args.report_topk,
    )

    print(f"Saved results to: {out_dir}")
    print(
        f"budget_images={results['budget_images']}, "
        f"decoder_image_count={meta['decoder_image_global_count']}, "
        f"decoder_state_count={meta['decoder_state_global_count']}"
    )


if __name__ == "__main__":
    main()
