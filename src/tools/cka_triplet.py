#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
from omegaconf import OmegaConf

from dust3r.datasets import get_data_loader
from dust3r.model import ARCroco3DStereo, ARCroco3DStereoConfig, inf, strip_module  # noqa: F401
from dust3r.utils.safe_eval import safe_eval_expr
import dust3r.utils.path_to_croco  # noqa: F401

from tools.layer_similarity_encoder import compute_layer_similarity, _cka_matrix
from tools.layer_similarity_decoder import compute_decoder_similarity


def parse_args():
    parser = argparse.ArgumentParser("CUT3R CKA triplet visualizer")
    parser.add_argument("--config-dir", type=str, default="config")
    parser.add_argument(
        "--config-name",
        type=str,
        default="dualdpt_3blocks_2day_aggressive",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--dataset-expr",
        type=str,
        default=None,
        help="Optional explicit dataset expression. If omitted, use first dataset from config split.",
    )
    parser.add_argument(
        "--plot-style",
        type=str,
        default="default",
        choices=["default", "paper"],
        help="Plot rendering style. 'paper' uses cleaner labels and no in-cell text by default.",
    )
    parser.add_argument(
        "--separate-scales",
        action="store_true",
        help="Use per-matrix auto color scales instead of a shared [0, 1] range.",
    )
    parser.add_argument(
        "--auto-vmin-quantile",
        type=float,
        default=0.05,
        help="If --separate-scales is set, use this quantile of off-diagonal values as vmin.",
    )
    parser.add_argument("--encoder-vmin", type=float, default=None)
    parser.add_argument("--encoder-vmax", type=float, default=None)
    parser.add_argument("--decoder-readout-vmin", type=float, default=None)
    parser.add_argument("--decoder-readout-vmax", type=float, default=None)
    parser.add_argument("--decoder-update-vmin", type=float, default=None)
    parser.add_argument("--decoder-update-vmax", type=float, default=None)
    parser.add_argument(
        "--annotate-values",
        action="store_true",
        help="Force per-cell numeric annotations.",
    )
    parser.add_argument(
        "--no-annotate-values",
        action="store_true",
        help="Disable per-cell numeric annotations.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default=None,
        help="Matplotlib colormap name to use for all CKA heatmaps.",
    )
    return parser.parse_args()


def _first_dataset_expr(expr: str) -> str:
    return expr.split("+")[0].strip()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _instantiate_model(expr: str):
    namespace = {
        "ARCroco3DStereo": ARCroco3DStereo,
        "ARCroco3DStereoConfig": ARCroco3DStereoConfig,
        "inf": inf,
    }
    return safe_eval_expr(expr, namespace, context="model")


def _load_model_from_checkpoint(cfg_model_expr: str, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_expr = cfg_model_expr
    if isinstance(ckpt, dict):
        ckpt_args = ckpt.get("args", None)
        ckpt_model_expr = getattr(ckpt_args, "model", None)
        if isinstance(ckpt_model_expr, str) and ckpt_model_expr:
            model_expr = ckpt_model_expr

    model = _instantiate_model(str(model_expr))
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format")
    msg = model.load_state_dict(strip_module(state_dict), strict=False)
    print(msg)
    model.to(device)
    model.eval()
    return model


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return matrix.reshape(-1)
    idx = np.triu_indices(matrix.shape[0], k=1)
    vals = matrix[idx]
    return vals if vals.size > 0 else matrix.reshape(-1)


def _auto_vmin_from_quantile(matrix: np.ndarray, quantile: float) -> float:
    vals = _offdiag_values(matrix)
    if vals.size == 0:
        return 0.0
    q = float(np.clip(quantile, 0.0, 1.0))
    vmin = float(np.quantile(vals, q))
    return float(np.clip(vmin, 0.0, 1.0))


def _save_cka_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: Path,
    *,
    vmin: float,
    vmax: float,
    annotate_values: bool,
    plot_style: str,
    cmap: str,
):
    import matplotlib.pyplot as plt

    L = matrix.shape[0]
    if plot_style == "paper":
        fig = plt.figure(figsize=(max(6.8, 0.62 * L), max(5.8, 0.58 * L)))
        title_size = 18
        label_size = 14
        tick_size = 11
        value_font_size = 9
        dpi = 260
    else:
        fig = plt.figure(figsize=(max(7, 0.65 * L), max(6, 0.6 * L)))
        title_size = 15
        label_size = 13
        tick_size = 11
        value_font_size = 9
        dpi = 180

    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=title_size)
    ax.set_xlabel("Layer", fontsize=label_size)
    ax.set_ylabel("Layer", fontsize=label_size)
    ax.set_xticks(np.arange(L))
    ax.set_yticks(np.arange(L))
    ax.tick_params(axis="both", labelsize=tick_size)

    if annotate_values and L <= 20:
        mid = 0.5 * (vmin + vmax)
        for i in range(L):
            for j in range(L):
                val = matrix[i, j]
                text_color = "black" if val >= mid else "white"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}",
                    ha="center",
                    va="center",
                    fontsize=value_font_size,
                    color=text_color,
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("CKA", fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def _build_loader(cfg, split: str, dataset_expr: str | None, batch_size: int | None, num_workers: int | None):
    if dataset_expr is None:
        source_expr = str(cfg.test_dataset if split == "test" else cfg.train_dataset)
        dataset_expr = _first_dataset_expr(source_expr)

    final_batch_size = int(batch_size if batch_size is not None else cfg.batch_size)
    final_num_workers = int(num_workers if num_workers is not None else cfg.num_workers)

    loader = get_data_loader(
        dataset_expr,
        batch_size=final_batch_size,
        num_workers=final_num_workers,
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

    return loader, dataset_expr


def main():
    args = parse_args()

    if args.annotate_values and args.no_annotate_values:
        raise ValueError("Use only one of --annotate-values or --no-annotate-values")
    if not (0.0 <= args.auto_vmin_quantile <= 1.0):
        raise ValueError("--auto-vmin-quantile must be in [0, 1]")

    cfg_path = Path(args.config_dir) / f"{args.config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(str(cfg_path))
    OmegaConf.resolve(cfg)

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    model = _load_model_from_checkpoint(str(cfg.model), checkpoint_path, device)

    loader, dataset_expr = _build_loader(
        cfg,
        split=args.split,
        dataset_expr=args.dataset_expr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    enc = compute_layer_similarity(
        model,
        loader,
        device,
        max_images=args.max_images,
        use_amp=(not args.no_amp),
    )
    dec = compute_decoder_similarity(
        model,
        loader,
        device,
        max_images=args.max_images,
        use_amp=(not args.no_amp),
    )

    encoder_cka = _cka_matrix(enc["image_bank"])
    decoder_readout_cka = _cka_matrix(dec["decoder_image_bank"])
    decoder_update_cka = _cka_matrix(dec["decoder_state_bank"])

    if args.annotate_values:
        annotate_values = True
    elif args.no_annotate_values:
        annotate_values = False
    else:
        annotate_values = args.plot_style != "paper"

    cmap = args.cmap if args.cmap else ("magma" if args.plot_style == "paper" else "viridis")

    def resolve_bounds(name: str, matrix: np.ndarray):
        if name == "encoder":
            vmin = args.encoder_vmin
            vmax = args.encoder_vmax
        elif name == "decoder_readout":
            vmin = args.decoder_readout_vmin
            vmax = args.decoder_readout_vmax
        elif name == "decoder_update":
            vmin = args.decoder_update_vmin
            vmax = args.decoder_update_vmax
        else:
            raise ValueError(f"Unknown matrix name: {name}")

        if args.separate_scales:
            if vmin is None:
                vmin = _auto_vmin_from_quantile(matrix, args.auto_vmin_quantile)
            if vmax is None:
                vmax = 1.0
        else:
            if vmin is None:
                vmin = 0.0
            if vmax is None:
                vmax = 1.0

        if vmax <= vmin:
            raise ValueError(f"Invalid bounds for {name}: vmin={vmin}, vmax={vmax}")

        return float(vmin), float(vmax)

    default_out = (
        Path(str(cfg.output_dir))
        / "interpretability"
        / f"cka_triplet_{args.split}"
    )
    out_dir = Path(args.output_dir) if args.output_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    if encoder_cka is not None:
        encoder_vmin, encoder_vmax = resolve_bounds("encoder", encoder_cka)
        np.save(out_dir / "encoder_cka.npy", encoder_cka)
        _save_cka_heatmap(
            encoder_cka,
            "Encoder Layer CKA",
            out_dir / "encoder_cka.png",
            vmin=encoder_vmin,
            vmax=encoder_vmax,
            annotate_values=annotate_values,
            plot_style=args.plot_style,
            cmap=cmap,
        )
    else:
        encoder_vmin = None
        encoder_vmax = None
    if decoder_readout_cka is not None:
        decoder_readout_vmin, decoder_readout_vmax = resolve_bounds("decoder_readout", decoder_readout_cka)
        np.save(out_dir / "decoder_readout_cka.npy", decoder_readout_cka)
        _save_cka_heatmap(
            decoder_readout_cka,
            "Decoder Readout Layer CKA",
            out_dir / "decoder_readout_cka.png",
            vmin=decoder_readout_vmin,
            vmax=decoder_readout_vmax,
            annotate_values=annotate_values,
            plot_style=args.plot_style,
            cmap=cmap,
        )
    else:
        decoder_readout_vmin = None
        decoder_readout_vmax = None
    if decoder_update_cka is not None:
        decoder_update_vmin, decoder_update_vmax = resolve_bounds("decoder_update", decoder_update_cka)
        np.save(out_dir / "decoder_update_cka.npy", decoder_update_cka)
        _save_cka_heatmap(
            decoder_update_cka,
            "Decoder Update Layer CKA",
            out_dir / "decoder_update_cka.png",
            vmin=decoder_update_vmin,
            vmax=decoder_update_vmax,
            annotate_values=annotate_values,
            plot_style=args.plot_style,
            cmap=cmap,
        )
    else:
        decoder_update_vmin = None
        decoder_update_vmax = None

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config_name": args.config_name,
        "config_path": str(cfg_path.resolve()),
        "split": args.split,
        "dataset_expr": dataset_expr,
        "checkpoint": str(checkpoint_path.resolve()),
        "device": str(device),
        "max_images": int(args.max_images),
        "encoder_layers": int(encoder_cka.shape[0]) if encoder_cka is not None else None,
        "decoder_readout_layers": int(decoder_readout_cka.shape[0]) if decoder_readout_cka is not None else None,
        "decoder_update_layers": int(decoder_update_cka.shape[0]) if decoder_update_cka is not None else None,
        "encoder_count": int(enc["image_global"]["count"]) if enc["image_global"] is not None else 0,
        "decoder_readout_count": int(dec["decoder_image_global"]["count"]) if dec["decoder_image_global"] is not None else 0,
        "decoder_update_count": int(dec["decoder_state_global"]["count"]) if dec["decoder_state_global"] is not None else 0,
        "plot": {
            "style": args.plot_style,
            "cmap": cmap,
            "annotate_values": bool(annotate_values),
            "separate_scales": bool(args.separate_scales),
            "auto_vmin_quantile": float(args.auto_vmin_quantile),
            "encoder": {"vmin": encoder_vmin, "vmax": encoder_vmax},
            "decoder_readout": {"vmin": decoder_readout_vmin, "vmax": decoder_readout_vmax},
            "decoder_update": {"vmin": decoder_update_vmin, "vmax": decoder_update_vmax},
        },
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved triplet CKA results to: {out_dir}")
    print(
        f"encoder_layers={meta['encoder_layers']}, "
        f"decoder_readout_layers={meta['decoder_readout_layers']}, "
        f"decoder_update_layers={meta['decoder_update_layers']}"
    )


if __name__ == "__main__":
    main()
