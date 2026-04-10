# --------------------------------------------------------
# training code for CUT3R
# --------------------------------------------------------
# References:
# DUSt3R: https://github.com/naver/dust3r
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import re
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

from dust3r.model import (
    PreTrainedModel,
    ARCroco3DStereo,
    ARCroco3DStereoConfig,
    inf,
    strip_module,
)  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader
from dust3r.losses import *  # noqa: F401, needed when loading the model
import dust3r.losses as dust3r_losses
from dust3r.inference import (  # noqa
    loss_of_one_batch,
    loss_of_one_batch_tbptt,
    loss_of_one_batch_encoder,
)
from dust3r.viz import colorize
from dust3r.utils.render import get_render_results
import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa

import hydra
from omegaconf import OmegaConf
import logging
import pathlib
from tqdm import tqdm
import random
import builtins
import shutil

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datetime import timedelta
import torch.multiprocessing
from dust3r.utils.safe_eval import safe_eval_expr

torch.multiprocessing.set_sharing_strategy("file_system")

printer = get_logger(__name__, log_level="DEBUG")


def _build_expr_namespace():
    namespace = {
        "ARCroco3DStereo": ARCroco3DStereo,
        "ARCroco3DStereoConfig": ARCroco3DStereoConfig,
        "inf": inf,
    }
    for name, value in vars(dust3r_losses).items():
        if name.startswith("_"):
            continue
        if callable(value) or isinstance(value, (int, float, bool, str)):
            namespace[name] = value
    return namespace


EXPR_NAMESPACE = _build_expr_namespace()


VARIANT_ORIGINAL_CUT3R = "original_cut3r"
VARIANT_LOOP_NEW = "loop_new"
_HEAD_TYPE_RE = re.compile(r"head_type\s*=\s*'([^']+)'")

_VARIANT_PROFILES = {
    VARIANT_ORIGINAL_CUT3R: {
        "head_type": "linear",
        "train_criterion": "ConfLoss(Regr3DPose(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)",
        "test_criterion": "Regr3DPose(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + Regr3DPose_ScaleInv(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + RGBLoss(L21)",
    },
    VARIANT_LOOP_NEW: {
        "head_type": "dualdpt",
        "train_criterion": "ConfLoss(Regr3DPose(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE) + 0.1 * RayConsistencyLoss()",
        "test_criterion": "Regr3DPose(L21, norm_mode='?avg_dis', gt_scale=True, sky_loss_value=0) + RGBLoss(L21)",
    },
}

_VARIANT_ALIASES = {
    "original": VARIANT_ORIGINAL_CUT3R,
    "cut3r": VARIANT_ORIGINAL_CUT3R,
    "original_cut3r": VARIANT_ORIGINAL_CUT3R,
    "loop": VARIANT_LOOP_NEW,
    "loop_new": VARIANT_LOOP_NEW,
    "dualdpt": VARIANT_LOOP_NEW,
    "new": VARIANT_LOOP_NEW,
}


def _normalize_variant_name(raw_variant):
    if raw_variant is None:
        return None
    name = str(raw_variant).strip().lower()
    if not name or name in {"none", "default", "config"}:
        return None
    if name in _VARIANT_ALIASES:
        return _VARIANT_ALIASES[name]
    supported = ", ".join(sorted(_VARIANT_PROFILES.keys()))
    raise ValueError(
        f"Unknown variant '{raw_variant}'. Supported variants: {supported}"
    )


def _extract_head_type(model_expr: str):
    match = _HEAD_TYPE_RE.search(str(model_expr))
    return match.group(1) if match else None


def _replace_head_type(model_expr: str, new_head_type: str):
    model_expr = str(model_expr)
    if _HEAD_TYPE_RE.search(model_expr) is None:
        raise ValueError(
            "Model expression does not contain head_type='...'; cannot apply variant override."
        )
    return _HEAD_TYPE_RE.sub(
        f"head_type='{new_head_type}'",
        model_expr,
        count=1,
    )


def _apply_variant_profile(args):
    variant = _normalize_variant_name(getattr(args, "variant", None))
    if variant is None:
        setattr(args, "variant", "config_default")
        return "config_default"

    setattr(args, "variant", variant)
    if bool(getattr(args, "encoder_only_training", False)):
        printer.warning(
            "Variant '%s' requested with encoder_only_training=True; skipping head/loss override.",
            variant,
        )
        return variant

    profile = _VARIANT_PROFILES[variant]
    args.model = _replace_head_type(args.model, profile["head_type"])
    args.train_criterion = profile["train_criterion"]
    args.test_criterion = profile["test_criterion"]

    printer.info(
        "Applied variant profile '%s' | head_type=%s",
        variant,
        profile["head_type"],
    )
    printer.info("Resolved train criterion: %s", args.train_criterion)
    printer.info("Resolved test criterion : %s", args.test_criterion)
    return variant


def _resolve_expression(expression: str, *, context: str):
    return safe_eval_expr(expression, EXPR_NAMESPACE, context=context)


def _component_parameters(component):
    if component is None:
        return []
    if isinstance(component, torch.nn.Parameter):
        return [component]
    if isinstance(component, torch.nn.Module):
        return list(component.parameters())
    return []


def estimate_encoder_parameter_counts(model):
    """Estimate CUT3R encoder parameters (total/trainable).

    Includes image/ray patch embed, encoder blocks/norms, and encoder mask tokens.
    """
    encoder_components = [
        "patch_embed",
        "patch_embed_ray_map",
        "masked_img_token",
        "masked_ray_map_token",
        "enc_blocks",
        "enc_blocks_ray_map",
        "enc_norm",
        "enc_norm_ray_map",
    ]
    params = {}
    for name in encoder_components:
        if not hasattr(model, name):
            continue
        for p in _component_parameters(getattr(model, name)):
            params[id(p)] = p

    total = sum(p.numel() for p in params.values())
    trainable = sum(p.numel() for p in params.values() if p.requires_grad)
    return total, trainable


def setup_for_distributed(accelerator: Accelerator):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (accelerator.num_processes > 8)
        if accelerator.is_main_process or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def save_current_code(outdir):
    now = datetime.datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = "."
    dst_dir = os.path.join(outdir, "code", "{}".format(date_time))
    shutil.copytree(
        src_dir,
        dst_dir,
        ignore=shutil.ignore_patterns(
            ".vscode*",
            "assets*",
            "example*",
            "checkpoints*",
            "OLD*",
            "logs*",
            "out*",
            "runs*",
            "*.png",
            "*.mp4",
            "*__pycache__*",
            "*.git*",
            "*.idea*",
            "*.zip",
            "*.jpg",
        ),
        dirs_exist_ok=True,
    )
    return dst_dir


def train(args):
    force_cpu = bool(getattr(args, "force_cpu", False))
    default_mixed_precision = "bf16" if torch.cuda.is_available() else "no"
    mixed_precision = str(getattr(args, "mixed_precision", default_mixed_precision))
    if (force_cpu or not torch.cuda.is_available()) and mixed_precision in {
        "fp16",
        "bf16",
    }:
        printer.info(
            "CPU execution requested or CUDA unavailable: overriding mixed_precision=%s -> no",
            mixed_precision,
        )
        mixed_precision = "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.accum_iter,
        mixed_precision=mixed_precision,
        cpu=force_cpu,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=True),
            InitProcessGroupKwargs(timeout=timedelta(seconds=6000)),
        ],
    )
    device = accelerator.device
    printer.info(
        "Runtime device: %s | force_cpu=%s | cuda_available=%s | mixed_precision=%s",
        device,
        force_cpu,
        torch.cuda.is_available(),
        mixed_precision,
    )

    setup_for_distributed(accelerator)

    printer.info("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "checkpoints").mkdir(parents=True, exist_ok=True)
        (Path(args.output_dir) / "events").mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        dst_dir = save_current_code(outdir=args.output_dir)
        printer.info(f"Saving current code to {dst_dir}")

    # auto resume
    auto_resume = bool(getattr(args, "auto_resume", True))
    if auto_resume and not args.resume:
        last_ckpt_fname = os.path.join(
            args.output_dir, "checkpoints", f"checkpoint-last.pth"
        )
        args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    printer.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))

    # fix the seed
    seed = args.seed + accelerator.state.process_index
    printer.info(
        f"Setting seed to {seed} for process {accelerator.state.process_index}"
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = args.benchmark

    selected_variant = _apply_variant_profile(args)
    printer.info("Variant selection: %s", selected_variant)
    printer.info("Model head_type after variant resolution: %s", _extract_head_type(args.model))

    # training dataset and loader
    printer.info("Building train dataset %s", args.train_dataset)
    #  dataset and loader
    data_loader_train = build_dataset(
        args.train_dataset,
        args.batch_size,
        args.num_workers,
        accelerator=accelerator,
        test=False,
        fixed_length=args.fixed_length
    )
    printer.info("Building test dataset %s", args.test_dataset)
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset,
            args.batch_size,
            args.num_workers,
            accelerator=accelerator,
            test=True,
            fixed_length=True
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    printer.info("Loading model: %s", args.model)
    model: PreTrainedModel = _resolve_expression(args.model, context="model")

    recurrent_ckpt = getattr(args, "recurrent_ckpt", None)
    model_cfg_recurrent_ckpt = getattr(getattr(model, "config", None), "recurrent_ckpt", None)
    should_load_recurrent = (
        recurrent_ckpt
        and hasattr(model, "load_recurrent_backbone")
        and not args.resume
        and not model_cfg_recurrent_ckpt
    )
    if should_load_recurrent:
        recurrent_block_index = int(getattr(args, "recurrent_block_index", 0))
        recurrent_strict = bool(getattr(args, "recurrent_strict", False))
        printer.info(
            "Loading recurrent backbone from %s (block_index=%d, strict=%s)",
            recurrent_ckpt,
            recurrent_block_index,
            recurrent_strict,
        )
        recurrent_info = model.load_recurrent_backbone(
            recurrent_ckpt,
            block_index=recurrent_block_index,
            strict=recurrent_strict,
        )
        printer.info(
            "Recurrent backbone loaded | image=%s | ray=%s",
            recurrent_info["image"]["resolved_path"],
            recurrent_info["ray"]["resolved_path"],
        )
    elif recurrent_ckpt and model_cfg_recurrent_ckpt:
        printer.info(
            "Skipping external recurrent load from args because model config already provides recurrent_ckpt=%s",
            model_cfg_recurrent_ckpt,
        )

    # ── Parameter summary ────────────────────────────────────────────────────
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_total, enc_trainable = estimate_encoder_parameter_counts(model)
    dec_params = sum(p.numel() for p in model.dec_blocks.parameters()) if hasattr(model, 'dec_blocks') else 0
    head_params = sum(p.numel() for n, p in model.named_parameters()
                      if n.startswith('head') or n.startswith('downstream_head'))
    frozen_params = total_params - trainable_params

    printer.info(
        "\n"
        "┌─────────────────────────────────────────────────┐\n"
        "│              MODEL PARAMETER SUMMARY             │\n"
        "├──────────────────────────────────┬──────────────┤\n"
        "│  Total parameters                │ %12s │\n"
        "│  Trainable parameters            │ %12s │\n"
        "│  Frozen parameters               │ %12s │\n"
        "├──────────────────────────────────┼──────────────┤\n"
        "│  Encoder  (total / trainable)    │ %5s / %5s │\n"
        "│  Decoder  (total)                │ %12s │\n"
        "│  Head     (total)                │ %12s │\n"
        "└──────────────────────────────────┴──────────────┘",
        f"{total_params:,}", f"{trainable_params:,}", f"{frozen_params:,}",
        f"{enc_total:,}", f"{enc_trainable:,}",
        f"{dec_params:,}", f"{head_params:,}",
    )
    # ─────────────────────────────────────────────────────────────────────────



    encoder_only_training = getattr(args, "encoder_only_training", False)
    if encoder_only_training:
        encoder_train_criterion = getattr(
            args,
            "encoder_train_criterion",
            "EncoderFeatureConsistencyLoss()",
        )
        encoder_test_criterion = getattr(
            args,
            "encoder_test_criterion",
            encoder_train_criterion,
        )
        printer.info(f">> Creating encoder train criterion = {encoder_train_criterion}")
        train_criterion = _resolve_expression(
            encoder_train_criterion,
            context="encoder train criterion",
        ).to(device)
        printer.info(f">> Creating encoder test criterion = {encoder_test_criterion}")
        test_criterion = _resolve_expression(
            encoder_test_criterion,
            context="encoder test criterion",
        ).to(device)
    else:
        printer.info(f">> Creating train criterion = {args.train_criterion}")
        train_criterion = _resolve_expression(
            args.train_criterion,
            context="train criterion",
        ).to(device)
        printer.info(
            f">> Creating test criterion = {args.test_criterion or args.train_criterion}"
        )
        test_criterion = _resolve_expression(
            args.test_criterion or args.train_criterion,
            context="test criterion",
        ).to(device)

    model.to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if args.long_context:
        model.fixed_input_length = False

    if args.pretrained and not args.resume:
        printer.info(f"Loading pretrained: {args.pretrained}")
        try:
            ckpt = torch.load(
                args.pretrained, map_location=device, weights_only=False
            )
        except TypeError:
            ckpt = torch.load(args.pretrained, map_location=device)
        load_only_encoder = getattr(args, "load_only_encoder", False)
        if load_only_encoder:
            filtered_state_dict = {
                k: v
                for k, v in ckpt["model"].items()
                if "enc_blocks" in k or "patch_embed" in k
            }
            printer.info(
                model.load_state_dict(strip_module(filtered_state_dict), strict=False)
            )
        else:
            printer.info(
                model.load_state_dict(strip_module(ckpt["model"]), strict=False)
            )
        del ckpt  # in case it occupies memory

    # # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model, args.weight_decay)
    optimizer_name = str(getattr(args, "optimizer", "adamw")).lower()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=float(getattr(args, "momentum", 0.9)),
            nesterov=bool(getattr(args, "nesterov", True)),
            foreach=bool(getattr(args, "foreach", False)),
        )
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    loss_scaler = NativeScaler(accelerator=accelerator)

    accelerator.even_batches = False
    optimizer, model, data_loader_train = accelerator.prepare(
        optimizer, model, data_loader_train
    )

    def write_log_stats(epoch, train_stats, test_stats):
        if accelerator.is_main_process:
            if log_writer is not None:
                log_writer.flush()

            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_name in data_loader_test:
                if test_name not in test_stats:
                    continue
                log_stats.update(
                    {test_name + "_" + k: v for k, v in test_stats[test_name].items()}
                )

            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    best_so_far = misc.load_model(
        args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler
    )
    if best_so_far is None:
        best_so_far = float("inf")
    log_writer = (
        SummaryWriter(log_dir=os.path.join(args.output_dir, "events")) if accelerator.is_main_process else None
    )

    printer.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):
        epoch_start_time = time.time()

        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if (
                args.save_freq
                and np.allclose(epoch / args.save_freq, int(epoch / args.save_freq))
                or epoch == args.epochs
            ):
                save_model(epoch - 1, "last", best_so_far)

        # Test on multiple datasets
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                stats = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    accelerator,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=test_name,
                )
                test_stats[test_name] = stats

                # Save best of all
                if stats["loss_med"] < best_so_far:
                    best_so_far = stats["loss_med"]
                    new_best = True
        # Save more stuff
        write_log_stats(epoch, train_stats, test_stats)

        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.epochs:
            break  # exit after writing last test to disk

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            accelerator,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
        )

        epoch_time = time.time() - epoch_start_time
        trained_epochs = epoch - args.start_epoch + 1
        remaining_train_epochs = max(args.epochs - (epoch + 1), 0)
        avg_epoch_time = (time.time() - start_time) / max(trained_epochs, 1)
        eta_seconds = int(avg_epoch_time * remaining_train_epochs)
        eta_finish = datetime.datetime.now() + datetime.timedelta(seconds=eta_seconds)
        printer.info(
            "Epoch %d done in %s | avg/epoch %s | remaining train epochs %d | ETA %s | est finish %s",
            epoch,
            str(datetime.timedelta(seconds=int(epoch_time))),
            str(datetime.timedelta(seconds=int(avg_epoch_time))),
            remaining_train_epochs,
            str(datetime.timedelta(seconds=eta_seconds)),
            eta_finish.strftime("%Y-%m-%d %H:%M:%S"),
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    printer.info("Training time {}".format(total_time_str))

    save_final_model(accelerator, args, args.epochs, model, best_so_far=best_so_far)


def save_final_model(accelerator, args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir) / "checkpoints"
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": (
            model_without_ddp
            if isinstance(model_without_ddp, dict)
            else model_without_ddp.cpu().state_dict()
        ),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    printer.info(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(accelerator, to_save, checkpoint_path)


def build_dataset(dataset, batch_size, num_workers, accelerator, test=False, fixed_length=False):
    split = ["Train", "Test"][test]
    printer.info(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not (test),
        drop_last=not (test),
        accelerator=accelerator,
        fixed_length=fixed_length
    )
    return loader


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    accelerator: Accelerator,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
):
    assert torch.backends.cuda.matmul.allow_tf32 == True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            accelerator=accelerator,
            args=args,
            model_without_ddp=model,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        with accelerator.accumulate(model):
            epoch_f = epoch + data_iter_step / len(data_loader)
            step = int(epoch_f * len(data_loader))
            # we use a per iteration (instead of per epoch) lr scheduler
            if data_iter_step % accum_iter == 0:
                misc.adjust_learning_rate(optimizer, epoch_f, args)
            if getattr(args, "encoder_only_training", False):
                result = loss_of_one_batch_encoder(
                    batch,
                    model,
                    criterion,
                    accelerator,
                    use_amp=bool(args.amp),
                )
            elif not args.long_context:
                result = loss_of_one_batch(
                    batch,
                    model,
                    criterion,
                    accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            else:
                result = loss_of_one_batch_tbptt(
                    batch,
                    model,
                    criterion,
                    chunk_size=4,
                    loss_scaler=loss_scaler,
                    optimizer=optimizer,
                    accelerator=accelerator,
                    symmetrize_batch=False,
                    use_amp=bool(args.amp),
                )
            loss, loss_details = result["loss"]  # criterion returns two values

            loss_value = float(loss)

            if not math.isfinite(loss_value):
                print(
                    f"Loss is {loss_value}, stopping training, loss details: {loss_details}"
                )
                sys.exit(1)
            if not result.get("already_backprop", False):
                should_step = accelerator.sync_gradients
                loss_scaler(
                    loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=should_step,
                    clip_grad=1.0,
                )
                if should_step:
                    optimizer.zero_grad()

            is_metric = batch[0]["is_metric"]
            curr_num_view = len(batch)

            del loss
            tb_vis_img = (data_iter_step + 1) % accum_iter == 0 and (
                (step + 1) % (args.print_img_freq)
            ) == 0
            if not tb_vis_img:
                del batch
            else:
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(epoch=epoch_f)
            metric_logger.update(lr=lr)
            metric_logger.update(step=step)

            metric_logger.update(loss=loss_value, **loss_details)

            if (data_iter_step + 1) % accum_iter == 0 and (
                (data_iter_step + 1) % (accum_iter * args.print_freq)
            ) == 0:
                loss_value_reduce = accelerator.gather(
                    torch.tensor(loss_value).to(accelerator.device)
                ).mean()  # MUST BE EXECUTED BY ALL NODES

                if log_writer is None:
                    continue
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, step)
                log_writer.add_scalar("train_lr", lr, step)
                log_writer.add_scalar("train_iter", epoch_1000x, step)
                for name, val in loss_details.items():
                    if isinstance(val, torch.Tensor):
                        if val.ndim > 0:
                            continue
                    if isinstance(val, dict):
                        continue
                    log_writer.add_scalar("train_" + name, val, step)

            if tb_vis_img and not getattr(args, "encoder_only_training", False):
                if log_writer is None:
                    continue
                with torch.no_grad():
                    depths_self, gt_depths_self = get_render_results(
                        batch, result["pred"], self_view=True
                    )
                    depths_cross, gt_depths_cross = get_render_results(
                        batch, result["pred"], self_view=False
                    )
                    for k in range(len(batch)):
                        loss_details[f"self_pred_depth_{k+1}"] = (
                            depths_self[k].detach().cpu()
                        )
                        loss_details[f"self_gt_depth_{k+1}"] = (
                            gt_depths_self[k].detach().cpu()
                        )
                        loss_details[f"pred_depth_{k+1}"] = (
                            depths_cross[k].detach().cpu()
                        )
                        loss_details[f"gt_depth_{k+1}"] = (
                            gt_depths_cross[k].detach().cpu()
                        )

                imgs_stacked_dict = get_vis_imgs_new(
                    loss_details, args.num_imgs_vis, curr_num_view, is_metric=is_metric
                )
                for name, imgs_stacked in imgs_stacked_dict.items():
                    log_writer.add_images(
                        "train" + "/" + name, imgs_stacked, step, dataformats="HWC"
                    )
                del batch

        if (
            data_iter_step % int(args.save_freq * len(data_loader)) == 0
            and data_iter_step != 0
            and data_iter_step != len(data_loader) - 1
        ):
            print("saving at step", data_iter_step)
            save_model(epoch - 1, "last", float("inf"))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes(accelerator)
    printer.info("Averaged stats: %s", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    accelerator: Accelerator,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
):

    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "Test Epoch: [{}]".format(epoch)

    if log_writer is not None:
        printer.info("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if (
        hasattr(data_loader, "batch_sampler")
        and hasattr(data_loader.batch_sampler, "batch_sampler")
        and hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")
    ):
        data_loader.batch_sampler.batch_sampler.set_epoch(epoch)

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, accelerator, header)
    ):
        if getattr(args, "encoder_only_training", False):
            result = loss_of_one_batch_encoder(
                batch,
                model,
                criterion,
                accelerator,
                use_amp=bool(args.amp),
            )
        else:
            result = loss_of_one_batch(
                batch,
                model,
                criterion,
                accelerator,
                symmetrize_batch=False,
                use_amp=bool(args.amp),
            )

        loss_value, loss_details = result["loss"]  # criterion returns two values
        metric_logger.update(loss=float(loss_value), **loss_details)

    printer.info("Averaged stats: %s", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None and not getattr(args, "encoder_only_training", False):
        for name, val in results.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 0:
                    continue
            if isinstance(val, dict):
                continue
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

        try:
            depths_self, gt_depths_self = get_render_results(
                batch, result["pred"], self_view=True
            )
            depths_cross, gt_depths_cross = get_render_results(
                batch, result["pred"], self_view=False
            )
            for k in range(len(batch)):
                loss_details[f"self_pred_depth_{k+1}"] = depths_self[k].detach().cpu()
                loss_details[f"self_gt_depth_{k+1}"] = gt_depths_self[k].detach().cpu()
                loss_details[f"pred_depth_{k+1}"] = depths_cross[k].detach().cpu()
                loss_details[f"gt_depth_{k+1}"] = gt_depths_cross[k].detach().cpu()
        except Exception as e:
            printer.warning(
                "Skipping render-based eval visualization due to runtime error: %s", e
            )

        required_depth_keys = [
            f"pred_depth_{i+1}" for i in range(args.num_test_views)
        ] + [f"gt_depth_{i+1}" for i in range(args.num_test_views)]
        if all(k in loss_details for k in required_depth_keys):
            imgs_stacked_dict = get_vis_imgs_new(
                loss_details,
                args.num_imgs_vis,
                args.num_test_views,
                is_metric=batch[0]["is_metric"],
            )
            for name, imgs_stacked in imgs_stacked_dict.items():
                log_writer.add_images(
                    prefix + "/" + name, imgs_stacked, 1000 * epoch, dataformats="HWC"
                )
        else:
            printer.warning(
                "Skipping eval image visualization because rendered depth keys are unavailable."
            )

    del loss_details, loss_value, batch
    torch.cuda.empty_cache()

    return results


def batch_append(original_list, new_list):
    for sublist, new_item in zip(original_list, new_list):
        sublist.append(new_item)
    return original_list


def gen_mask_indicator(img_mask_list, ray_mask_list, num_views, h, w):
    output = []
    for img_mask, ray_mask in zip(img_mask_list, ray_mask_list):
        out = torch.zeros((h, w * num_views, 3))
        for i in range(num_views):
            if img_mask[i] and not ray_mask[i]:
                offset = 0
            elif not img_mask[i] and ray_mask[i]:
                offset = 1
            else:
                offset = 0.5
            out[:, i * w : (i + 1) * w] += offset
        output.append(out)
    return output


def vis_and_cat(
    gt_imgs,
    pred_imgs,
    cross_gt_depths,
    cross_pred_depths,
    self_gt_depths,
    self_pred_depths,
    cross_conf,
    self_conf,
    ray_indicator,
    is_metric,
):
    cross_depth_gt_min = torch.quantile(cross_gt_depths, 0.01).item()
    cross_depth_gt_max = torch.quantile(cross_gt_depths, 0.99).item()
    cross_depth_pred_min = torch.quantile(cross_pred_depths, 0.01).item()
    cross_depth_pred_max = torch.quantile(cross_pred_depths, 0.99).item()
    cross_depth_min = min(cross_depth_gt_min, cross_depth_pred_min)
    cross_depth_max = max(cross_depth_gt_max, cross_depth_pred_max)

    cross_gt_depths_vis = colorize(
        cross_gt_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_gt_min, cross_depth_gt_max)
        ),
        append_cbar=True,
    )
    cross_pred_depths_vis = colorize(
        cross_pred_depths,
        range=(
            (cross_depth_min, cross_depth_max)
            if is_metric
            else (cross_depth_pred_min, cross_depth_pred_max)
        ),
        append_cbar=True,
    )

    self_depth_gt_min = torch.quantile(self_gt_depths, 0.01).item()
    self_depth_gt_max = torch.quantile(self_gt_depths, 0.99).item()
    self_depth_pred_min = torch.quantile(self_pred_depths, 0.01).item()
    self_depth_pred_max = torch.quantile(self_pred_depths, 0.99).item()
    self_depth_min = min(self_depth_gt_min, self_depth_pred_min)
    self_depth_max = max(self_depth_gt_max, self_depth_pred_max)

    self_gt_depths_vis = colorize(
        self_gt_depths,
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_gt_min, self_depth_gt_max)
        ),
        append_cbar=True,
    )
    self_pred_depths_vis = colorize(
        self_pred_depths,
        range=(
            (self_depth_min, self_depth_max)
            if is_metric
            else (self_depth_pred_min, self_depth_pred_max)
        ),
        append_cbar=True,
    )
    if len(cross_conf) > 0:
        cross_conf_vis = colorize(cross_conf, append_cbar=True)
    if len(self_conf) > 0:
        self_conf_vis = colorize(self_conf, append_cbar=True)
    gt_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    gt_imgs_vis[: gt_imgs.shape[0], : gt_imgs.shape[1]] = gt_imgs
    pred_imgs_vis = torch.zeros_like(cross_gt_depths_vis)
    pred_imgs_vis[: pred_imgs.shape[0], : pred_imgs.shape[1]] = pred_imgs
    ray_indicator_vis = torch.cat(
        [
            ray_indicator,
            torch.zeros(
                ray_indicator.shape[0],
                cross_pred_depths_vis.shape[1] - ray_indicator.shape[1],
                3,
            ),
        ],
        dim=1,
    )
    out = torch.cat(
        [
            ray_indicator_vis,
            gt_imgs_vis,
            pred_imgs_vis,
            self_gt_depths_vis,
            self_pred_depths_vis,
            self_conf_vis,
            cross_gt_depths_vis,
            cross_pred_depths_vis,
            cross_conf_vis,
        ],
        dim=0,
    )
    return out


def get_vis_imgs_new(loss_details, num_imgs_vis, num_views, is_metric):
    if num_views <= 0 or num_imgs_vis <= 0:
        return {}
    if "gt_img1" not in loss_details or not torch.is_tensor(loss_details["gt_img1"]):
        return {}

    available_imgs = int(loss_details["gt_img1"].shape[0])
    num_imgs_vis = min(int(num_imgs_vis), available_imgs)
    if num_imgs_vis <= 0:
        return {}

    ret_dict = {}
    gt_img_list = [[] for _ in range(num_imgs_vis)]
    pred_img_list = [[] for _ in range(num_imgs_vis)]

    cross_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    cross_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    self_gt_depth_list = [[] for _ in range(num_imgs_vis)]
    self_pred_depth_list = [[] for _ in range(num_imgs_vis)]

    cross_view_conf_list = [[] for _ in range(num_imgs_vis)]
    self_view_conf_list = [[] for _ in range(num_imgs_vis)]
    cross_view_conf_exits = False
    self_view_conf_exits = False

    img_mask_list = [[] for _ in range(num_imgs_vis)]
    ray_mask_list = [[] for _ in range(num_imgs_vis)]

    if num_views > 30:
        stride = 5
    elif num_views > 20:
        stride = 3
    elif num_views > 10:
        stride = 2
    else:
        stride = 1
    for i in range(0, num_views, stride):
        gt_imgs = 0.5 * (loss_details[f"gt_img{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        width = gt_imgs.shape[2]
        pred_imgs = (
            0.5 * (loss_details[f"pred_rgb_{i+1}"] + 1)[:num_imgs_vis].detach().cpu()
        )
        gt_img_list = batch_append(gt_img_list, gt_imgs.unbind(dim=0))
        pred_img_list = batch_append(pred_img_list, pred_imgs.unbind(dim=0))

        cross_pred_depths = (
            loss_details[f"pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        cross_gt_depths = (
            loss_details[f"gt_depth_{i+1}"]
            .to(gt_imgs.device)[:num_imgs_vis]
            .detach()
            .cpu()
        )
        cross_pred_depth_list = batch_append(
            cross_pred_depth_list, cross_pred_depths.unbind(dim=0)
        )
        cross_gt_depth_list = batch_append(
            cross_gt_depth_list, cross_gt_depths.unbind(dim=0)
        )

        self_gt_depths = (
            loss_details[f"self_gt_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_pred_depths = (
            loss_details[f"self_pred_depth_{i+1}"][:num_imgs_vis].detach().cpu()
        )
        self_gt_depth_list = batch_append(
            self_gt_depth_list, self_gt_depths.unbind(dim=0)
        )
        self_pred_depth_list = batch_append(
            self_pred_depth_list, self_pred_depths.unbind(dim=0)
        )

        if f"conf_{i+1}" in loss_details:
            cross_view_conf = loss_details[f"conf_{i+1}"][:num_imgs_vis].detach().cpu()
            cross_view_conf_list = batch_append(
                cross_view_conf_list, cross_view_conf.unbind(dim=0)
            )
            cross_view_conf_exits = True

        if f"self_conf_{i+1}" in loss_details:
            self_view_conf = (
                loss_details[f"self_conf_{i+1}"][:num_imgs_vis].detach().cpu()
            )
            self_view_conf_list = batch_append(
                self_view_conf_list, self_view_conf.unbind(dim=0)
            )
            self_view_conf_exits = True

        img_mask_list = batch_append(
            img_mask_list,
            loss_details[f"img_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )
        ray_mask_list = batch_append(
            ray_mask_list,
            loss_details[f"ray_mask_{i+1}"][:num_imgs_vis].detach().cpu().unbind(dim=0),
        )

    # each element in the list is [H, num_views * W, (3)], the size of the list is num_imgs_vis
    gt_img_list = [torch.cat(sublist, dim=1) for sublist in gt_img_list]
    pred_img_list = [torch.cat(sublist, dim=1) for sublist in pred_img_list]
    cross_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in cross_pred_depth_list
    ]
    cross_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in cross_gt_depth_list]
    self_gt_depth_list = [torch.cat(sublist, dim=1) for sublist in self_gt_depth_list]
    self_pred_depth_list = [
        torch.cat(sublist, dim=1) for sublist in self_pred_depth_list
    ]
    cross_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in cross_view_conf_list]
        if cross_view_conf_exits
        else []
    )
    self_view_conf_list = (
        [torch.cat(sublist, dim=1) for sublist in self_view_conf_list]
        if self_view_conf_exits
        else []
    )
    # each elment in the list is [num_views,], the size of the list is num_imgs_vis
    img_mask_list = [torch.stack(sublist, dim=0) for sublist in img_mask_list]
    ray_mask_list = [torch.stack(sublist, dim=0) for sublist in ray_mask_list]

    ray_indicator = gen_mask_indicator(
        img_mask_list, ray_mask_list, len(img_mask_list[0]), 30, width
    )

    for i in range(num_imgs_vis):
        out = vis_and_cat(
            gt_img_list[i],
            pred_img_list[i],
            cross_gt_depth_list[i],
            cross_pred_depth_list[i],
            self_gt_depth_list[i],
            self_pred_depth_list[i],
            cross_view_conf_list[i],
            self_view_conf_list[i],
            ray_indicator[i],
            is_metric[i],
        )
        ret_dict[f"imgs_{i}"] = out
    return ret_dict


@hydra.main(
    version_base=None,
    config_path=str(os.path.dirname(os.path.abspath(__file__))) + "/../config",
    config_name="train.yaml",
)
def run(cfg: OmegaConf):
    OmegaConf.resolve(cfg)
    logdir = pathlib.Path(cfg.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train(cfg)


def _inject_variant_override(argv):
    rewritten = [argv[0]]
    cli_variant = None
    i = 1
    while i < len(argv):
        token = argv[i]
        if token == "--variant":
            if i + 1 >= len(argv):
                raise SystemExit(
                    "Missing value for --variant. Example: --variant original_cut3r"
                )
            cli_variant = argv[i + 1]
            i += 2
            continue
        if token.startswith("--variant="):
            cli_variant = token.split("=", 1)[1]
            i += 1
            continue
        rewritten.append(token)
        i += 1

    if cli_variant is None:
        return rewritten

    has_existing_override = any(
        item.startswith("variant=")
        or item.startswith("+variant=")
        or item.startswith("++variant=")
        for item in rewritten[1:]
    )
    if has_existing_override:
        raise SystemExit(
            "Use either --variant or a Hydra variant override (variant=... / +variant=... / ++variant=...), not both."
        )

    rewritten.append(f"++variant={cli_variant}")
    return rewritten


if __name__ == "__main__":
    sys.argv = _inject_variant_override(sys.argv)
    run()
