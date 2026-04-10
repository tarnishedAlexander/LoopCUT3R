import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from models.croco import load_recurrent_blocks_from_checkpoint, strip_module  # noqa: strip_module re-exported for train.py
from models.blocks import RecurrentBlock
from dust3r.utils.safe_eval import safe_eval_expr
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
    CustomDecoderBlock,
)  # noqa

inf = float("inf")
import logging

printer = logging.getLogger(__name__)
printer.setLevel(logging.DEBUG)


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None



def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    try:
        # PyTorch >=2.6 defaults to weights_only=True, which breaks loading
        # CUT3R checkpoints that store full training args/config objects.
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility with older torch versions.
        ckpt = torch.load(model_path, map_location="cpu")
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = safe_eval_expr(
        args,
        {
            "ARCroco3DStereo": ARCroco3DStereo,
            "ARCroco3DStereoConfig": ARCroco3DStereoConfig,
            "inf": inf,
        },
        context="model",
    )
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        num_blocks=1,
        num_steps=None,
        use_tbptl=True,
        tbptl_warmup=2,
        tbptl_thinking=0,
        tbptl_grad_steps=None,
        recurrent_ckpt=None,
        recurrent_block_index=0,
        recurrent_strict=False,
        recurrent_t_scale=False,
        recurrent_swiglu=False,
        ls_init=1e-5,
        dec_num_blocks_img=None,
        dec_num_steps_img=None,
        dec_num_blocks_state=None,
        dec_num_steps_state=None,
        dec_use_tbptl=False,
        dec_tbptl_warmup=0,
        dec_tbptl_thinking=0,
        dec_tbptl_grad_steps=None,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.use_tbptl = use_tbptl
        self.tbptl_warmup = tbptl_warmup
        self.tbptl_thinking = tbptl_thinking
        self.tbptl_grad_steps = tbptl_grad_steps
        self.recurrent_ckpt = recurrent_ckpt
        self.recurrent_block_index = recurrent_block_index
        self.recurrent_strict = recurrent_strict
        self.recurrent_t_scale = recurrent_t_scale
        self.recurrent_swiglu = recurrent_swiglu
        self.ls_init = ls_init
        self.dec_num_blocks_img = dec_num_blocks_img
        self.dec_num_steps_img = dec_num_steps_img
        self.dec_num_blocks_state = dec_num_blocks_state
        self.dec_num_steps_state = dec_num_steps_state
        self.dec_use_tbptl = dec_use_tbptl
        self.dec_tbptl_warmup = dec_tbptl_warmup
        self.dec_tbptl_thinking = dec_tbptl_thinking
        self.dec_tbptl_grad_steps = dec_tbptl_grad_steps
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        )
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        """Retrieve a pose feature from memory given an image-level query.

        Layout assumption: the stored token is [query_proj | masked_token] along
        the last dim (size 2*v_dim). After cross-attention with ``mem``, the
        second half (``x[..., -v_dim:]``) carries the memory-retrieved feature.
        If the concatenation order ever changes, this slice must be updated too.
        """
        x = self.proj_q(query)  # [B, 1, C=v_dim]
        # Concat a learnable blank slot; the read_blocks fill it via cross-attn.
        # Layout: [query_proj (v_dim) | masked_token (v_dim)] → 2*v_dim
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        # Slice the second half: the filled slot that cross-attn wrote into.
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        self.croco_args.update(
            {
                "num_blocks": config.num_blocks,
                "num_steps": config.num_steps,
                "use_tbptl": config.use_tbptl,
                "tbptl_warmup": config.tbptl_warmup,
                "recurrent_ckpt": config.recurrent_ckpt,
                "recurrent_block_index": config.recurrent_block_index,
                "recurrent_strict": config.recurrent_strict,
                "recurrent_t_scale": config.recurrent_t_scale,
                "recurrent_swiglu": config.recurrent_swiglu,
                "ls_init": config.ls_init,
            }
        )
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                RecurrentBlock(
                    dim=self.enc_embed_dim,
                    num_heads=self.croco_args["enc_num_heads"],
                    mlp_ratio=self.croco_args["mlp_ratio"],
                    init_values=config.ls_init,
                    t_scale=config.recurrent_t_scale,
                    swiglu=config.recurrent_swiglu,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        # Load recurrent backbone weights to ray-map blocks as well
        recurrent_ckpt = config.recurrent_ckpt
        if recurrent_ckpt is not None:
            info = load_recurrent_blocks_from_checkpoint(
                self.enc_blocks_ray_map,
                recurrent_ckpt,
                block_index=config.recurrent_block_index,
                strict=config.recurrent_strict,
            )
            printer.info(
                "Loaded recurrent ray-map backbone from %s (%s)",
                info["resolved_path"],
                info["source"],
            )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            )
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim)
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        legacy_dec_depth = int(self.dec_depth)
        self.dec_num_blocks_img = int(
            getattr(config, "dec_num_blocks_img", None) or legacy_dec_depth
        )
        self.dec_num_steps_img = int(
            getattr(config, "dec_num_steps_img", None) or legacy_dec_depth
        )
        self.dec_num_blocks_state = int(
            getattr(config, "dec_num_blocks_state", None) or self.dec_num_blocks_img
        )
        self.dec_num_steps_state = int(
            getattr(config, "dec_num_steps_state", None) or self.dec_num_steps_img
        )
        self.dec_use_tbptl = bool(getattr(config, "dec_use_tbptl", False))
        self.dec_tbptl_warmup = int(getattr(config, "dec_tbptl_warmup", 0) or 0)
        self.dec_tbptl_thinking = int(getattr(config, "dec_tbptl_thinking", 0) or 0)
        self.dec_tbptl_grad_steps = getattr(config, "dec_tbptl_grad_steps", None)
        if self.dec_num_blocks_img < len(self.dec_blocks):
            self.dec_blocks = nn.ModuleList(
                [self.dec_blocks[i] for i in range(self.dec_num_blocks_img)]
            )
        else:
            self.dec_num_blocks_img = len(self.dec_blocks)
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_num_blocks_state,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        # effective unroll depth used by downstream taps/losses
        self.dec_depth = self.dec_num_steps_img
        self.dec_depth_state = self.dec_num_steps_state
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)
        self.tbptl_thinking = int(getattr(config, "tbptl_thinking", 0) or 0)
        self.tbptl_grad_steps = getattr(config, "tbptl_grad_steps", None)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def load_recurrent_backbone(
        self,
        recurrent_ckpt: str,
        block_index: int = 0,
        strict: bool = False,
    ):
        image_info = load_recurrent_blocks_from_checkpoint(
            self.enc_blocks,
            recurrent_ckpt,
            block_index=block_index,
            strict=strict,
        )
        ray_info = load_recurrent_blocks_from_checkpoint(
            self.enc_blocks_ray_map,
            recurrent_ckpt,
            block_index=block_index,
            strict=strict,
        )
        return {"image": image_info, "ray": ray_info}

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except Exception as e1:
            printer.warning("load_state_dict: first attempt failed (%s), retrying without dec_blocks/dec_norm/decoder_embed keys", e1)
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except Exception as e2:
                printer.warning("load_state_dict: second attempt failed (%s), falling back to size-matched keys only", e2)
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        _pose_modules = (
            [self.pose_retriever, self.pose_token] if self.pose_head_flag else []
        )
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                *_pose_modules,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                *_pose_modules,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _run_recurrent_encoder_blocks(self, x, pos, blocks):
        # TBPTL already controls activation memory. Combining it with per-block
        # checkpointing in recurrent loops can cause checkpoint metadata mismatch
        # during backward for repeated block applications.
        use_block_checkpoint = (
            self.gradient_checkpointing and self.training and not self.use_tbptl
        )

        def run_one_step(x_in):
            for blk in blocks:
                if use_block_checkpoint:
                    x_in = checkpoint(blk, x_in, pos, use_reentrant=False)
                else:
                    x_in = blk(x_in, pos)
            return x_in

        total_steps = int(self.num_steps)
        if not (self.use_tbptl and self.training):
            for _ in range(total_steps):
                x = run_one_step(x)
            return x

        warmup_steps = min(max(0, int(self.tbptl_warmup)), total_steps)
        thinking_steps = min(
            max(0, int(getattr(self, "tbptl_thinking", 0) or 0)),
            max(0, total_steps - warmup_steps),
        )
        grad_steps_cfg = getattr(self, "tbptl_grad_steps", None)
        if grad_steps_cfg is None:
            grad_start = warmup_steps + thinking_steps
        else:
            grad_steps = max(0, int(grad_steps_cfg))
            grad_start = max(total_steps - grad_steps, warmup_steps + thinking_steps)

        with torch.no_grad():
            for _ in range(0, grad_start):
                x = run_one_step(x)

        if grad_start > 0:
            x = x.detach()
            x.requires_grad = True

        for _ in range(grad_start, total_steps):
            x = run_one_step(x)

        return x

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None

        x = self._run_recurrent_encoder_blocks(x, pos, self.enc_blocks)

        x = self.enc_norm(x)
        return [x], pos, None

    @torch.no_grad()
    def _encode_image_all_layers(self, image, true_shape):
        """Return image-encoder activations for all layers.

        Output list format:
            [layer0_patch_embed, layer1, ..., layerN]
        where layerN is normalized by `enc_norm`.
        """
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        layer_acts = [x]
        for step in range(self.num_steps):
            for blk in self.enc_blocks:
                x = blk(x, pos)
            layer_acts.append(x)
        x = self.enc_norm(x)
        layer_acts[-1] = x
        return layer_acts, pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None

        x = self._run_recurrent_encoder_blocks(x, pos, self.enc_blocks_ray_map)
                        
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    @torch.no_grad()
    def _encode_ray_map_all_layers(self, ray_map, true_shape):
        """Return ray-map encoder activations for all layers.

        Output list format:
            [layer0_patch_embed, layer1, ..., layerN]
        where layerN is normalized by `enc_norm_ray_map`.
        """
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        layer_acts = [x]
        for step in range(self.num_steps):
            for blk in self.enc_blocks_ray_map:
                x = blk(x, pos)
            layer_acts.append(x)
        x = self.enc_norm_ray_map(x)
        layer_acts[-1] = x
        return layer_acts, pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        )
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        return self._assemble_view_tokens(
            views, all_layers=False, img_mask=img_mask, ray_mask=ray_mask
        )

    @torch.no_grad()
    def _encode_views_all_layers(self, views, img_mask=None, ray_mask=None):
        return self._assemble_view_tokens(
            views, all_layers=True, img_mask=img_mask, ray_mask=ray_mask
        )

    def _assemble_view_tokens(self, views, all_layers=False, img_mask=None, ray_mask=None):
        """Shared body for _encode_views (all_layers=False) and _encode_views_all_layers.

        Returns:
            shapes_chunked  – tuple[num_views] of [B, 2] tensors
            feat_chunked    – list[num_steps_or_layers] of tuple[num_views] tensors
            pos_chunked     – tuple[num_views] of pos tensors
        """
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]

        given = img_mask is not None or ray_mask is not None
        if not given:
            img_mask = torch.stack([view["img_mask"] for view in views], dim=0)
            ray_mask = torch.stack([view["ray_mask"] for view in views], dim=0)

        imgs = torch.stack([view["img"] for view in views], dim=0)
        ray_maps = torch.stack([view["ray_map"] for view in views], dim=0)

        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(imgs.device)

        # Flatten (num_views, batch) → (num_views * batch)
        imgs = imgs.view(-1, *imgs.shape[2:])
        ray_maps = ray_maps.view(-1, *ray_maps.shape[2:])
        shapes = shapes.view(-1, 2)
        img_masks_flat = img_mask.view(-1)
        ray_masks_flat = ray_mask.view(-1)

        # ── Image encoder ──────────────────────────────────────────────────────
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) == 0:
            raise NotImplementedError("All img_mask entries are False — cannot encode.")

        _encode_img = self._encode_image_all_layers if all_layers else self._encode_image
        img_out, img_pos, _ = _encode_img(selected_imgs, selected_shapes)

        # img_out is list[num_steps_or_layers] of [N_img, seq, C]
        num_feat = len(img_out)
        fill_shape = (len(views) * batch_size, *img_out[0].shape[1:])
        full_out = [
            torch.zeros(fill_shape, device=img_out[0].device) for _ in range(num_feat)
        ]
        full_pos = torch.zeros(
            len(views) * batch_size, *img_pos.shape[1:],
            device=img_pos.device, dtype=img_pos.dtype,
        )
        for i in range(num_feat):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos

        # ── Ray-map encoder ────────────────────────────────────────────────────
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # (N, H, W, 6) → (N, 6, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]

        _encode_ray = (
            self._encode_ray_map_all_layers if all_layers else self._encode_ray_map
        )
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = _encode_ray(selected_ray_maps, selected_shapes_ray)
            assert len(ray_out) == len(full_out), (
                f"Ray/image layer count mismatch: {len(ray_out)} vs {len(full_out)}"
            )
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            # No ray maps present — run a zeroed dummy pass to keep grad graph intact
            dummy_ray = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1],
                device=img_out[0].device,
            )
            dummy_mask = torch.zeros_like(img_masks_flat)
            dummy_mask[:1] = True
            ray_out_dummy, _, _ = _encode_ray(dummy_ray, shapes[dummy_mask])
            for i in range(len(ray_out_dummy)):
                full_out[i][dummy_mask] += ray_out_dummy[i] * 0.0
                full_out[i][~dummy_mask] += self.masked_ray_map_token * 0.0

        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
        # Keep legacy first image output (before projection) for downstream taps.
        image_layers = [f_img]
        state_layers = [f_state]

        assert f_state.shape[-1] == self.dec_embed_dim
        f_img_proj = self.decoder_embed(f_img)
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img_proj = torch.cat([f_pose, f_img_proj], dim=1)
            pos_img = torch.cat([pos_pose, pos_img], dim=1)

        f_state_cur = f_state
        f_img_cur = f_img_proj

        steps_img = int(getattr(self, "dec_num_steps_img", self.dec_depth))
        steps_state = int(getattr(self, "dec_num_steps_state", self.dec_depth_state))
        blocks_img = max(1, len(self.dec_blocks))
        blocks_state = max(1, len(self.dec_blocks_state))
        total_steps = max(steps_img, steps_state)

        def run_step(step_idx, f_state_in, f_img_in):
            prev_state, prev_img = f_state_in, f_img_in
            next_state, next_img = prev_state, prev_img

            if step_idx < steps_state:
                blk_state = self.dec_blocks_state[step_idx % blocks_state]
                if (
                    self.gradient_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                    and not self.dec_use_tbptl
                ):
                    next_state, _ = checkpoint(
                        blk_state,
                        prev_state,
                        prev_img,
                        pos_state,
                        pos_img,
                        use_reentrant=not self.fixed_input_length,
                    )
                else:
                    next_state, _ = blk_state(prev_state, prev_img, pos_state, pos_img)

            if step_idx < steps_img:
                blk_img = self.dec_blocks[step_idx % blocks_img]
                if (
                    self.gradient_checkpointing
                    and self.training
                    and torch.is_grad_enabled()
                    and not self.dec_use_tbptl
                ):
                    next_img, _ = checkpoint(
                        blk_img,
                        prev_img,
                        prev_state,
                        pos_img,
                        pos_state,
                        use_reentrant=not self.fixed_input_length,
                    )
                else:
                    next_img, _ = blk_img(prev_img, prev_state, pos_img, pos_state)

            return next_state, next_img

        if not (self.dec_use_tbptl and self.training):
            for step_idx in range(total_steps):
                f_state_cur, f_img_cur = run_step(step_idx, f_state_cur, f_img_cur)
                if step_idx < steps_state:
                    state_layers.append(f_state_cur)
                if step_idx < steps_img:
                    image_layers.append(f_img_cur)
        else:
            warmup_steps = min(max(0, int(self.dec_tbptl_warmup)), total_steps)
            thinking_steps = min(
                max(0, int(getattr(self, "dec_tbptl_thinking", 0) or 0)),
                max(0, total_steps - warmup_steps),
            )
            grad_steps_cfg = getattr(self, "dec_tbptl_grad_steps", None)
            if grad_steps_cfg is None:
                grad_start = warmup_steps + thinking_steps
            else:
                grad_steps = max(0, int(grad_steps_cfg))
                grad_start = max(total_steps - grad_steps, warmup_steps + thinking_steps)

            with torch.no_grad():
                for step_idx in range(0, grad_start):
                    f_state_cur, f_img_cur = run_step(step_idx, f_state_cur, f_img_cur)
                    if step_idx < steps_state:
                        state_layers.append(f_state_cur)
                    if step_idx < steps_img:
                        image_layers.append(f_img_cur)

            if grad_start > 0:
                f_state_cur = f_state_cur.detach()
                f_img_cur = f_img_cur.detach()
                f_state_cur.requires_grad = True
                f_img_cur.requires_grad = True

            for step_idx in range(grad_start, total_steps):
                f_state_cur, f_img_cur = run_step(step_idx, f_state_cur, f_img_cur)
                if step_idx < steps_state:
                    state_layers.append(f_state_cur)
                if step_idx < steps_img:
                    image_layers.append(f_img_cur)

        # Normalize final elements to preserve legacy behavior.
        state_layers[-1] = self.dec_norm_state(state_layers[-1])
        image_layers[-1] = self.dec_norm(image_layers[-1])
        return state_layers, image_layers

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat)
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_decoder_layers=False,
    ):
        state_layers, image_layers = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos
        )
        new_state_feat = state_layers[-1]
        if return_decoder_layers:
            return new_state_feat, image_layers, state_layers
        return new_state_feat, image_layers

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def forward_encoder_recurrent(self, views, ret_state=True):
        """Public encoder-only recurrent entrypoint.

        Returns encoder tokens/positions/shapes and optionally initialized
        recurrent states for downstream recurrent decoding.
        """
        encoder_out, state_args = self._forward_encoder(views)
        if ret_state:
            return encoder_out, state_args
        return encoder_out

    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
        return_decoder_layers=False,
    ):
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        if return_decoder_layers:
            new_state_feat, dec, dec_state = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_decoder_layers=True,
            )
        else:
            new_state_feat, dec = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
            )
        if self.pose_head_flag:
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
        else:
            new_mem = mem
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )  # update global state
        if mem is not None and new_mem is not None:
            mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            if mem is not None and init_mem is not None:
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
        if return_decoder_layers:
            return res, (state_feat, mem), {
                "image_layers": dec,
                "state_layers": dec_state,
            }
        return res, (state_feat, mem)

    def _forward_impl(self, views, ret_state=False):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        if self.pose_head_flag:
            mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        else:
            mem = None
        init_state_feat = state_feat.clone()
        init_mem = mem.clone() if mem is not None else None
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            res, (state_feat, mem) = self._forward_decoder_step(
                views,
                i,
                feat_i,
                pos_i,
                shape[i],
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            )
            ress.append(res)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem
    ):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shape)

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
        )
        if self.pose_head_flag:
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
        else:
            new_mem = mem
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    def forward_recurrent(self, views, device, ret_state=False):
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                if self.pose_head_flag:
                    mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                else:
                    mem = None
                init_state_feat = state_feat.clone()
                init_mem = mem.clone() if mem is not None else None
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )

            res, (state_feat, mem) = self._forward_decoder_step(
                views,
                i,
                feat_i,
                pos_i,
                shape,
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            )
            ress.append(res)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views


if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
