# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# CroCo model during pretraining
# --------------------------------------------------------


import torch
import torch.nn as nn
import glob
import os
import sys
import importlib

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
from functools import partial

from models.blocks import Block, DecoderBlock, PatchEmbed, RecurrentBlock
from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
from models.masking import RandomMask

from transformers import PretrainedConfig
from transformers import PreTrainedModel


def strip_module(state_dict):
    """Remove the 'module.' prefix from state_dict keys (DDP artifact)."""
    return {
        (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _resolve_checkpoint_path(path: str) -> str:
    if os.path.isfile(path):
        return path

    candidates = []
    if any(ch in path for ch in "*?[]"):
        # explicit glob pattern — use it directly
        candidates.extend(glob.glob(path))
    else:
        # exact or ambiguous path — try direct glob then fuzzy prefix match
        candidates.extend(glob.glob(path))
        dirname = os.path.dirname(path) or "."
        basename = os.path.basename(path)
        candidates.extend(glob.glob(os.path.join(dirname, f"*{basename}")))

    candidates = [p for p in candidates if os.path.isfile(p)]
    if not candidates:
        raise FileNotFoundError(f"Cannot resolve recurrent checkpoint path: {path}")
    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        preview = "\n  - " + "\n  - ".join(candidates[:10])
        extra = "" if len(candidates) <= 10 else f"\n  ... (+{len(candidates) - 10} more)"
        raise ValueError(
            "Ambiguous recurrent checkpoint path. Please provide an exact file path "
            f"or a pattern matching one file only: {path}\nCandidates:{preview}{extra}"
        )
    return candidates[0]


def _extract_recurrent_state_dict(loaded_obj, block_index=0):
    if isinstance(loaded_obj, nn.Module):
        if hasattr(loaded_obj, "blocks") and isinstance(
            loaded_obj.blocks, (nn.ModuleList, list, tuple)
        ):
            if len(loaded_obj.blocks) == 0:
                raise ValueError("Loaded recurrent module has empty `blocks`.")
            idx = max(0, min(int(block_index), len(loaded_obj.blocks) - 1))
            return loaded_obj.blocks[idx].state_dict(), f"module.blocks[{idx}]"
        return loaded_obj.state_dict(), loaded_obj.__class__.__name__

    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
            return strip_module(loaded_obj["state_dict"]), "dict.state_dict"

        if "model" in loaded_obj:
            model_obj = loaded_obj["model"]
            if isinstance(model_obj, nn.Module):
                return _extract_recurrent_state_dict(model_obj, block_index=block_index)
            if isinstance(model_obj, dict):
                return strip_module(model_obj), "dict.model"

        looks_like_state_dict = all(
            isinstance(k, str)
            for k in loaded_obj.keys()
        ) and any(
            key.startswith("norm1") or key.startswith("attn") or key.startswith("mlp")
            for key in loaded_obj.keys()
        )
        if looks_like_state_dict:
            return _strip_module_prefix(loaded_obj), "raw_state_dict"

    raise TypeError(
        f"Unsupported recurrent checkpoint object type: {type(loaded_obj).__name__}"
    )


def _extract_recurrent_state_dict_list(loaded_obj):
    """Best-effort extraction of per-block recurrent state dicts.

    Returns:
        (list_of_state_dicts, source) if recurrent module exposes `blocks`
        (None, None) otherwise.
    """
    if isinstance(loaded_obj, nn.Module):
        if hasattr(loaded_obj, "blocks") and isinstance(
            loaded_obj.blocks, (nn.ModuleList, list, tuple)
        ):
            if len(loaded_obj.blocks) == 0:
                raise ValueError("Loaded recurrent module has empty `blocks`.")
            return [blk.state_dict() for blk in loaded_obj.blocks], "module.blocks[*]"
        return None, None

    if isinstance(loaded_obj, dict) and "model" in loaded_obj:
        model_obj = loaded_obj["model"]
        if isinstance(model_obj, nn.Module):
            return _extract_recurrent_state_dict_list(model_obj)

    return None, None


def _candidate_source_dirs_from_checkpoint(resolved_path: str):
    """Return candidate source roots to resolve pickled checkpoint modules."""
    candidates = []
    cur = os.path.abspath(os.path.dirname(resolved_path))
    for _ in range(7):
        if cur in ("", os.path.sep):
            break
        candidates.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    source_dirs = []
    for d in candidates:
        # direct source root (e.g., .../raptor1/src)
        if os.path.isfile(os.path.join(d, "raptor.py")) or os.path.isfile(
            os.path.join(d, "block.py")
        ):
            source_dirs.append(d)
        # repo root containing src/
        src_d = os.path.join(d, "src")
        if os.path.isdir(src_d) and (
            os.path.isfile(os.path.join(src_d, "raptor.py"))
            or os.path.isfile(os.path.join(src_d, "block.py"))
        ):
            source_dirs.append(src_d)

    # de-duplicate preserving order
    seen = set()
    unique = []
    for d in source_dirs:
        if d not in seen:
            unique.append(d)
            seen.add(d)
    return unique


def load_recurrent_blocks_from_checkpoint(
    blocks: nn.ModuleList,
    recurrent_ckpt: str,
    block_index: int = 0,
    strict: bool = False,
):
    if recurrent_ckpt is None:
        return None

    resolved_path = _resolve_checkpoint_path(recurrent_ckpt)
    try:
        loaded_obj = torch.load(resolved_path, map_location="cpu", weights_only=False)
    except ModuleNotFoundError as e:
        # RAPTOR checkpoints are often pickled with classes from `raptor.py`
        # and/or `block.py`. Recover by locating nearby source dirs and
        # importing the missing module before retrying torch.load.
        missing_mod = getattr(e, "name", "")
        if missing_mod not in {"block", "raptor"}:
            raise

        imported = False
        import_errors = []
        for src_dir in _candidate_source_dirs_from_checkpoint(resolved_path):
            try:
                if src_dir not in sys.path:
                    sys.path.insert(0, src_dir)
                mod = importlib.import_module(missing_mod)
                sys.modules[missing_mod] = mod
                imported = True
                break
            except Exception as ie:
                import_errors.append(f"{src_dir}: {ie}")

        if not imported:
            detail = "\n  - " + "\n  - ".join(import_errors[:6]) if import_errors else ""
            raise ModuleNotFoundError(
                f"Could not import missing module '{missing_mod}' while loading recurrent checkpoint {resolved_path}.{detail}"
            ) from e

        loaded_obj = torch.load(resolved_path, map_location="cpu", weights_only=False)

    multi_state_dicts, multi_source = _extract_recurrent_state_dict_list(loaded_obj)

    load_results = []
    source_indices = []
    if multi_state_dicts is not None and len(blocks) > 1:
        n_src = len(multi_state_dicts)
        n_tgt = len(blocks)
        for i, blk in enumerate(blocks):
            src_idx = i if i < n_src else (i % n_src)
            source_indices.append(src_idx)
            load_results.append(
                blk.load_state_dict(multi_state_dicts[src_idx], strict=strict)
            )
        source = f"{multi_source} mapped {n_src}-> {n_tgt}"
    else:
        state_dict, source = _extract_recurrent_state_dict(
            loaded_obj, block_index=block_index
        )
        for blk in blocks:
            load_results.append(blk.load_state_dict(state_dict, strict=strict))
            source_indices.append(int(block_index))

    return {
        "resolved_path": resolved_path,
        "source": source,
        "num_blocks": len(blocks),
        "strict": strict,
        "source_indices": source_indices,
        "load_results": load_results,
    }


class CrocoConfig(PretrainedConfig):
    model_type = "croco"

    def __init__(
        self,
        img_size=224,  # input image size
        patch_size=16,  # patch_size
        mask_ratio=0.9,  # ratios of masked tokens
        enc_embed_dim=768,  # encoder feature dimension
        enc_depth=12,  # encoder depth
        enc_num_heads=12,  # encoder number of heads in the transformer block
        dec_embed_dim=512,  # decoder feature dimension
        dec_depth=8,  # decoder depth
        dec_num_heads=16,  # decoder number of heads in the transformer block
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_im2_in_dec=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder
        pos_embed="cosine",  # positional embedding (either cosine or RoPE100)
        num_blocks=1,
        num_steps=None,
        use_tbptl=True,
        tbptl_warmup=2,
        recurrent_ckpt=None,
        recurrent_block_index=0,
        recurrent_strict=False,
        recurrent_t_scale=False,
        recurrent_swiglu=False,
        ls_init=1e-5,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.enc_embed_dim = enc_embed_dim
        self.enc_depth = enc_depth
        self.enc_num_heads = enc_num_heads
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth
        self.dec_num_heads = dec_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.norm_im2_in_dec = norm_im2_in_dec
        self.pos_embed = pos_embed
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.use_tbptl = use_tbptl
        self.tbptl_warmup = tbptl_warmup
        self.recurrent_ckpt = recurrent_ckpt
        self.recurrent_block_index = recurrent_block_index
        self.recurrent_strict = recurrent_strict
        self.recurrent_t_scale = recurrent_t_scale
        self.recurrent_swiglu = recurrent_swiglu
        self.ls_init = ls_init


class CroCoNet(PreTrainedModel):

    config_class = CrocoConfig
    base_model_prefix = "croco"

    def __init__(self, config: CrocoConfig):

        super().__init__(config)

        # patch embeddings  (with initialization done as in MAE)
        self._set_patch_embed(config.img_size, config.patch_size, config.enc_embed_dim)

        # mask generations
        self._set_mask_generator(self.patch_embed.num_patches, config.mask_ratio)

        self.pos_embed = config.pos_embed
        if config.pos_embed == "cosine":
            # positional embedding of the encoder
            enc_pos_embed = get_2d_sincos_pos_embed(
                config.enc_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer(
                "enc_pos_embed", torch.from_numpy(enc_pos_embed).float()
            )
            # positional embedding of the decoder
            dec_pos_embed = get_2d_sincos_pos_embed(
                config.dec_embed_dim,
                int(self.patch_embed.num_patches**0.5),
                n_cls_token=0,
            )
            self.register_buffer(
                "dec_pos_embed", torch.from_numpy(dec_pos_embed).float()
            )
            # pos embedding in each block
            self.rope = None  # nothing for cosine
        elif config.pos_embed.startswith("RoPE"):  # eg RoPE100
            self.enc_pos_embed = None  # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None  # nothing to add in the decoder with RoPE
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(config.pos_embed[len("RoPE") :])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError("Unknown pos_embed " + config.pos_embed)

        # transformer for the encoder
        self.enc_depth = config.enc_depth
        self.enc_embed_dim = config.enc_embed_dim
        
        # Recurrent Parameters
        self.num_blocks = int(config.num_blocks)
        self.num_steps = (
            int(config.num_steps)
            if config.num_steps is not None
            else self.enc_depth // self.num_blocks
        )
        if self.enc_depth != self.num_steps * self.num_blocks:
            print(
                "[RecurrentEncoder] Warning: enc_depth "
                f"({self.enc_depth}) != num_steps*num_blocks "
                f"({self.num_steps}*{self.num_blocks}={self.num_steps * self.num_blocks})."
            )
        self.use_tbptl = bool(config.use_tbptl)
        self.tbptl_warmup = int(config.tbptl_warmup)
        recurrent_ckpt = config.recurrent_ckpt
        recurrent_block_index = int(config.recurrent_block_index)
        recurrent_strict = bool(config.recurrent_strict)
        
        self.enc_blocks = nn.ModuleList(
            [
                RecurrentBlock(
                    dim=config.enc_embed_dim,
                    num_heads=config.enc_num_heads,
                    mlp_ratio=config.mlp_ratio,
                    init_values=config.ls_init,
                    t_scale=config.recurrent_t_scale,
                    swiglu=config.recurrent_swiglu,
                )
                for i in range(self.num_blocks)
            ]
        )
        
        if recurrent_ckpt is not None:
            print(f"Loading recurrent weights from {recurrent_ckpt}")
            info = load_recurrent_blocks_from_checkpoint(
                self.enc_blocks,
                recurrent_ckpt,
                block_index=recurrent_block_index,
                strict=recurrent_strict,
            )
            print(
                "Loaded recurrent backbone from "
                f"{info['resolved_path']} ({info['source']}) into {info['num_blocks']} block(s)"
            )
            print(
                "Recurrent source block mapping -> target blocks: "
                f"{info['source_indices']}"
            )

        self.enc_norm = config.norm_layer(config.enc_embed_dim)

        # masked tokens
        self._set_mask_token(config.dec_embed_dim)

        # decoder
        self._set_decoder(
            config.enc_embed_dim,
            config.dec_embed_dim,
            config.dec_num_heads,
            config.dec_depth,
            config.mlp_ratio,
            config.norm_layer,
            config.norm_im2_in_dec,
        )

        # prediction head
        self._set_prediction_head(config.dec_embed_dim, config.patch_size)

        # initializer weights
        self.initialize_weights()

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, enc_embed_dim)

    def _set_mask_generator(self, num_patches, mask_ratio):
        self.mask_generator = RandomMask(num_patches, mask_ratio)

    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

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
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
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
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_prediction_head(self, dec_embed_dim, patch_size):
        self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)

    def initialize_weights(self):
        # patch embed
        self.patch_embed._init_weights()
        # mask tokens
        if self.mask_token is not None:
            torch.nn.init.normal_(self.mask_token, std=0.02)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _encode_image(self, image, do_mask=False, return_all_blocks=False):
        """
        image has B x 3 x img_size x img_size
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C)
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)
        # add positional embedding without cls token
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]
        # apply masking
        B, N, C = x.size()
        if do_mask:
            masks = self.mask_generator(x)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B, N, C = x.size()
            masks = torch.zeros((B, N), dtype=bool)
            posvis = pos
        # now apply the transformer encoder and normalization
        if return_all_blocks:
            out = []
            if self.use_tbptl and self.training:
                with torch.no_grad():
                    for step in range(self.tbptl_warmup):
                        for blk in self.enc_blocks:
                            x = blk(x, posvis)
                        out.append(x)
                if self.tbptl_warmup > 0:
                    x = x.detach()
                    x.requires_grad = True
                    if self.tbptl_warmup == self.num_steps:
                        out[-1] = x
                for step in range(self.tbptl_warmup, self.num_steps):
                    for blk in self.enc_blocks:
                        x = blk(x, posvis)
                    out.append(x)
            else:
                for step in range(self.num_steps):
                    for blk in self.enc_blocks:
                        x = blk(x, posvis)
                    out.append(x)
            out[-1] = self.enc_norm(out[-1])
            return out, pos, masks
        else:
            if self.use_tbptl and self.training:
                with torch.no_grad():
                    for step in range(self.tbptl_warmup):
                        for blk in self.enc_blocks:
                            x = blk(x, posvis)
                if self.tbptl_warmup > 0:
                    x = x.detach()
                    x.requires_grad = True
                for step in range(self.tbptl_warmup, self.num_steps):
                    for blk in self.enc_blocks:
                        x = blk(x, posvis)
            else:
                for step in range(self.num_steps):
                    for blk in self.enc_blocks:
                        x = blk(x, posvis)
            x = self.enc_norm(x)
            return x, pos, masks

    def _decoder(self, feat1, pos1, masks1, feat2, pos2, return_all_blocks=False):
        """
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)

        masks1 can be None => assume image1 fully visible
        """
        # encoder to decoder layer
        visf1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)
        # append masked tokens to the sequence
        B, Nenc, C = visf1.size()
        if masks1 is None:  # downstreams
            f1_ = visf1
        else:  # pretraining
            Ntotal = masks1.size(1)
            f1_ = self.mask_token.repeat(B, Ntotal, 1).to(dtype=visf1.dtype)
            f1_[~masks1] = visf1.view(B * Nenc, C)
        # add positional embedding
        if self.dec_pos_embed is not None:
            f1_ = f1_ + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed
        # apply Transformer blocks
        out = f1_
        out2 = f2
        if return_all_blocks:
            _out, out = out, []
            for blk in self.dec_blocks:
                _out, out2 = blk(_out, out2, pos1, pos2)
                out.append(_out)
            out[-1] = self.dec_norm(out[-1])
        else:
            for blk in self.dec_blocks:
                out, out2 = blk(out, out2, pos1, pos2)
            out = self.dec_norm(out)
        return out

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))
        return imgs

    # def forward(self, img1, img2):
    # """
    # img1: tensor of size B x 3 x img_size x img_size
    # img2: tensor of size B x 3 x img_size x img_size

    # out will be    B x N x (3*patch_size*patch_size)
    # masks are also returned as B x N just in case
    # """
    # # encoder of the masked first image
    # feat1, pos1, mask1 = self._encode_image(img1, do_mask=True)
    # # encoder of the second image
    # feat2, pos2, _ = self._encode_image(img2, do_mask=False)
    # # decoder
    # decfeat = self._decoder(feat1, pos1, mask1, feat2, pos2)
    # # prediction head
    # out = self.prediction_head(decfeat)
    # # get target
    # target = self.patchify(img1)
    # return out, mask1, target
