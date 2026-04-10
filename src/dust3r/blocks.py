# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import torch
import torch.nn as nn

from itertools import repeat
import collections.abc
from torch.nn.functional import scaled_dot_product_attention
from functools import partial


import dust3r.utils.path_to_croco  # ensures sys.path includes croco/src
from models.blocks import (          # re-export for downstream callers
    _ntuple,
    to_2tuple,
    drop_path,
    DropPath,
    Mlp,
    Attention,
    Block,
    CrossAttention,
    DecoderBlock,
    PositionGetter,
    PatchEmbed,
    LayerScale,
    SwiGLU,
    RecurrentBlock,
)


class CustomDecoderBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cross_attn = CrossAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        self.norm_z = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, z, xpos, ypos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        z_ = self.norm_z(z)
        x = x + self.drop_path(self.cross_attn(self.norm2(x), y_, z_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y


class ModLN(nn.Module):
    """
    Modulation with adaLN.

    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """

    def __init__(self, inner_dim: int, mod_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):

        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mlp(mod).chunk(2, dim=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]


class ConditionModulationBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(ModLN, eps=1e-6),
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, dim)
        self.attn = Attention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, mod, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x, mod), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x, mod)))
        return x


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    import dust3r.utils.path_to_croco
    from models.pos_embed import get_2d_sincos_pos_embed, RoPE2D
    from functools import partial
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(0)

    enc_blocks_ray_map = (
        nn.ModuleList(
            [
                Block(
                    768,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=RoPE2D(100),
                )
                for _ in range(2)
            ]
        )
        .cuda()
        .train()
    )

    x = torch.randn(2, 196, 768, requires_grad=True).cuda()
    xpos = torch.arange(0, 196).unsqueeze(0).unsqueeze(-1).repeat(2, 1, 2).cuda().long()
    enc_blocks_ray_map.zero_grad()
    for blk in enc_blocks_ray_map:

        x = checkpoint(blk, x, xpos)
    enc_blocks_ray_map.zero_grad()
    x.sum().backward()

    grad_not_checkpointed = {}
    for name, param in enc_blocks_ray_map.named_parameters():
        grad_not_checkpointed[name] = param.grad.data.clone()
        print(name, grad_not_checkpointed[name])
        break
