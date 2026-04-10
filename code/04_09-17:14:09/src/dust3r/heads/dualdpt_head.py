import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple, Union
from dust3r.heads.postprocess import postprocess_pose, postprocess_rgb
from dust3r.utils.geometry import geotrf
from dust3r.utils.camera import pose_encoding_to_camera, PoseDecoder

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def custom_interpolate(
    x: torch.Tensor,
    size: Union[Tuple[int, int], None] = None,
    scale_factor: Union[float, None] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        assert scale_factor is not None, "Either size or scale_factor must be provided."
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]

    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [
            nn.functional.interpolate(c, size=size, mode=mode, align_corners=align_corners)
            for c in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()

    return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    return torch.stack((uu, vv), dim=-1)

def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  
    pos = pos.reshape(-1)  
    out = torch.einsum("m,d->md", pos, omega)  
    emb_sin = torch.sin(out)  
    emb_cos = torch.cos(out)  
    return torch.cat([emb_sin, emb_cos], dim=1).float()

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)
    emb = torch.cat([emb_x, emb_y], dim=-1)
    return emb.view(H, W, embed_dim)

class Permute(nn.Module):
    def __init__(self, dims: Tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)

# -----------------------------------------------------------------------------
# DPT Building blocks
# -----------------------------------------------------------------------------
class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module, bn: bool, groups: int = 1) -> None:
        super().__init__()
        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1, bias=True, groups=groups)
        self.norm1 = None
        self.norm2 = None
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None: out = self.norm1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None: out = self.norm2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock(nn.Module):
    def __init__(
        self, features: int, activation: nn.Module, deconv: bool = False, bn: bool = False,
        expand: bool = False, align_corners: bool = True, size: Tuple[int, int] = None,
        has_residual: bool = True, groups: int = 1,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.size = size
        self.has_residual = has_residual
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=groups) if has_residual else None
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=groups)
        out_features = (features // 2) if expand else features
        self.out_conv = nn.Conv2d(features, out_features, 1, 1, 0, bias=True, groups=groups)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs: torch.Tensor, size: Tuple[int, int] = None) -> torch.Tensor:
        y = xs[0]
        if self.has_residual and len(xs) > 1 and self.resConfUnit1 is not None:
            y = self.skip_add.add(y, self.resConfUnit1(xs[1]))
        y = self.resConfUnit2(y)
        up_kwargs = {"scale_factor": 2} if (size is None and self.size is None) else {"size": size if size is not None else self.size}
        y = custom_interpolate(y, **up_kwargs, mode="bilinear", align_corners=self.align_corners)
        return self.out_conv(y)

def _make_fusion_block(features: int, size: Tuple[int, int] = None, has_residual: bool = True, groups: int = 1, inplace: bool = False) -> nn.Module:
    return FeatureFusionBlock(features=features, activation=nn.ReLU(inplace=inplace), deconv=False, bn=False,
                              expand=False, align_corners=True, size=size, has_residual=has_residual, groups=groups)

def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    c1 = out_shape
    c2 = out_shape * (2 if expand else 1)
    c3 = out_shape * (4 if expand else 1)
    c4 = out_shape * (8 if expand else 1)
    scratch.layer1_rn = nn.Conv2d(in_shape[0], c1, 3, 1, 1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], c2, 3, 1, 1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], c3, 3, 1, 1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], c4, 3, 1, 1, bias=False, groups=groups)
    return scratch

# -----------------------------------------------------------------------------
# Dual Head Core implementation
# -----------------------------------------------------------------------------
class DualDPTHead(nn.Module):
    def __init__(self, net, has_conf=True, has_rgb=False, has_pose=False):
        super(DualDPTHead, self).__init__()
        self.return_all_layers = True
        
        # Pull required dims from CUT3R net
        dim_in = net.dec_embed_dim
        
        # DualDPT Configuration
        self.patch_size = net.patch_embed.patch_size[0]
        self.activation = "exp"
        self.conf_activation = "expp1"
        self.pose_mode = net.pose_mode
        self.pos_embed = True
        self.down_ratio = 1
        self.aux_levels = 4
        self.aux_out1_conv_num = 5
        self.head_main, self.head_aux = ("depth", "ray")
        self.has_conf = has_conf
        self.has_rgb = has_rgb
        self.has_pose = has_pose
        
        output_dim = 1
        features = 256
        out_channels = (net.enc_embed_dim, net.dec_embed_dim, net.dec_embed_dim, net.dec_embed_dim) 
        # Hooks corresponding to out_channels above
        self.hooks_idx = [0, net.dec_depth * 2 // 4, net.dec_depth * 3 // 4, net.dec_depth]
        self.intermediate_layer_idx = (0, 1, 2, 3)

        self.norms = nn.ModuleList([nn.LayerNorm(c) for c in out_channels])
        self.projects = nn.ModuleList([nn.Conv2d(c, oc, kernel_size=1, stride=1, padding=0) for c, oc in zip(out_channels, out_channels)])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Main fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim + 1, kernel_size=1, stride=1, padding=0), # +1 for conf
        )

        if self.has_rgb:
            self.scratch.output_conv2_rgb = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, 3, kernel_size=1, stride=1, padding=0),
            )

        if self.has_pose:
            self.pose_head = PoseDecoder(hidden_size=net.dec_embed_dim)

        # Aux fusion chain
        self.scratch.refinenet1_aux = _make_fusion_block(features)
        self.scratch.refinenet2_aux = _make_fusion_block(features)
        self.scratch.refinenet3_aux = _make_fusion_block(features)
        self.scratch.refinenet4_aux = _make_fusion_block(features, has_residual=False)

        self.scratch.output_conv1_aux = nn.ModuleList([
            self._make_aux_out1_block(head_features_1) for _ in range(self.aux_levels)
        ])
        
        ln_seq = [Permute((0, 2, 3, 1)), nn.LayerNorm(head_features_2), Permute((0, 3, 1, 2))]
        self.scratch.output_conv2_aux = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                *ln_seq,
                nn.ReLU(inplace=True),
                # 3 ray-direction channels + 1 confidence.
                # The original 7-channel layout had 4 unsupervised dead channels;
                # removing them makes the architecture match its semantics exactly.
                nn.Conv2d(head_features_2, 3 + 1, kernel_size=1, stride=1, padding=0),
            )
            for _ in range(self.aux_levels)
        ])

    def forward(self, x, img_info, **kwargs):
        H, W = img_info[0], img_info[1]

        pose = None
        if self.has_pose:
            pose_token = x[-1][:, 0].clone()
            with torch.cuda.amp.autocast(enabled=False):
                pose = self.pose_head(pose_token)
        
        # The hooks are already extracted by the ARCroco3DStereo model before passing to the head
        assert len(x) == 4, f"DualDPTHead expects exactly 4 hooks, got {len(x)}"
        feats = list(x)
        
        B, N, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            f = feats[take_idx]
            
            # Extract only the spatial tokens (ignoring potential cls/pose tokens prepended)
            f = f[:, -ph * pw:] 
            f = self.norms[stage_idx](f)
            
            C_f = f.shape[-1]
            f = f.permute(0, 2, 1).contiguous().reshape(B, C_f, ph, pw)
            f = self.projects[stage_idx](f)
            
            if self.pos_embed:
                f = self._add_pos_embed(f, W, H)
                
            f = self.resize_layers[stage_idx](f)
            resized_feats.append(f)

        fused_main, fused_aux_pyr = self._fuse(resized_feats)

        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused_main = custom_interpolate(fused_main, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)

        rgb_output = None
        if self.has_rgb:
            rgb_logits = self.scratch.output_conv2_rgb(fused_main)
            rgb_output = postprocess_rgb(rgb_logits)

        last_aux = fused_aux_pyr[-1]
        if self.pos_embed:
            last_aux = self._add_pos_embed(last_aux, W, H)
            
        last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
        fmap_last = last_aux_logits.permute(0, 2, 3, 1)
        aux_pred = self._apply_activation_single(fmap_last[..., :-1], "linear")
        aux_conf = self._apply_activation_single(fmap_last[..., -1], self.conf_activation)
        
        main_pred_sq = main_pred.squeeze(-1)
        
        # Extrapolate pts3d representation from ray spatial coords scaled by depth.
        # Since ray is patch-level, we must interpolate it to high-res match depth geometry first.
        aux_pred_up = custom_interpolate(aux_pred.permute(0, 3, 1, 2), size=main_pred.shape[1:3], mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
        aux_conf_up = custom_interpolate(aux_conf.unsqueeze(1), size=main_conf.shape[1:3], mode="bilinear", align_corners=True).squeeze(1)

        # Seal the "magnitude leak": L2-normalise so depth carries 100% of the
        # scale information and the ray branch carries 100% of the direction.
        # This makes the factorisation pts3d = depth * unit_ray geometrically strict
        # and ensures the RayConsistencyLoss gradient signal is scale-free.
        unit_rays = F.normalize(aux_pred_up[..., :3], dim=-1, eps=1e-6)  # [B,H,W,3]
        pts3d = unit_rays * main_pred
        conf = main_conf * aux_conf_up  # Joint confidence

        final_output = {
            self.head_main: main_pred_sq,
            f"{self.head_main}_conf": main_conf,
            self.head_aux: aux_pred,           # raw patch-level (7ch), kept for viz
            f"{self.head_aux}_conf": aux_conf,
            "pts3d": pts3d,
            "conf": conf,
            "unit_rays": unit_rays,            # normalised full-res rays for loss
        }

        # Standard CUT3R keys used by losses.
        final_output["pts3d_in_self_view"] = final_output.pop("pts3d")
        final_output["conf_self"] = final_output.pop("conf")

        if self.has_pose:
            pose = postprocess_pose(pose, self.pose_mode)
            final_output["camera_pose"] = pose
            final_output["pts3d_in_other_view"] = geotrf(
                pose_encoding_to_camera(final_output["camera_pose"]),
                final_output["pts3d_in_self_view"],
            )
            # Reuse dense confidence for cross-view when no dedicated cross-conf branch.
            final_output["conf"] = final_output["conf_self"]
        else:
            final_output["pts3d_in_other_view"] = final_output["pts3d_in_self_view"]
            final_output["conf"] = final_output["conf_self"]

        if rgb_output is not None:
            final_output.update(rgb_output)
        
        return final_output

    def _fuse(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        l1, l2, l3, l4 = feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        aux_list: List[torch.Tensor] = []
        if self.aux_levels >= 4: aux_list.append(aux_out)

        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3: aux_list.append(aux_out)

        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2: aux_list.append(aux_out)

        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        aux_list.append(aux_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](aux) for i, aux in enumerate(aux_list)]
        return out, aux_list

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _make_aux_out1_block(self, in_ch: int) -> nn.Sequential:
        if self.aux_out1_conv_num == 5:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        raise ValueError(f"aux_out1_conv_num {self.aux_out1_conv_num} not supported")

    def _apply_activation_single(self, x: torch.Tensor, activation: str = "linear") -> torch.Tensor:
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp": return torch.exp(x)
        if act == "expp1": return torch.exp(x) + 1
        if act == "expm1": return torch.expm1(x)
        if act == "relu": return torch.relu(x)
        if act == "sigmoid": return torch.sigmoid(x)
        if act == "softplus": return torch.nn.functional.softplus(x)
        if act == "tanh": return torch.tanh(x)
        return x


def create_dualdpt_head(net, has_conf=True, has_rgb=False, has_pose=False):
    return DualDPTHead(net, has_conf=has_conf, has_rgb=has_rgb, has_pose=has_pose)
