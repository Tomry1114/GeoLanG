"""VMamba (VSSM) vision backbone for GeoLanG.

Ported from CLIP-Mamba (https://github.com/raytrun/mamba-clip), whose VSSM definition
lives in ``model/vmamba/vmamba.py``. This wrapper exposes the backbone in the shape the
GeoLanG pipeline expects: the last three multi-scale feature maps (downsample 8/16/32,
i.e. excluding the lowest H/4 stage, exactly as described in the paper §II-A), each
projected to a common channel width so the downstream DGGM / ADCI modules stay unchanged.

forward(img) -> [feat_s8, feat_s16, feat_s32], each [B, out_dim, H/s, W/s].

Notes on pretrained weights (CLIP-Mamba HF: weiquan/mamba-clip):
  * The architecture presets below match the ``vssm1`` configs used by CLIP-Mamba
    (CLIP_VMamba_T220 -> "tiny", _S -> "small", _B -> "base"), so a VSSM state_dict
    can be loaded via ``load_vssm_weights``.
  * ``forward_type`` only selects the selective-scan *backend*, not any parameters, so
    "v2pt" (portable pure-pytorch, CPU-friendly) and "v3"/"v2" (compiled CUDA kernels)
    share the same weights. Use "v2pt" when the selective_scan CUDA kernel is unavailable.
"""

import torch
import torch.nn as nn

from .vmamba.vmamba import VSSM


# Architecture presets matching CLIP-Mamba's vssm1 configs (dims[-1] == vision_width).
VMAMBA_PRESETS = {
    "tiny":  dict(dims=[96, 192, 384, 768],   depths=[2, 2, 4, 2]),    # CLIP_VMamba_T220
    "small": dict(dims=[96, 192, 384, 768],   depths=[2, 2, 15, 2]),   # CLIP_VMamba_S
    "base":  dict(dims=[128, 256, 512, 1024], depths=[2, 2, 15, 2]),   # CLIP_VMamba_B
}

# Shared SSM/MLP settings for the vssm1 presets above.
_COMMON = dict(
    ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank="auto", ssm_conv=3, ssm_conv_bias=False,
    mlp_ratio=4.0, downsample_version="v3", patchembed_version="v2", norm_layer="LN",
)


class VMambaBackbone(nn.Module):
    def __init__(self, size="tiny", out_dim=768, forward_type="v2pt", num_scales=3):
        super().__init__()
        assert size in VMAMBA_PRESETS, f"unknown VMamba size {size}"
        preset = VMAMBA_PRESETS[size]
        self.vssm = VSSM(num_classes=0, forward_type=forward_type, **preset, **_COMMON)
        self.channel_first = self.vssm.channel_first
        self.num_scales = num_scales

        dims = preset["dims"]
        # Output channel of layer i (blocks + downsample) is dims[i+1] for i < L-1, else dims[-1].
        stage_out = dims[1:] + [dims[-1]]
        sel_dims = stage_out[:num_scales]          # take the first `num_scales` stages = s8/s16/s32
        self.sel_dims = sel_dims
        # 1x1 projection to unify channels so DGGM(CrossGSA)/ADCI see a single width.
        self.projs = nn.ModuleList([nn.Conv2d(d, out_dim, kernel_size=1) for d in sel_dims])

    def forward_features(self, x):
        """Return the per-stage feature maps as a list (channel-last unless channel_first)."""
        x = self.vssm.patch_embed(x)
        outs = []
        for layer in self.vssm.layers:
            x = layer(x)
            outs.append(x)
        return outs  # [o0@H/8, o1@H/16, o2@H/32, o3@H/32]

    def forward(self, img):
        outs = self.forward_features(img)[: self.num_scales]
        feats = []
        for proj, o in zip(self.projs, outs):
            if not self.channel_first:                 # (B, H, W, C) -> (B, C, H, W)
                o = o.permute(0, 3, 1, 2).contiguous()
            feats.append(proj(o))
        return feats

    @torch.no_grad()
    def load_vssm_weights(self, state_dict, strict=False):
        """Load a CLIP-Mamba VSSM vision state_dict into ``self.vssm`` (projections stay
        randomly initialised). Pass the inner VSSM weights (strip any ``visual.`` prefix)."""
        cleaned = {k[len("visual."):] if k.startswith("visual.") else k: v
                   for k, v in state_dict.items()}
        missing, unexpected = self.vssm.load_state_dict(cleaned, strict=strict)
        return missing, unexpected


def build_vmamba_backbone(cfg):
    """Build a VMambaBackbone from a GeoLanG cfg namespace."""
    return VMambaBackbone(
        size=getattr(cfg, "vmamba_size", "tiny"),
        out_dim=getattr(cfg, "vmamba_out_dim", 768),
        forward_type=getattr(cfg, "vmamba_forward_type", "v2pt"),
        num_scales=3,
    )
