import torch
import torch.nn as nn

from .vmamba.vmamba import VSSM


VMAMBA_PRESETS = {
    "tiny":  dict(dims=[96, 192, 384, 768],   depths=[2, 2, 4, 2]),
    "small": dict(dims=[96, 192, 384, 768],   depths=[2, 2, 15, 2]),
    "base":  dict(dims=[128, 256, 512, 1024], depths=[2, 2, 15, 2]),
}

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
        stage_out = dims[1:] + [dims[-1]]
        sel_dims = stage_out[:num_scales]
        self.sel_dims = sel_dims
        self.projs = nn.ModuleList([nn.Conv2d(d, out_dim, kernel_size=1) for d in sel_dims])

    def forward_features(self, x):
        x = self.vssm.patch_embed(x)
        outs = []
        for layer in self.vssm.layers:
            x = layer(x)
            outs.append(x)
        return outs

    def forward(self, img):
        outs = self.forward_features(img)[: self.num_scales]
        feats = []
        for proj, o in zip(self.projs, outs):
            if not self.channel_first:
                o = o.permute(0, 3, 1, 2).contiguous()
            feats.append(proj(o))
        return feats

    @torch.no_grad()
    def load_vssm_weights(self, state_dict, strict=False):
        cleaned = {k[len("visual."):] if k.startswith("visual.") else k: v
                   for k, v in state_dict.items()}
        missing, unexpected = self.vssm.load_state_dict(cleaned, strict=strict)
        return missing, unexpected


def build_vmamba_backbone(cfg):
    return VMambaBackbone(
        size=getattr(cfg, "vmamba_size", "tiny"),
        out_dim=getattr(cfg, "vmamba_out_dim", 768),
        forward_type=getattr(cfg, "vmamba_forward_type", "v2pt"),
        num_scales=3,
    )
