"""GeoLanG-specific building blocks (paper §II-B / §II-C).

Kept separate from ``model/layers.py`` so the original CROGVIT (ViT baseline) and its
modules stay untouched. These are used only by ``model/GeoLanG_vmamba.py`` (class GeoLanG):

  * ``DGGM`` — Depth-guided Geometric Module (paper §II-B): geometry-aware self-attention with
    the geometry prior injected *multiplicatively after softmax* per Eq. 4:
    X_hat = (Softmax(QK^T) ⊙ η·G) V^T.
    (The ViT baseline's ``CrossGSA`` in layers.py injects it additively before softmax; this
    is the corrected, paper-faithful variant.)
  * ``ADCI`` — Adaptive Dense Channel Integration (Algorithm 1).

Both reuse ``GeoPriorGen`` / ``DWConv2d`` from layers.py unchanged.
"""

import torch
import torch.nn as nn

from .layers import DWConv2d


class DGGM(nn.Module):
    """Depth-guided Geometric Module (paper §II-B): geometry-aware self-attention that
    injects the depth+spatial geometry prior G multiplicatively after softmax (Eq. 4)."""

    def __init__(self, embed_dim, num_heads, value_factor=1):
        super().__init__()
        self.factor = value_factor
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim * self.factor // num_heads
        self.key_dim = self.embed_dim // num_heads
        self.scaling = self.key_dim**-0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor)

        self.lepe = DWConv2d(embed_dim, 5, 1, 2)
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim)

        self.layer_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, rel_pos=None):
        """
        x: image feature [b h w c]
        y: depth feature [b h w c] (optional; by default Q/K/V all come from x, see Fig. 3)
        rel_pos: ((sin, cos), geo_prior) from GeoPriorGen. Only ``geo_prior`` (the
                 depth+spatial geometry prior in log-decay form, i.e. log(eta * G)) is used.
        """
        bsz, h, w, _ = x.size()
        q = self.q_proj(x)
        if y is not None:
            k = self.k_proj(y)
            v = self.v_proj(y)
        else:
            k = self.k_proj(x)
            v = self.v_proj(x)
        lepe = self.lepe(v)

        k = k * self.scaling

        qr = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        kr = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = vr.flatten(2, 3)

        qk_mat = qr @ kr.transpose(-1, -2)

        # Geometry-Aware attention (paper Eq. 4):  X_hat = (Softmax(QK^T) (.) eta*G) V^T
        # The geometry prior is injected *multiplicatively after softmax* (element-wise),
        # not as an additive logit bias. ``geo_prior`` already carries the per-head decay
        # rate eta in log space, so exp(geo_prior) in (0, 1] suppresses geometrically
        # distant key-value pairs while keeping nearby ones near 1.
        attn = torch.softmax(qk_mat, -1)
        if rel_pos is not None:
            _, geo_prior = rel_pos
            attn = attn * torch.exp(geo_prior)
        output = torch.matmul(attn, vr)

        output = output.transpose(1, 2).reshape(bsz, h, w, -1)
        output = output + lepe
        output = x + self.out_proj(output)

        output = output + self.layer_norm(output)

        return output

    def reset_parameters(self):
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)
        nn.init.xavier_normal_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)


class ADCI(nn.Module):
    """Adaptive Dense Channel Integration (paper Sec. II-C, Algorithm 1).

    Input : a list of L multi-layer visual feature maps ``[B, C, H, W]`` (same C/H/W).
    Output: the fused visual embedding ``e_v`` of shape ``[B, out_dim, H, W]``.

    The L layers are split into ``G`` groups of ``M = L / G`` consecutive layers.
    Within each group a lightweight gating network predicts adaptive weights
    ``alpha_i`` (softmax-normalised inside the group, Eq. 6/9); the group feature is
    ``GC_g = sum_i alpha_i * C_i`` (Eq. 5). The G group features are concatenated with
    the last layer ``C_L`` (Eq. 10) and projected by an MLP to ``e_v``.
    """

    def __init__(self, in_dim, num_layers, num_groups=1, out_dim=None, hidden_dim=None):
        super().__init__()
        assert num_layers % num_groups == 0, \
            f"num_layers({num_layers}) must be divisible by num_groups({num_groups})"
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.group_size = num_layers // num_groups
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or max(in_dim // 4, 1)

        # Gating network: GAP descriptor -> 2-layer MLP -> per-feature score (Eq. 7/8).
        # One score per feature map; softmax is applied within each group (Eq. 9).
        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        # Final embedding: concat(G group feats + last layer) -> MLP (Eq. 10).
        self.fuse = nn.Sequential(
            nn.Conv2d((num_groups + 1) * in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        assert len(feats) == self.num_layers, \
            f"ADCI expects {self.num_layers} feature maps, got {len(feats)}"
        # Step 2: feature descriptors via global average pooling (Eq. 7).
        scores = [self.gate(C_i.mean(dim=(2, 3))) for C_i in feats]  # each [B, 1]

        group_feats = []
        for g in range(self.num_groups):
            idx = list(range(g * self.group_size, (g + 1) * self.group_size))
            s_g = torch.cat([scores[i] for i in idx], dim=1)         # [B, M]
            alpha = torch.softmax(s_g, dim=1)                        # within-group (Eq. 9)
            gc = 0
            for j, i in enumerate(idx):
                gc = gc + alpha[:, j].view(-1, 1, 1, 1) * feats[i]   # weighted sum (Eq. 5)
            group_feats.append(gc)

        cat = torch.cat(group_feats + [feats[-1]], dim=1)           # concat last layer (Eq. 10)
        return self.fuse(cat)
