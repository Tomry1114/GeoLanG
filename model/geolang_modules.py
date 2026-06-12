import torch
import torch.nn as nn

from .layers import DWConv2d


class DGGM(nn.Module):
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
    def __init__(self, in_dim, num_layers, num_groups=1, out_dim=None, hidden_dim=None):
        super().__init__()
        assert num_layers % num_groups == 0
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.num_groups = num_groups
        self.group_size = num_layers // num_groups
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or max(in_dim // 4, 1)

        self.gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d((num_groups + 1) * in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats):
        assert len(feats) == self.num_layers
        scores = [self.gate(C_i.mean(dim=(2, 3))) for C_i in feats]

        group_feats = []
        for g in range(self.num_groups):
            idx = list(range(g * self.group_size, (g + 1) * self.group_size))
            s_g = torch.cat([scores[i] for i in idx], dim=1)
            alpha = torch.softmax(s_g, dim=1)
            gc = 0
            for j, i in enumerate(idx):
                gc = gc + alpha[:, j].view(-1, 1, 1, 1) * feats[i]
            group_feats.append(gc)

        cat = torch.cat(group_feats + [feats[-1]], dim=1)
        return self.fuse(cat)
