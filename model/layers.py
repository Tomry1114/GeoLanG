import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .difattetion import DiffAttention


def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class MultiTaskProjector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim*5, 1))

        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        x = torch.tensor_split(x, 5, dim=1) # no tensor_split api in torch 1.7, please use it in higher version
        # x = torch.chunk(x, 5, dim=1)

        mask_x = x[0]
        grasp_qua_x = x[1]
        grasp_sin_x = x[2]
        grasp_cos_x = x[3]
        grasp_wid_x = x[4]

        B, C, H, W = mask_x.size()


        # 1, b*256, 104, 104
        mask_x = mask_x.reshape(1, B * C, H, W)
        grasp_qua_x = grasp_qua_x.reshape(1, B * C, H, W)
        grasp_sin_x = grasp_sin_x.reshape(1, B * C, H, W)
        grasp_cos_x = grasp_cos_x.reshape(1, B * C, H, W)
        grasp_wid_x = grasp_wid_x.reshape(1, B * C, H, W)


        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        mask_out = F.conv2d(mask_x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        
        grasp_qua_out = F.conv2d(grasp_qua_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
        
        grasp_sin_out = F.conv2d(grasp_sin_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)

        grasp_cos_out = F.conv2d(grasp_cos_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
        
        grasp_wid_out = F.conv2d(grasp_wid_x,
                            weight,
                            padding=self.kernel_size // 2,
                            groups=weight.size(0),
                            bias=bias)
            
        mask_out = mask_out.transpose(0, 1)
        grasp_qua_out = grasp_qua_out.transpose(0, 1)
        grasp_sin_out = grasp_sin_out.transpose(0, 1)
        grasp_cos_out = grasp_cos_out.transpose(0, 1)
        grasp_wid_out = grasp_wid_out.transpose(0, 1)
        # b, 1, 104, 104

        return mask_out, grasp_qua_out, grasp_sin_out, grasp_cos_out, grasp_wid_out


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(512, out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(
            -1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq




import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_channels, out_channels, kernel_size=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class CrossFusionBlock(nn.Module):
    def __init__(self, img_dim=768, txt_dim=512, hidden_dim=512, num_heads=8):
        super(CrossFusionBlock, self).__init__()
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** -0.5
        
        self.img_proj = conv_layer(img_dim, hidden_dim)
        self.txt_proj = nn.Linear(txt_dim, hidden_dim)

        # Cross attention
        self.q_proj = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.final_proj = conv_layer(hidden_dim, hidden_dim)
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, img_feat, txt_feat):
        B, C, H, W = img_feat.shape
        D = hidden_dim = self.img_proj[0].out_channels

        # 投影图像和文本到统一空间
        img = self.img_proj(img_feat)  # [B, D, H, W]
        txt = self.txt_proj(txt_feat)  # [B, D]

        # Query: from image
        q = self.q_proj(img).flatten(2).transpose(1, 2)  # [B, H*W, D]
        q = q.reshape(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # [B, h, HW, d]

        # Key / Value: from text
        k = self.k_proj(txt).reshape(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)  # [B, h, T, d]
        v = self.v_proj(txt).reshape(B, -1, self.num_heads, D // self.num_heads).permute(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, HW, T]
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, -1)  # [B, HW, D] -> [B, H, W, D]
        x = x.permute(0, 3, 1, 2)

       # Residual + Norm
        fused = self.norm(x + img)
        fused = self.final_proj(fused)
        return fused

# ---------- 工具函数 ----------
def fmap_to_tokens(x):
    B, C, H, W = x.shape
    tokens = x.flatten(2).transpose(1,2)  # [B, H*W, C]
    return tokens, (H, W)

def tokens_to_fmap(tokens, hw):
    B, N, C = tokens.shape
    H, W = hw
    return tokens.transpose(1,2).reshape(B, C, H, W)

class CrossFusionNeck(nn.Module):
    def __init__(self, in_channels=768, txt_dim=512, out_channels=512, num_levels=3, num_heads=8):
        super().__init__()
        # 假设 CrossFusionBlock 已经定义
        self.fuse_blocks = nn.ModuleList([
            CrossFusionBlock(in_channels, txt_dim) for _ in range(num_levels)
        ])
        self.final_proj = nn.Conv2d(num_levels * out_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

        # DiffAttention
        self.diff_attn = DiffAttention(dim=out_channels, num_heads=num_heads)

    def forward(self, feats, state):
        fused_feats = [block(feat, state) for block, feat in zip(self.fuse_blocks, feats)]

        # 上采样到统一尺寸
        fused_sizes = [f.shape[-2:] for f in fused_feats]
        if len(set(fused_sizes)) > 1:
            target_size = max(fused_sizes)
            fused_feats = [F.interpolate(f, size=target_size, mode='bilinear', align_corners=False) for f in fused_feats]

        # Cat + Conv 聚合
        output = torch.cat(fused_feats, dim=1)
        output = self.final_proj(output)  # [B, out_channels, H, W]

        # DiffAttention
        tokens, hw = fmap_to_tokens(output)
        tokens = self.diff_attn(tokens)
        output = tokens_to_fmap(tokens, hw)

        return output


class DWConv2d(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        """
        input (b h w c)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        return x
    
def angle_transform(x, sin, cos):
    x1 = x[:, :, :, :, ::2]
    x2 = x[:, :, :, :, 1::2]
    #print(x1.shape, x2.shape, sin.shape, cos.shape)
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)


class GeoPriorGen(nn.Module):
    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)

    def generate_depth_decay(self, H: int, W: int, depth_grid):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]
        mask_d = (mask_d.abs()).sum(dim=-1)
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]
        return mask_d

    def generate_pos_decay(self, H: int, W: int):
        """
        generate 2d decay mask, the result is (HW)*(HW)
        H, W are the numbers of patches at each column and row
        """
        index_h = torch.arange(H).to(self.decay)
        index_w = torch.arange(W).to(self.decay)
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)
        mask = grid[:, None, :] - grid[None, :, :]
        mask = (mask.abs()).sum(dim=-1)
        mask = mask * self.decay[:, None, None]
        return mask

    def generate_1d_depth_decay(self, H, W, depth_grid):
        """
        generate 1d depth decay mask, the result is l*l
        """
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None, None]
        assert mask.shape[2:] == (W, H, H)
        return mask

    def generate_1d_decay(self, l: int):
        """
        generate 1d decay mask, the result is l*l
        """
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]
        mask = mask.abs()
        mask = mask * self.decay[:, None, None]
        return mask

    def forward(self, HW_tuple: Tuple[int], depth_map, split_or_not=False):
        """
        depth_map: depth patches
        HW_tuple: (H, W)
        H * W == l
        """
        depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

        if split_or_not:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

            mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)

            mask_h = self.generate_1d_decay(HW_tuple[0])
            mask_w = self.generate_1d_decay(HW_tuple[1])

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w

            geo_prior = ((sin, cos), (mask_h, mask_w))

        else:
            index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :])
            sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
            cos = torch.cos(index[:, None] * self.angle[None, :])
            cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
            mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])

            mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior
    
    
class CrossGSA(nn.Module):
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

    def forward(self, x: torch.Tensor, y: torch.Tensor=None, rel_pos=None):
        """
        x: image feature [b h w c]
        y: depth feature [b h w c]
        rel_pos: (sin, cos), mask from GeoPriorGen
        """
        bsz, h, w, _ = x.size()
        q = self.q_proj(x) 
        if y:          
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

        # Apply angle transform if provided
        if rel_pos is not None:
            sin, cos = rel_pos[0]
            qr = angle_transform(qr, sin, cos)
            kr = angle_transform(kr, sin, cos)

        qr = qr.flatten(2, 3)
        kr = kr.flatten(2, 3)
        vr = vr.flatten(2, 3)

        qk_mat = qr @ kr.transpose(-1, -2)
        

        if rel_pos is not None:
            _, mask = rel_pos
            qk_mat = qk_mat + mask
        qk_mat = torch.softmax(qk_mat, -1)
        output = torch.matmul(qk_mat, vr)

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
        
        
        
    

def dense_connector_sti(image_features, image_forward_outs, is_siglip=True):
    avg_pooling_k8 = nn.AvgPool1d(kernel_size=8, stride=8)
    if not is_siglip:
        image_features_1 = image_forward_outs.hidden_states[7][:, 1:].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, 1:].to(image_features.dtype)
    else:
        image_features_1 = image_forward_outs.hidden_states[7][:, :].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, :].to(image_features.dtype)
    image_features_1 = avg_pooling_k8(image_features_1.permute(0, 2, 1)).permute(0, 2, 1)
    image_features_2 = avg_pooling_k8(image_features_2.permute(0, 2, 1)).permute(0, 2, 1)
    return torch.cat([image_features_1, image_features_2], dim=-2)

def dense_connector_sci(image_features,image_forward_outs, is_siglip=True):
    if not is_siglip:
        image_features_1 = image_forward_outs.hidden_states[7][:, 1:].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, 1:].to(image_features.dtype)
    else:
        image_features_1 = image_forward_outs.hidden_states[7][:, :].to(image_features.dtype)
        image_features_2 = image_forward_outs.hidden_states[15][:, :].to(image_features.dtype)
    return torch.cat([image_features_1, image_features_2], dim=-1)

def dense_connector_dci(image_features, image_forward_outs, is_siglip=True):
    """
    Dense Connector for DCI.
    自适应不同 ViT 模型的 hidden_states 数量 (12 / 24 / 26 层均可)
    """

    num_layers = len(image_forward_outs.hidden_states)
    half = num_layers // 2   # 自动分前半和后半

    image_features_1 = []
    image_features_2 = []

    # 前半层
    for i in range(0, half):
        feats = image_forward_outs.hidden_states[i]
        if not is_siglip:  # clip 结构去掉 cls token
            feats = feats[:, 1:]
        image_features_1.append(feats.to(image_features.dtype))
    image_features_1 = torch.stack(image_features_1, dim=0).mean(dim=0)

    # 后半层
    for i in range(half, num_layers):
        feats = image_forward_outs.hidden_states[i]
        if not is_siglip:
            feats = feats[:, 1:]
        image_features_2.append(feats.to(image_features.dtype))
    image_features_2 = torch.stack(image_features_2, dim=0).mean(dim=0)

    # 拼接前后两段特征
    return torch.cat([image_features_1, image_features_2], dim=-1)


def dense_connector(image_features, image_forward_outs, is_siglip=True, connector_type="dci"):
    """
    Dense Connector 入口函数
    根据 connector_type 调用不同实现
    """

    #num_layers = len(image_forward_outs.hidden_states)
    #print(f"[DenseConnector] backbone hidden_states 层数 = {num_layers}, connector_type = {connector_type}")

    return dense_connector_dci(image_features, image_forward_outs, is_siglip)

