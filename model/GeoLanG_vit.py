import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import FPN, Projector, TransformerDecoder, MultiTaskProjector, CrossFusionNeck, CrossGSA, GeoPriorGen
from .layers import dense_connector
from types import SimpleNamespace

class CROGVIT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        # Flags for ablation study
        self.use_contrastive = cfg.use_contrastive
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.use_grasp_masks = cfg.use_grasp_masks
        
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        # state_dict = torch.load(cfg.clip_pretrain, map_location="cpu")
        
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, self.use_pretrained_clip).float()
        # self.backbone = build_model(state_dict, cfg.word_len, self.use_pretrained_clip).float()
        # Multi-Modal FPN
        # self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        
        self.neck = CrossFusionNeck()
        
        # Decoder
        if self.use_contrastive:
            print("Use contrastive learning module")
            # Decoder
            self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                            d_model=cfg.vis_dim,
                                            nhead=cfg.num_head,
                                            dim_ffn=cfg.dim_ffn,
                                            dropout=cfg.dropout,
                                            return_intermediate=cfg.intermediate)
        else:
            print("Disable contrastive learning module")
        if self.use_grasp_masks:
            # Projector
            print("Use grasp masks")
            self.proj = MultiTaskProjector(cfg.word_dim, cfg.vis_dim // 2, 3)
        else:
            print("Disable grasp masks")
            self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)
            
        self.cross_gsa_modules = nn.ModuleList([
            CrossGSA(embed_dim=ch, num_heads=8) for ch in [768, 768, 768]
        ])
        
        self.geo_prior_gen = GeoPriorGen(
            embed_dim=768,
            num_heads=8,
            initial_value=4,
            heads_range=1
        )
        self.dense_reducer = None  # lazy 初始化 reducer
    
    @staticmethod
    def fmap_to_tokens(x):
        B, C, H, W = x.shape
        tokens = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return tokens, (H, W)

    @staticmethod
    def tokens_to_fmap(tokens, hw):
        B, N, C = tokens.shape
        H, W = hw
        fmap = tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return fmap
    
    def _build_image_forward_outs_from_vis(self, vis_list, target_hw):
        hidden_states = []
        for feat in vis_list:
            resized = F.interpolate(feat, size=target_hw, mode='bilinear', align_corners=False)
            tokens, _ = self.fmap_to_tokens(resized)
            hidden_states.append(tokens)
        return SimpleNamespace(hidden_states=hidden_states)
    

    def forward(self, img, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None, grasp_cos_mask=None, grasp_wid_mask=None, dp_image=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        
        vis = self.backbone.encode_image(img)[1:] # [B, C, H', W'] # 取3个不同层特征

        word, state = self.backbone.encode_text(word)
        
        fused_vis = []
        if dp_image is not None:
            for i, v in enumerate(vis):
                B, C, H, W = v.shape
                # 生成几何先验
                HW_tuple = (H, W)
                geo_prior = self.geo_prior_gen(HW_tuple, dp_image)
                
                # 使用 CrossGSA 模块，传入 image feature 和几何先验
                x = v.permute(0, 2, 3, 1)  # [B, H, W, C]
                x_fused = self.cross_gsa_modules[i](x, rel_pos=geo_prior)
                x_fused = x_fused.permute(0, 3, 1, 2)  # back to [B, C, H, W]
                fused_vis.append(x_fused)
            vis = fused_vis
             
        # b, 512, 26, 26 
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        
        if self.use_contrastive:
            fq = self.decoder(fq, word, pad_mask)
            fq = fq.reshape(b, c, h, w)
        
        # ------------------ dense_connector 整合 ------------------
        fq_tokens, hw = self.fmap_to_tokens(fq)
        image_forward_outs = self._build_image_forward_outs_from_vis(vis, target_hw=(h, w))

        enhanced_tokens = dense_connector(fq_tokens, image_forward_outs, True, 'dci')  # 固定 dci
        enhanced_fq = self.tokens_to_fmap(enhanced_tokens, hw)

        in_c = enhanced_fq.shape[1]
        if in_c != c:  # dci 输出通常是 3c
            if self.dense_reducer is None:
                # lazy 初始化 Conv2d
                self.dense_reducer = nn.Conv2d(in_c, c, kernel_size=1).to(enhanced_fq.device)
            fq = self.dense_reducer(enhanced_fq)
        else:
            fq = enhanced_fq
        # ------------------ dense_connector 结束 ------------------


        if self.use_grasp_masks:
            
            # b, 1, 104, 104
            pred, grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
                    grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode='nearest').detach()
                    grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode='nearest').detach()
                    grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode='nearest').detach()
                    grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode='nearest').detach()

                # Ratio Augmentation
                total_area = mask.shape[2] * mask.shape[3]
                coef = 1 - (mask.sum(dim=(2,3)) / total_area)

                # Generate weight
                weight = mask * 0.5 + 1

                loss = F.binary_cross_entropy_with_logits(pred, mask, weight=weight)
                grasp_qua_loss = F.smooth_l1_loss(grasp_qua_pred, grasp_qua_mask)
                grasp_sin_loss = F.smooth_l1_loss(grasp_sin_pred, grasp_sin_mask)
                grasp_cos_loss = F.smooth_l1_loss(grasp_cos_pred, grasp_cos_mask)
                grasp_wid_loss = F.smooth_l1_loss(grasp_wid_pred, grasp_wid_mask)

                # @TODO adjust coef of different loss items
                total_loss = 2*loss + grasp_qua_loss + 1.5*grasp_sin_loss + 2*grasp_cos_loss + grasp_wid_loss

                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = grasp_qua_loss.item()
                loss_dict["m_sin"] = grasp_sin_loss.item()
                loss_dict["m_cos"] = grasp_cos_loss.item()
                loss_dict["m_wid"] = grasp_wid_loss.item()

                # loss = F.binary_cross_entropy_with_logits(pred, mask, reduction="none").sum(dim=(2,3))
                # loss = torch.dot(coef.squeeze(), loss.squeeze()) / (mask.shape[0] * mask.shape[2] * mask.shape[3])

                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask), total_loss, loss_dict
            else:
                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        else:
            # b, 1, 104, 104
            pred = self.proj(fq, state)

            if self.training:
                # resize mask
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:],
                                        mode='nearest').detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                loss_dict = {}
                loss_dict["m_ins"] = loss.item()
                loss_dict["m_qua"] = 0
                loss_dict["m_sin"] = 0
                loss_dict["m_cos"] = 0
                loss_dict["m_wid"] = 0
                return (pred.detach(), None, None, None, None), (mask, None, None, None, None), loss, loss_dict
            else:
                return pred.detach(), mask