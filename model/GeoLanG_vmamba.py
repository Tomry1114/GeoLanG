"""GeoLanG (VMamba) model — the paper's full method (ICRA'25, §II).

Separate from the ViT baseline ``CROGVIT`` (model/GeoLanG_vit.py): this class wires the
paper's pipeline (Fig. 2): CLIP-VMamba image encoder -> per-scale DGGM (Eq.4) -> ADCI
(Algorithm 1) -> Multimodal Fusion Neck (C_v ⊗ C_s) -> multi-task decoder -> seg/grasp heads.

Vision backbone: CLIP-Mamba VSSM (cfg.vision_backbone='vmamba'); 'vit' falls back to the
CLIP-ViT visual encoder for ablation. Text branch: CLIP text transformer by default, or the
paper's CLIP-BERT when cfg.text_encoder='bert'. The paper-specific blocks DGGM (§II-B, Eq.4)
and ADCI (Algorithm 1) live in model/geolang_modules.py; GeoPriorGen / CrossFusionBlock are
reused from layers.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model

from .layers import Projector, TransformerDecoder, MultiTaskProjector, CrossFusionBlock, GeoPriorGen
from .geolang_modules import DGGM, ADCI
from .vmamba_backbone import build_vmamba_backbone


class GeoLanG(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Flags for ablation study
        self.use_contrastive = cfg.use_contrastive
        self.use_pretrained_clip = cfg.use_pretrained_clip
        self.use_grasp_masks = cfg.use_grasp_masks

        # Vision & Text Encoder (CLIP). The CLIP backbone also provides the default text branch.
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        print(f"Load pretrained CLIP: {self.use_pretrained_clip}")
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, self.use_pretrained_clip).float()

        # Multimodal Fusion Neck: a single cross-attention block fusing C_v with the
        # sentence embedding C_s (paper Fig. 2). Multi-scale aggregation is handled by ADCI.
        self.neck = CrossFusionBlock(img_dim=cfg.vis_dim, txt_dim=cfg.word_dim, hidden_dim=cfg.vis_dim)

        # Decoder
        if self.use_contrastive:
            print("Use contrastive learning module")
            self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                              d_model=cfg.vis_dim,
                                              nhead=cfg.num_head,
                                              dim_ffn=cfg.dim_ffn,
                                              dropout=cfg.dropout,
                                              return_intermediate=cfg.intermediate)
        else:
            print("Disable contrastive learning module")
        if self.use_grasp_masks:
            print("Use grasp masks")
            self.proj = MultiTaskProjector(cfg.word_dim, cfg.vis_dim // 2, 3)
        else:
            print("Disable grasp masks")
            self.proj = Projector(cfg.word_dim, cfg.vis_dim // 2, 3)

        # DGGM: depth-guided geometric self-attention per scale (paper §II-B, Eq.4).
        self.dggm_modules = nn.ModuleList([
            DGGM(embed_dim=ch, num_heads=8) for ch in [768, 768, 768]
        ])
        self.geo_prior_gen = GeoPriorGen(
            embed_dim=768,
            num_heads=8,
            initial_value=4,
            heads_range=1
        )

        # ADCI: adaptively integrate the 3 DGGM-enhanced multi-scale features into C_v.
        self.adci = ADCI(in_dim=768, num_layers=3, num_groups=1, out_dim=cfg.vis_dim)

        # Text encoder selection. 'bert' uses the paper's CLIP-BERT text encoder (§II-A);
        # 'clip' keeps the default CLIP text transformer. Imported lazily so the default
        # path still works without the `transformers` package installed.
        self.text_encoder = getattr(cfg, 'text_encoder', 'clip')
        if self.text_encoder == 'bert':
            from .bert_text_encoder import BertTextEncoder
            bert_pretrain = getattr(cfg, 'bert_pretrain', 'google-bert/bert-base-uncased')
            print(f"Use CLIP-BERT text encoder: {bert_pretrain}")
            self.bert = BertTextEncoder(pretrain=bert_pretrain, out_dim=cfg.word_dim)

        # Vision backbone selection. 'vmamba' uses the ported CLIP-Mamba VSSM encoder
        # (paper §II-A); 'vit' falls back to the CLIP-ViT visual encoder for ablation.
        self.vision_backbone = getattr(cfg, 'vision_backbone', 'vmamba')
        if self.vision_backbone == 'vmamba':
            print("Use VMamba (CLIP-Mamba VSSM) vision backbone")
            self.vmamba = build_vmamba_backbone(cfg)
            # Initialise the visual encoder with pretrained CLIP-VMamba weights (paper §III-A).
            # Provide a CLIP-Mamba VSSM checkpoint path via TRAIN.vmamba_pretrain; without it the
            # VSSM stays randomly initialised — fine for a forward pass, but not for matching the
            # paper's trained accuracy.
            vmamba_pretrain = getattr(cfg, 'vmamba_pretrain', '')
            if vmamba_pretrain:
                print(f"Load VMamba pretrained weights: {vmamba_pretrain}")
                sd = torch.load(vmamba_pretrain, map_location='cpu')
                sd = sd.get('state_dict', sd.get('model', sd)) if isinstance(sd, dict) else sd
                missing, unexpected = self.vmamba.load_vssm_weights(sd, strict=False)
                print(f"  VMamba load: {len(missing)} missing, {len(unexpected)} unexpected keys")
            else:
                print("  VMamba pretrained weights NOT provided -> randomly initialised "
                      "(set TRAIN.vmamba_pretrain to a CLIP-Mamba VSSM checkpoint)")

    def forward(self, img, word, mask=None, grasp_qua_mask=None, grasp_sin_mask=None,
                grasp_cos_mask=None, grasp_wid_mask=None, dp_image=None):
        '''
            img: b, 3, h, w
            word: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        if self.vision_backbone == 'vmamba':
            vis = self.vmamba(img)                      # 3 multi-scale feats [B, 768, H/8/16/32]
        else:
            vis = self.backbone.encode_image(img)[1:]   # CLIP-ViT 3 multi-scale feats

        if self.text_encoder == 'bert':
            word, state = self.bert(word)               # Ct: [B,L,word_dim], Cs: [B,word_dim]
        else:
            word, state = self.backbone.encode_text(word)

        # ---- DGGM: inject the depth-guided geometry prior per scale (Eq.4, §II-B) ----
        if dp_image is not None:
            fused_vis = []
            for i, v in enumerate(vis):
                B, C, H, W = v.shape
                geo_prior = self.geo_prior_gen((H, W), dp_image)
                x = v.permute(0, 2, 3, 1)                          # [B, H, W, C]
                x = self.dggm_modules[i](x, rel_pos=geo_prior)
                fused_vis.append(x.permute(0, 3, 1, 2))            # back to [B, C, H, W]
            vis = fused_vis

        # ---- ADCI: adaptively integrate multi-scale features into C_v (§II-C) ----
        target_hw = vis[0].shape[-2:]
        vis = [v if v.shape[-2:] == target_hw
               else F.interpolate(v, size=target_hw, mode='bilinear', align_corners=False)
               for v in vis]
        c_v = self.adci(vis)                                       # [B, vis_dim, h, w]

        # ---- Multimodal Fusion Neck: fuse C_v with sentence embedding C_s -> C_m ----
        fq = self.neck(c_v, state)
        b, c, h, w = fq.size()

        # ---- Multi-task decoder ----
        if self.use_contrastive:
            fq = self.decoder(fq, word, pad_mask)
            fq = fq.reshape(b, c, h, w)

        if self.use_grasp_masks:
            pred, grasp_qua_pred, grasp_sin_pred, grasp_cos_pred, grasp_wid_pred = self.proj(fq, state)

            if self.training:
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
                    grasp_qua_mask = F.interpolate(grasp_qua_mask, grasp_qua_pred.shape[-2:], mode='nearest').detach()
                    grasp_sin_mask = F.interpolate(grasp_sin_mask, grasp_sin_pred.shape[-2:], mode='nearest').detach()
                    grasp_cos_mask = F.interpolate(grasp_cos_mask, grasp_cos_pred.shape[-2:], mode='nearest').detach()
                    grasp_wid_mask = F.interpolate(grasp_wid_mask, grasp_wid_pred.shape[-2:], mode='nearest').detach()

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

                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask), total_loss, loss_dict
            else:
                return (pred.detach(), grasp_qua_pred.detach(), grasp_sin_pred.detach(), grasp_cos_pred.detach(), grasp_wid_pred.detach()), (mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        else:
            pred = self.proj(fq, state)

            if self.training:
                if pred.shape[-2:] != mask.shape[-2:]:
                    mask = F.interpolate(mask, pred.shape[-2:], mode='nearest').detach()
                loss = F.binary_cross_entropy_with_logits(pred, mask)
                loss_dict = {"m_ins": loss.item(), "m_qua": 0, "m_sin": 0, "m_cos": 0, "m_wid": 0}
                return (pred.detach(), None, None, None, None), (mask, None, None, None, None), loss, loss_dict
            else:
                return pred.detach(), mask
