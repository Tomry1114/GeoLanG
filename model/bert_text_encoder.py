"""CLIP-BERT text encoder for GeoLanG (paper §II-A, ref. [22]).

The paper encodes the language query with **CLIP-BERT** to obtain token embeddings
``Ct ∈ R^{K×C}`` and a sentence embedding ``Cs ∈ R^{C'}``. This module provides that
branch as a drop-in replacement for the default CLIP text transformer
(``backbone.encode_text``), returning the SAME two tensors so the downstream neck /
decoder / projectors stay unchanged.

Pretrained weights (fill TRAIN.bert_pretrain with one of these, or a local path):
  * BERT base (recommended default):
        google-bert/bert-base-uncased
        https://huggingface.co/google-bert/bert-base-uncased
  * Paper's literal citation [22] = ClipBERT (Lei et al., CVPR'21):
        https://github.com/jayleicn/ClipBERT

NOTE — to actually RUN this branch you also need:
  1. `pip install transformers` in the runtime env (e.g. hku_mtl).
  2. The dataset to tokenize queries with BERT's WordPiece tokenizer (input ids must be
     BERT ids, not CLIP-BPE ids). Pad token id stays 0, so the existing pad_mask
     (word == 0) is still correct.
Until then the model keeps using the CLIP text encoder (cfg.text_encoder = 'clip').
"""

import torch
import torch.nn as nn

from transformers import BertModel  # imported only when the BERT branch is selected


class BertTextEncoder(nn.Module):
    """Wrap a HuggingFace BERT as GeoLanG's text encoder.

    forward(ids) -> (token_emb, sentence_emb)
        token_emb     : [B, L, out_dim]   token embeddings  (Ct)
        sentence_emb  : [B, out_dim]      sentence embedding (Cs)
    """

    def __init__(self, pretrain="google-bert/bert-base-uncased", out_dim=512, freeze=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain)
        hidden = self.bert.config.hidden_size            # 768 for bert-base
        # Project BERT's 768-d outputs to GeoLanG's text width (word_dim, e.g. 512).
        self.tok_proj = nn.Linear(hidden, out_dim)
        self.sent_proj = nn.Linear(hidden, out_dim)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, ids):
        # Padding mask from BERT ids (pad id = 0). attention_mask: 1 = keep, 0 = pad.
        attn_mask = (ids != 0).long()
        out = self.bert(input_ids=ids, attention_mask=attn_mask)
        token_emb = self.tok_proj(out.last_hidden_state)         # [B, L, out_dim]
        # Sentence embedding: use the pooled [CLS] representation.
        sent = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        sentence_emb = self.sent_proj(sent)                      # [B, out_dim]
        return token_emb, sentence_emb
