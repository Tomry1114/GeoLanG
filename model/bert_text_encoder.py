# CLIP-BERT text encoder. Needs `pip install transformers` and BERT-tokenized input ids
# (pad id = 0). Weights: google-bert/bert-base-uncased (or local path).

import torch
import torch.nn as nn

from transformers import BertModel


class BertTextEncoder(nn.Module):
    def __init__(self, pretrain="google-bert/bert-base-uncased", out_dim=512, freeze=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain)
        hidden = self.bert.config.hidden_size
        self.tok_proj = nn.Linear(hidden, out_dim)
        self.sent_proj = nn.Linear(hidden, out_dim)
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, ids):
        attn_mask = (ids != 0).long()
        out = self.bert(input_ids=ids, attention_mask=attn_mask)
        token_emb = self.tok_proj(out.last_hidden_state)
        sent = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        sentence_emb = self.sent_proj(sent)
        return token_emb, sentence_emb
