"""ED-STANCE model: BERTweet with dependency-aware multi-label heads.

The promotion head consumes the model's predicted suffering probability, not
the gold suffering label. This avoids teacher-forcing leakage at inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


@dataclass
class LossWeights:
    auxiliary: float = 0.15
    hierarchy: float = 0.03
    contrastive: float = 0.10
    temperature: float = 0.08


@dataclass
class EDStanceConfig:
    encoder_name: str = "vinai/bertweet-base"
    dropout: float = 0.15
    hierarchy_penalty_weight: float = 0.0
    use_dependency_head: bool = True
    use_hierarchy_penalty: bool = False


class EDStanceModel(nn.Module):
    def __init__(self, config: EDStanceConfig):
        super().__init__();self.config=config
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.suffering_head = nn.Linear(hidden, 1)
        self.promotion_text_head = nn.Linear(hidden, 1)
        self.dependency = nn.Linear(1, 1, bias=False)
        nn.init.zeros_(self.dependency.weight)
        self.projection = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 128))

    def _pool(self, outputs, attention_mask):
        return outputs.last_hidden_state[:, 0]

    def encode_projection(self, input_ids, attention_mask, **kwargs):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = self.dropout(self._pool(encoded, attention_mask))
        return F.normalize(self.projection(pooled), dim=-1)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        suffering_labels=None,
        promotion_labels=None,
        **kwargs,
    ):
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        pooled = self.dropout(self._pool(encoded, attention_mask))
        suffering_logits = self.suffering_head(pooled).squeeze(-1)
        suffering_probability = torch.sigmoid(suffering_logits).detach().unsqueeze(-1)
        promotion_logits = self.promotion_text_head(pooled).squeeze(-1)
        if self.config.use_dependency_head:
            promotion_logits = promotion_logits + self.dependency(suffering_probability).squeeze(-1)

        loss = None
        parts = {}
        if suffering_labels is not None and promotion_labels is not None:
            suffer_loss = F.binary_cross_entropy_with_logits(suffering_logits, suffering_labels.float())
            promote_loss = F.binary_cross_entropy_with_logits(promotion_logits, promotion_labels.float())
            loss = 0.5 * (suffer_loss + promote_loss)
            parts.update(suffering_bce=suffer_loss.detach(), promotion_bce=promote_loss.detach())

            if self.config.use_hierarchy_penalty:
                p_s = torch.sigmoid(suffering_logits)
                p_p = torch.sigmoid(promotion_logits)
                hierarchy = torch.square(F.relu(p_p - p_s)).mean()
                loss = loss + self.config.hierarchy_penalty_weight * hierarchy
                parts["soft_hierarchy"] = hierarchy.detach()

        return {
            "loss": loss,
            "suffering_logits": suffering_logits,
            "promotion_logits": promotion_logits,
            "loss_parts": parts,
        }
