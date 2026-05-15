"""Frozen BERT text encoder with a learned 768 -> D projection."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import BertModel, BertTokenizer


@dataclass
class TextEncoderOutput:
    tokens: torch.Tensor          # (B, K, D) — all token features (projected)
    eos: torch.Tensor             # (B, D)    — [SEP] / EOS token feature (projected)
    attention_mask: torch.Tensor  # (B, K)    — 1 = real token, 0 = pad


class BERTTextEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", embed_dim: int = 512):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)
        self.embed_dim = embed_dim
        self.projection = nn.Linear(self.bert.config.hidden_size, embed_dim, bias=False)

        for p in self.bert.parameters():
            p.requires_grad = False

    def tokenize(self, prompts: list[str], max_length: int = 32) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> TextEncoderOutput:
        with torch.no_grad():
            out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        hidden = out.last_hidden_state  # (B, K, 768)
        projected = self.projection(hidden)  # (B, K, D)

        # EOS = last real (non-pad) token: index = sum(attention_mask, dim=1) - 1
        last_idx = attention_mask.sum(dim=1).long() - 1  # (B,)
        batch_idx = torch.arange(projected.size(0), device=projected.device)
        eos = projected[batch_idx, last_idx]  # (B, D)

        return TextEncoderOutput(tokens=projected, eos=eos, attention_mask=attention_mask)
