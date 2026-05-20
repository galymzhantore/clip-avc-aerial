"""Frozen text encoders used by CLIP-AVC."""
from __future__ import annotations

from dataclasses import dataclass

import clip
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
        self.bert.eval()

    def train(self, mode: bool = True):
        # Keep frozen BERT deterministic while leaving the projection trainable.
        super().train(mode)
        self.bert.eval()
        return self

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


class CLIPTextEncoder(nn.Module):
    """Frozen OpenAI CLIP text tower with token-level joint-space features.

    The CLIP-AVC paper describes the text tower as BERT, but also says it is
    pretrained on CLIP's 400M WIT image-text pairs. That pretraining belongs to
    CLIP's own text transformer, so this path preserves the WIT-aligned text
    space used by CLIP zero-shot classification.
    """

    def __init__(self, model_name: str = "ViT-B/32", embed_dim: int = 512):
        super().__init__()
        model, _ = clip.load(model_name, device="cpu", jit=False)
        model = model.float()

        self.context_length = model.context_length
        self.token_embedding = model.token_embedding
        self.positional_embedding = model.positional_embedding
        self.transformer = model.transformer
        self.ln_final = model.ln_final
        self.text_projection = model.text_projection
        self.embed_dim = embed_dim

        if self.text_projection.shape[1] != embed_dim:
            raise ValueError(
                f"CLIP text projection outputs {self.text_projection.shape[1]} dims, "
                f"but embed_dim={embed_dim}."
            )

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        # Keep the CLIP text tower frozen and deterministic.
        return super().train(False)

    def tokenize(self, prompts: list[str], max_length: int = 77) -> dict[str, torch.Tensor]:
        if max_length != self.context_length:
            raise ValueError(
                f"OpenAI CLIP text transformer requires context_length={self.context_length}; "
                f"got max_length={max_length}."
            )
        input_ids = clip.tokenize(prompts, context_length=self.context_length, truncate=True)
        eot_idx = input_ids.argmax(dim=-1)
        positions = torch.arange(self.context_length, device=input_ids.device)
        return {
            "input_ids": input_ids,
            "attention_mask": (positions.unsqueeze(0) <= eot_idx.unsqueeze(1)).long(),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> TextEncoderOutput:
        del token_type_ids
        with torch.no_grad():
            x = self.token_embedding(input_ids)
            x = x + self.positional_embedding.to(dtype=x.dtype, device=x.device)
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.ln_final(x)
            projected = x @ self.text_projection

        # CLIP's EOT token is the highest token id in each sequence.
        eot_idx = input_ids.argmax(dim=-1)
        batch_idx = torch.arange(projected.size(0), device=projected.device)
        eos = projected[batch_idx, eot_idx]
        return TextEncoderOutput(tokens=projected, eos=eos, attention_mask=attention_mask)
