"""Context-Enriched Transformer (architecture.md §8).

N=2 joint self-attention layers over concat([V; S]) with modality-type embeddings.
"""
from __future__ import annotations

import torch
from torch import nn

from models.transformer_blocks import SelfAttentionBlock


class ContextEnrichedTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        max_tokens: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.type_vis = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_txt = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        nn.init.trunc_normal_(self.type_vis, std=0.02)
        nn.init.trunc_normal_(self.type_txt, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        v: torch.Tensor,
        s: torch.Tensor,
        text_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # v: (B, T, D)  s: (B, K, D)
        t, k = v.size(1), s.size(1)
        v_in = v + self.type_vis
        s_in = s + self.type_txt
        z = torch.cat([v_in, s_in], dim=1)  # (B, T+K, D)
        z = z + self.pos[:, : t + k]

        if text_attention_mask is not None:
            vis_mask = torch.zeros(v.size(0), t, dtype=torch.bool, device=v.device)
            txt_pad = text_attention_mask == 0  # True = pad position to mask
            key_pad = torch.cat([vis_mask, txt_pad], dim=1)
        else:
            key_pad = None

        for layer in self.layers:
            z = layer(z, key_padding_mask=key_pad)

        v_hat_seq = z[:, :t]
        s_hat_seq = z[:, t:]
        return v_hat_seq, s_hat_seq
