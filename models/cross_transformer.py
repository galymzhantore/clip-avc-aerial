"""Cross-Transformer (architecture.md §7).

M=2 cross-attention layers: queries from temporal-CLIP, keys/values from Video Swin.
"""
from __future__ import annotations

import torch
from torch import nn

from models.transformer_blocks import CrossAttentionBlock


class CrossTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # query: (B, T_q, D)  — temporal-CLIP features
        # kv:    (B, T_kv, D) — Swin space-time tokens
        x = query
        for layer in self.layers:
            x = layer(x, kv)
        return x
