"""Temporal Transformer (architecture.md §5.2).

L=3 self-attention layers over the per-frame CLIP feature sequence.
"""
from __future__ import annotations

import torch
from torch import nn

from models.transformer_blocks import SelfAttentionBlock


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 3,
        n_heads: int = 8,
        max_frames: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_frames, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads=n_heads, dropout=dropout) for _ in range(n_layers)]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B, T, D)
        t = u.size(1)
        x = u + self.pos[:, :t]
        for layer in self.layers:
            x = layer(x)
        return x
