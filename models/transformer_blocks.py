"""Post-norm Transformer building blocks shared by the three learned modules.

Equations (architecture.md §5.2):
    Y' = LN(MHA(X) + X)
    Y  = LN(FFN(Y') + Y')
"""
from __future__ import annotations

import torch
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttentionBlock(nn.Module):
    """Post-norm self-attention encoder layer."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0, ffn_expansion: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, expansion=ffn_expansion, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(attn_out + x)
        x = self.ln2(self.ffn(x) + x)
        return x


class CrossAttentionBlock(nn.Module):
    """Post-norm cross-attention encoder layer (queries from one stream, KV from another)."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0, ffn_expansion: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, expansion=ffn_expansion, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(query, kv, kv, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(attn_out + query)
        x = self.ln2(self.ffn(x) + x)
        return x
