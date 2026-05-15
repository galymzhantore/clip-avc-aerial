"""Bidirectional InfoNCE (architecture.md §10).

L_NCE = 0.5 * (L_{v->s} + L_{s->v}), each a cross-entropy over a cosine-similarity
matrix scaled by a learnable temperature.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_pair(v: torch.Tensor, s: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    v_n = F.normalize(v, dim=-1)
    s_n = F.normalize(s, dim=-1)
    logits = logit_scale * v_n @ s_n.t()  # (B, B)
    targets = torch.arange(v.size(0), device=v.device)
    return F.cross_entropy(logits, targets)


def bidirectional_info_nce(v: torch.Tensor, s: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    v_n = F.normalize(v, dim=-1)
    s_n = F.normalize(s, dim=-1)
    logits_v2s = logit_scale * v_n @ s_n.t()
    logits_s2v = logits_v2s.t()
    targets = torch.arange(v.size(0), device=v.device)
    l_v2s = F.cross_entropy(logits_v2s, targets)
    l_s2v = F.cross_entropy(logits_s2v, targets)
    return 0.5 * (l_v2s + l_s2v)
