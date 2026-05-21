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


def _multi_positive_cross_entropy(logits: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
    positives = positives.to(dtype=logits.dtype)
    positives = positives / positives.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -(positives * log_probs).sum(dim=1).mean()


def bidirectional_info_nce(
    v: torch.Tensor,
    s: torch.Tensor,
    logit_scale: torch.Tensor,
    labels: torch.Tensor | None = None,
) -> torch.Tensor:
    v_n = F.normalize(v, dim=-1)
    s_n = F.normalize(s, dim=-1)
    logits_v2s = logit_scale * v_n @ s_n.t()
    logits_s2v = logits_v2s.t()

    if labels is None:
        targets = torch.arange(v.size(0), device=v.device)
        l_v2s = F.cross_entropy(logits_v2s, targets)
        l_s2v = F.cross_entropy(logits_s2v, targets)
    else:
        positives = labels[:, None] == labels[None, :]
        l_v2s = _multi_positive_cross_entropy(logits_v2s, positives)
        l_s2v = _multi_positive_cross_entropy(logits_s2v, positives.t())
    return 0.5 * (l_v2s + l_s2v)
