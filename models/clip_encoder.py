"""CLIP ViT-B/32 wrapper that returns per-frame joint-space features."""
from __future__ import annotations

from typing import Sequence

import clip
import torch
from torch import nn


class CLIPViTEncoder(nn.Module):
    """Per-frame ViT encoder using the released OpenAI CLIP checkpoint.

    Input  : (B, T, 3, H, W) — frames already normalised with CLIP mean/std.
    Output : (B, T, D)       — D = 512 for ViT-B/32, post joint-space projection.
    """

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        download_root: str | None = None,
        trainable: bool = False,
    ):
        super().__init__()
        model, _ = clip.load(model_name, device="cpu", jit=False, download_root=download_root)
        self.visual = model.visual.float()
        self.embed_dim: int = self.visual.output_dim
        self.trainable = trainable
        if not self.trainable:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def train(self, mode: bool = True):
        if not self.trainable:
            # Keep frozen weights in eval mode regardless of parent .train() calls.
            return super().train(False)
        return super().train(mode)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t = frames.shape[0], frames.shape[1]
        flat = frames.reshape(b * t, *frames.shape[2:])
        if self.trainable:
            u = self.visual(flat)
        else:
            with torch.no_grad():
                u = self.visual(flat)
        return u.reshape(b, t, -1)

    @staticmethod
    def normalize_mean_std() -> tuple[Sequence[float], Sequence[float]]:
        return (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
