"""ViT wrappers that return per-frame joint-space features."""
from __future__ import annotations

from typing import Sequence

import clip
import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, vit_b_16, vit_b_32


SUPPORTED_VISUAL_PRETRAINING = ("wit", "imagenet")
SUPPORTED_IMAGENET_VIT_MODELS = ("vit_b_32", "vit_b_16")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


class ImageNetViTEncoder(nn.Module):
    """Per-frame torchvision ViT encoder initialized from ImageNet weights.

    Torchvision ViT features are 768-D before the classification head, so this
    wrapper learns a projection into the 512-D CLIP-AVC joint space.
    """

    def __init__(
        self,
        model_name: str = "vit_b_32",
        embed_dim: int = 512,
        weights: str = "IMAGENET1K_V1",
        trainable: bool = False,
    ):
        super().__init__()
        if model_name == "vit_b_32":
            backbone = vit_b_32(weights=ViT_B_32_Weights[weights])
        elif model_name == "vit_b_16":
            backbone = vit_b_16(weights=ViT_B_16_Weights[weights])
        else:
            raise ValueError(
                f"Unsupported ImageNet ViT model {model_name!r}; "
                f"choose one of {SUPPORTED_IMAGENET_VIT_MODELS!r}."
            )

        self.backbone = backbone
        self.projection = nn.Linear(backbone.hidden_dim, embed_dim, bias=False)
        self.embed_dim = embed_dim
        self.trainable = trainable

        if not self.trainable:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.backbone.eval()
        return self

    def _forward_features(self, frames: torch.Tensor) -> torch.Tensor:
        x = self.backbone._process_input(frames)
        batch_class_token = self.backbone.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)
        return x[:, 0]

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, t = frames.shape[0], frames.shape[1]
        flat = frames.reshape(b * t, *frames.shape[2:])
        if self.trainable:
            u = self._forward_features(flat)
        else:
            with torch.no_grad():
                u = self._forward_features(flat)
        return self.projection(u).reshape(b, t, -1)

    @staticmethod
    def normalize_mean_std() -> tuple[Sequence[float], Sequence[float]]:
        return IMAGENET_MEAN, IMAGENET_STD
