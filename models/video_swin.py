"""3-D Video Swin (Swin-B / Kinetics-400) Stage-4 grid extractor + projection to D."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import Swin3D_B_Weights, swin3d_b


class VideoSwinEncoder(nn.Module):
    """Returns the Stage-4 spatiotemporal grid, projected to the joint dim D.

    Input  : (B, 3, T_f, H, W) — clip with T_f = 16 frames at 224x224.
    Output : (B, T*H'*W', D)   — T = 8, H' = W' = 7 for the standard config.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        pretrained: bool = True,
        weights: str = "KINETICS400_V1",
        gradient_checkpointing: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if pretrained:
            w = Swin3D_B_Weights[weights]
        else:
            w = None
        backbone = swin3d_b(weights=w)
        self.patch_embed = backbone.patch_embed
        self.pos_drop = backbone.pos_drop
        self.features = backbone.features
        self.norm = backbone.norm
        self.out_channels: int = backbone.num_features  # 1024 for Swin-B
        self.projection = nn.Linear(self.out_channels, embed_dim)
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for module in (self.patch_embed, self.pos_drop, self.features, self.norm):
                for p in module.parameters():
                    p.requires_grad = False
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for module in (self.patch_embed, self.pos_drop, self.features, self.norm):
                module.eval()
        return self

    def _forward_impl(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, C, T_f, H, W)
        h = self.patch_embed(clip)
        h = self.pos_drop(h)
        h = self.features(h)
        h = self.norm(h)  # (B, T', H', W', C_s)
        b, t, hh, ww, c = h.shape
        h = h.reshape(b, t * hh * ww, c)  # (B, T'*H'*W', C_s)
        h = self.projection(h)            # (B, T'*H'*W', D)
        return h

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                h = self.patch_embed(clip)
                h = self.pos_drop(h)
                h = self.features(h)
                h = self.norm(h)
            b, t, hh, ww, c = h.shape
            h = h.reshape(b, t * hh * ww, c)
            return self.projection(h)
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, clip, use_reentrant=False)
        return self._forward_impl(clip)
