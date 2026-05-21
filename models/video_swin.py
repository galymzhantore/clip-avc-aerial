"""3-D video backbones that return token grids projected to the joint dim D."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.models.video import (
    MC3_18_Weights,
    MViT_V2_S_Weights,
    R2Plus1D_18_Weights,
    R3D_18_Weights,
    S3D_Weights,
    Swin3D_B_Weights,
    Swin3D_S_Weights,
    Swin3D_T_Weights,
    mc3_18,
    mvit_v2_s,
    r2plus1d_18,
    r3d_18,
    s3d,
    swin3d_b,
    swin3d_s,
    swin3d_t,
)


_SWIN_BACKBONES = {
    "swin3d_t": (swin3d_t, Swin3D_T_Weights),
    "swin3d_s": (swin3d_s, Swin3D_S_Weights),
    "swin3d_b": (swin3d_b, Swin3D_B_Weights),
}

_VIDEO_RESNET_BACKBONES = {
    "r3d_18": (r3d_18, R3D_18_Weights),
    "mc3_18": (mc3_18, MC3_18_Weights),
    "r2plus1d_18": (r2plus1d_18, R2Plus1D_18_Weights),
}

_S3D_BACKBONES = {
    "s3d": (s3d, S3D_Weights),
}

_MVIT_BACKBONES = {
    "mvit_v2_s": (mvit_v2_s, MViT_V2_S_Weights),
}

SUPPORTED_VIDEO_MODELS = tuple(
    list(_SWIN_BACKBONES)
    + list(_VIDEO_RESNET_BACKBONES)
    + list(_S3D_BACKBONES)
    + list(_MVIT_BACKBONES)
)


class VideoSwinEncoder(nn.Module):
    """Returns a spatiotemporal token grid, projected to the joint dim D.

    Input  : (B, 3, T_f, H, W) — clip with T_f = 16 frames at 224x224.
    Output : (B, T*H'*W', D)   — token count depends on the chosen backbone.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        model_name: str = "swin3d_b",
        pretrained: bool = True,
        weights: str = "KINETICS400_V1",
        gradient_checkpointing: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if model_name not in SUPPORTED_VIDEO_MODELS:
            raise ValueError(
                f"Unsupported video model {model_name!r}. "
                f"Choose one of {SUPPORTED_VIDEO_MODELS!r}."
            )

        self.model_name = model_name
        self.gradient_checkpointing = gradient_checkpointing
        self.freeze_backbone = freeze_backbone

        if model_name in _SWIN_BACKBONES:
            self.backbone_kind = "swin"
            factory, weights_enum = _SWIN_BACKBONES[model_name]
            backbone = factory(weights=weights_enum[weights] if pretrained else None)
            self.patch_embed = backbone.patch_embed
            self.pos_drop = backbone.pos_drop
            self.features = backbone.features
            self.norm = backbone.norm
            self.out_channels = backbone.num_features
            self._backbone_modules = (self.patch_embed, self.pos_drop, self.features, self.norm)
        elif model_name in _VIDEO_RESNET_BACKBONES:
            self.backbone_kind = "video_resnet"
            factory, weights_enum = _VIDEO_RESNET_BACKBONES[model_name]
            backbone = factory(weights=weights_enum[weights] if pretrained else None)
            self.stem = backbone.stem
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.out_channels = backbone.fc.in_features
            self._backbone_modules = (self.stem, self.layer1, self.layer2, self.layer3, self.layer4)
        elif model_name in _S3D_BACKBONES:
            self.backbone_kind = "s3d"
            factory, weights_enum = _S3D_BACKBONES[model_name]
            backbone = factory(weights=weights_enum[weights] if pretrained else None)
            self.features = backbone.features
            self.out_channels = backbone.classifier[1].in_channels
            self._backbone_modules = (self.features,)
        else:
            self.backbone_kind = "mvit"
            factory, weights_enum = _MVIT_BACKBONES[model_name]
            backbone = factory(weights=weights_enum[weights] if pretrained else None)
            self.conv_proj = backbone.conv_proj
            self.pos_encoding = backbone.pos_encoding
            self.blocks = backbone.blocks
            self.norm = backbone.norm
            self.out_channels = backbone.head[1].in_features
            self._backbone_modules = (self.conv_proj, self.pos_encoding, self.blocks, self.norm)

        self.projection = nn.Linear(self.out_channels, embed_dim)
        if self.freeze_backbone:
            for module in self._backbone_modules:
                for p in module.parameters():
                    p.requires_grad = False
                module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for module in self._backbone_modules:
                module.eval()
        return self

    @staticmethod
    def _grid_to_tokens(h: torch.Tensor) -> torch.Tensor:
        b, c, t, hh, ww = h.shape
        return h.permute(0, 2, 3, 4, 1).reshape(b, t * hh * ww, c)

    def _forward_features(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, C, T_f, H, W)
        if self.backbone_kind == "swin":
            h = self.patch_embed(clip)
            h = self.pos_drop(h)
            h = self.features(h)
            h = self.norm(h)  # (B, T', H', W', C)
            b, t, hh, ww, c = h.shape
            return h.reshape(b, t * hh * ww, c)

        if self.backbone_kind == "video_resnet":
            h = self.stem(clip)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.layer3(h)
            h = self.layer4(h)
            return self._grid_to_tokens(h)

        if self.backbone_kind == "s3d":
            h = self.features(clip)
            return self._grid_to_tokens(h)

        x = self.conv_proj(clip)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)
        return x[:, 1:]  # drop classifier token

    def _forward_impl(self, clip: torch.Tensor) -> torch.Tensor:
        return self.projection(self._forward_features(clip))

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                h = self._forward_features(clip)
            return self.projection(h)
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, clip, use_reentrant=False)
        return self._forward_impl(clip)
