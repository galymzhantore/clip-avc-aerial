"""decord-based video clip dataset.

Expected folder layout (the user arranges these):

    data_root/
        train/
            class_name_1/clip_001.mp4
            class_name_1/clip_002.mp4
            class_name_2/...
        val/
            class_name_1/...
        test/
            class_name_1/...

Each video is uniformly sub-sampled for both visual paths, resized then cropped
to ``image_size``, and returned twice: once normalised with CLIP mean/std for the
ViT path, once with the torchvision Swin3D weights mean/std for the Video Swin path.
The paper samples 8 frames for the CLIP/2-D path and 16 frames for the Video
Swin/3-D path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from models.clip_encoder import CLIPViTEncoder, ImageNetViTEncoder

CLIP_MEAN, CLIP_STD = CLIPViTEncoder.normalize_mean_std()
IMAGENET_MEAN, IMAGENET_STD = ImageNetViTEncoder.normalize_mean_std()
# Torchvision's Swin3D_B_Weights.KINETICS400_V1 preprocessing uses ImageNet
# normalization after resizing the short side to 256 and cropping to 224.
SWIN_MEAN = (0.485, 0.456, 0.406)
SWIN_STD = (0.229, 0.224, 0.225)

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@dataclass
class ClipSample:
    path: Path
    class_idx: int
    class_name: str


def _scan_split(root: Path, classes: Sequence[str]) -> list[ClipSample]:
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples: list[ClipSample] = []
    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.is_dir():
            continue
        for video in sorted(cls_dir.iterdir()):
            if video.suffix.lower() in VIDEO_SUFFIXES:
                samples.append(ClipSample(path=video, class_idx=cls_to_idx[cls], class_name=cls))
    return samples


class VideoClipDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        classes: Sequence[str],
        split: str = "train",
        n_frames: int = 16,
        clip_frames: int | None = None,
        swin_frames: int | None = None,
        image_size: int = 224,
        resize_size: int = 256,
        random_crop: bool = True,
        horizontal_flip: bool = True,
        color_jitter: bool = True,
        clip_normalization: str = "clip",
    ):
        self.root = Path(root) / split
        self.classes = list(classes)
        self.split = split
        self.clip_frames = clip_frames if clip_frames is not None else n_frames
        self.swin_frames = swin_frames if swin_frames is not None else n_frames
        self.image_size = image_size
        self.resize_size = resize_size
        if self.resize_size < self.image_size:
            raise ValueError("resize_size must be >= image_size")
        self.random_crop = random_crop and split == "train"
        self.horizontal_flip = horizontal_flip and split == "train"
        self.color_jitter = color_jitter and split == "train"
        if clip_normalization == "clip":
            self.clip_mean, self.clip_std = CLIP_MEAN, CLIP_STD
        elif clip_normalization == "imagenet":
            self.clip_mean, self.clip_std = IMAGENET_MEAN, IMAGENET_STD
        else:
            raise ValueError("clip_normalization must be 'clip' or 'imagenet'")
        self.samples = _scan_split(self.root, self.classes)
        if not self.samples:
            raise FileNotFoundError(
                f"No videos found under {self.root} for classes {self.classes!r}."
            )

    # ------------------------------------------------------------------ frame ops
    @staticmethod
    def _sample_indices(total: int, n_frames: int) -> np.ndarray:
        if total >= n_frames:
            return np.linspace(0, total - 1, n_frames).astype(np.int64)
        return np.concatenate(
            [np.arange(total), np.full(n_frames - total, total - 1, dtype=np.int64)]
        )

    def _read_frames(self, path: Path, n_frames: int) -> np.ndarray:
        vr = VideoReader(str(path), ctx=cpu(0))
        total = len(vr)
        idxs = self._sample_indices(total, n_frames)
        frames = vr.get_batch(list(idxs)).asnumpy()  # (T, H, W, 3) uint8 RGB
        return frames

    @staticmethod
    def _adjust_saturation(x: torch.Tensor, factor: float) -> torch.Tensor:
        # Rec. 601 luma coefficients, matching the usual PIL-style grayscale mix.
        gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
        return gray + factor * (x - gray)

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        if not self.color_jitter:
            return x
        brightness = float(torch.empty(1).uniform_(0.8, 1.2).item())
        contrast = float(torch.empty(1).uniform_(0.8, 1.2).item())
        saturation = float(torch.empty(1).uniform_(0.8, 1.2).item())

        ops = torch.randperm(3).tolist()
        for op in ops:
            if op == 0:
                x = x * brightness
            elif op == 1:
                mean = x.mean(dim=(1, 2, 3), keepdim=True)
                x = mean + contrast * (x - mean)
            else:
                x = self._adjust_saturation(x, saturation)
        return x.clamp_(0.0, 1.0)

    def _augment(self, frames: np.ndarray) -> np.ndarray:
        # frames: (T, H, W, 3) uint8
        t, h, w, _ = frames.shape
        short = min(h, w)
        scale = self.resize_size / short
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0  # (T, 3, H, W)
        x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        if self.random_crop:
            top = torch.randint(0, max(1, new_h - self.image_size + 1), (1,)).item()
            left = torch.randint(0, max(1, new_w - self.image_size + 1), (1,)).item()
        else:
            top = (new_h - self.image_size) // 2
            left = (new_w - self.image_size) // 2
        x = x[:, :, top : top + self.image_size, left : left + self.image_size]
        if self.horizontal_flip and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])
        x = self._color_jitter(x)
        return x  # (T, 3, H, W) float in [0,1]

    @staticmethod
    def _normalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
        m = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        s = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - m) / s

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        clip_raw = self._read_frames(sample.path, self.clip_frames)
        swin_raw = self._read_frames(sample.path, self.swin_frames)
        frames = np.concatenate([clip_raw, swin_raw], axis=0)
        x = self._augment(frames)  # (T_clip + T_swin, 3, H, W)
        clip_x = x[: self.clip_frames]
        swin_x = x[self.clip_frames :]
        clip_frames = self._normalize(clip_x, self.clip_mean, self.clip_std)  # (T_clip, 3, H, W)
        swin_video = self._normalize(swin_x, SWIN_MEAN, SWIN_STD)     # (T_swin, 3, H, W)
        swin_video = swin_video.permute(1, 0, 2, 3).contiguous()      # (3, T_swin, H, W)
        return {
            "clip_frames": clip_frames,
            "swin_video": swin_video,
            "label": sample.class_idx,
            "class_name": sample.class_name,
        }


def collate_clip_batch(batch: list[dict]) -> dict:
    return {
        "clip_frames": torch.stack([b["clip_frames"] for b in batch], dim=0),
        "swin_video": torch.stack([b["swin_video"] for b in batch], dim=0),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "class_name": [b["class_name"] for b in batch],
    }
