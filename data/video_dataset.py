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

Each video is uniformly sub-sampled to ``n_frames`` frames, resized to ``image_size``,
and returned twice: once normalised with CLIP mean/std for the ViT path, once with
Kinetics (ImageNet-style) mean/std for the Video Swin path.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from decord import VideoReader, cpu
from torch.utils.data import Dataset

from models.clip_encoder import CLIPViTEncoder

CLIP_MEAN, CLIP_STD = CLIPViTEncoder.normalize_mean_std()
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
        image_size: int = 224,
        random_crop: bool = True,
        horizontal_flip: bool = True,
    ):
        self.root = Path(root) / split
        self.classes = list(classes)
        self.split = split
        self.n_frames = n_frames
        self.image_size = image_size
        self.random_crop = random_crop and split == "train"
        self.horizontal_flip = horizontal_flip and split == "train"
        self.samples = _scan_split(self.root, self.classes)
        if not self.samples:
            raise FileNotFoundError(
                f"No videos found under {self.root} for classes {self.classes!r}."
            )

    # ------------------------------------------------------------------ frame ops
    def _read_frames(self, path: Path) -> np.ndarray:
        vr = VideoReader(str(path), ctx=cpu(0))
        total = len(vr)
        if total >= self.n_frames:
            idxs = np.linspace(0, total - 1, self.n_frames).astype(np.int64)
        else:
            idxs = np.concatenate(
                [np.arange(total), np.full(self.n_frames - total, total - 1, dtype=np.int64)]
            )
        frames = vr.get_batch(list(idxs)).asnumpy()  # (T, H, W, 3) uint8 RGB
        return frames

    def _augment(self, frames: np.ndarray) -> np.ndarray:
        # frames: (T, H, W, 3) uint8
        t, h, w, _ = frames.shape
        short = min(h, w)
        scale = self.image_size / short
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
        frames = self._read_frames(sample.path)
        x = self._augment(frames)                      # (T, 3, H, W)
        clip_frames = self._normalize(x, CLIP_MEAN, CLIP_STD)         # (T, 3, H, W)
        swin_video = self._normalize(x, SWIN_MEAN, SWIN_STD)          # (T, 3, H, W)
        swin_video = swin_video.permute(1, 0, 2, 3).contiguous()      # (3, T, H, W)
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
