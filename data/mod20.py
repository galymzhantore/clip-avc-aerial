"""MOD20 (Multi-Aerial Action Dataset, Choi et al. 2020) — 20 action classes."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from data.video_dataset import VideoClipDataset

MOD20_CLASSES: Sequence[str] = (
    "cycling",
    "motorbiking",
    "chainsawing_trees",
    "jetskiing",
    "fire_fighting",
    "dancing",
    "cutting_wood",
    "surfing",
    "fighting",
    "rock_climbing",
    "cliff_jumping",
    "figure_skating",
    "running",
    "kayaking",
    "backpacking",
    "skateboarding",
    "windsurfing",
    "standup_paddling",
    "skiing",
    "nfl_catches",
)


def mod20_dataset(
    root: str | Path,
    split: str = "train",
    n_frames: int = 16,
    clip_frames: int | None = None,
    swin_frames: int | None = None,
    image_size: int = 224,
    resize_size: int = 256,
    random_crop: bool = True,
    horizontal_flip: bool = True,
    color_jitter: bool = True,
) -> VideoClipDataset:
    return VideoClipDataset(
        root=root,
        classes=MOD20_CLASSES,
        split=split,
        n_frames=n_frames,
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        image_size=image_size,
        resize_size=resize_size,
        random_crop=random_crop,
        horizontal_flip=horizontal_flip,
        color_jitter=color_jitter,
    )
