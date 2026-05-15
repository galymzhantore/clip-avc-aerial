"""MOD20 (Multi-Aerial Action Dataset, Choi et al. 2020) — 20 action classes."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from data.video_dataset import VideoClipDataset

MOD20_CLASSES: Sequence[str] = (
    "biking",
    "climbing",
    "diving",
    "fencing",
    "golf",
    "horse_riding",
    "kayaking",
    "running",
    "skateboarding",
    "skiing",
    "soccer",
    "surfing",
    "swimming",
    "tennis",
    "throwing",
    "trampoline",
    "volleyball",
    "walking",
    "weightlifting",
    "yoga",
)


def mod20_dataset(
    root: str | Path,
    split: str = "train",
    n_frames: int = 16,
    image_size: int = 224,
    random_crop: bool = True,
    horizontal_flip: bool = True,
) -> VideoClipDataset:
    return VideoClipDataset(
        root=root,
        classes=MOD20_CLASSES,
        split=split,
        n_frames=n_frames,
        image_size=image_size,
        random_crop=random_crop,
        horizontal_flip=horizontal_flip,
    )
