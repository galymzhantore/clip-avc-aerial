"""ERA (Event Recognition in Aerial videos, Mou et al. 2020) — 25 classes."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from data.video_dataset import VideoClipDataset

ERA_CLASSES: Sequence[str] = (
    "post_earthquake",
    "flood",
    "fire",
    "landslide",
    "mudslide",
    "traffic_collision",
    "traffic_congestion",
    "harvesting",
    "ploughing",
    "constructing",
    "police_chase",
    "conflict",
    "baseball",
    "basketball",
    "boating",
    "cycling",
    "running",
    "soccer",
    "swimming",
    "car_racing",
    "party",
    "concert",
    "parade_protest",
    "religious_activity",
    "non_event",
)


def era_dataset(
    root: str | Path,
    split: str = "train",
    n_frames: int = 16,
    clip_frames: int | None = None,
    swin_frames: int | None = None,
    image_size: int = 224,
    random_crop: bool = True,
    horizontal_flip: bool = True,
    color_jitter: bool = True,
) -> VideoClipDataset:
    return VideoClipDataset(
        root=root,
        classes=ERA_CLASSES,
        split=split,
        n_frames=n_frames,
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        image_size=image_size,
        random_crop=random_crop,
        horizontal_flip=horizontal_flip,
        color_jitter=color_jitter,
    )
