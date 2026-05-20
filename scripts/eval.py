"""Classification evaluation for CLIP-AVC.

    uv run python -m scripts.eval --dataset era --data-root /data/era \
        --checkpoint checkpoints/era/era_epoch050.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ERA_CLASSES, MOD20_CLASSES, collate_clip_batch, era_dataset, mod20_dataset
from models.clip_avc import CLIP_AVC, CLIP_AVC_Config
from utils.prompts import build_prompts


def _resolve_frame_args(args: argparse.Namespace, saved: dict) -> tuple[int, int]:
    if args.frames is not None:
        return args.frames, args.frames
    clip_frames = args.clip_frames or saved.get("clip_frames") or saved.get("n_frames") or 8
    swin_frames = args.swin_frames or saved.get("swin_frames") or saved.get("n_frames") or 16
    return clip_frames, swin_frames


def _build_dataset(name: str, root: Path, split: str, clip_frames: int, swin_frames: int):
    common = dict(
        root=root,
        split=split,
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        random_crop=False,
        horizontal_flip=False,
        color_jitter=False,
    )
    if name == "era":
        return era_dataset(**common)
    if name == "mod20":
        return mod20_dataset(**common)
    raise ValueError(name)


def _classes(name: str):
    return list(ERA_CLASSES if name == "era" else MOD20_CLASSES)


def _config_from_checkpoint(saved: dict, clip_frames: int, swin_frames: int) -> CLIP_AVC_Config:
    cfg_dict = dict(saved)
    cfg_dict.pop("n_frames", None)
    cfg_dict["clip_frames"] = clip_frames
    cfg_dict["swin_frames"] = swin_frames
    return CLIP_AVC_Config(**cfg_dict)


def _score_with_context(
    model: CLIP_AVC,
    v: torch.Tensor,
    text,
    amp: bool,
) -> torch.Tensor:
    b, t, d = v.shape
    n_classes, k = text.tokens.shape[:2]
    v_flat = v[:, None].expand(b, n_classes, t, d).reshape(b * n_classes, t, d)
    s_flat = text.tokens[None].expand(b, n_classes, k, d).reshape(b * n_classes, k, d)
    mask_flat = text.attention_mask[None].expand(b, n_classes, k).reshape(b * n_classes, k)

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
        if model.config.use_context_transformer:
            v_seq, s_seq = model.context(v_flat, s_flat, text_attention_mask=mask_flat)
        else:
            v_seq, s_seq = v_flat, s_flat
        v_hat = F.normalize(v_seq.mean(dim=1), dim=-1)
        mask = mask_flat.unsqueeze(-1).type_as(s_seq)
        s_hat = (s_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        s_hat = F.normalize(s_hat, dim=-1)
        sims = (v_hat * s_hat).sum(dim=-1).reshape(b, n_classes)
    return sims


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["era", "mod20"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--frames", type=int, default=None, help="Backward-compatible alias that sets both paths.")
    p.add_argument("--clip-frames", type=int, default=None)
    p.add_argument("--swin-frames", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    device = torch.device("cuda")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt.get("config", {}) or {}
    clip_frames, swin_frames = _resolve_frame_args(args, saved)
    cfg = _config_from_checkpoint(saved, clip_frames, swin_frames)
    model = CLIP_AVC(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    classes = _classes(args.dataset)
    classifier = None
    if "classifier" in ckpt:
        classifier = nn.Linear(cfg.embed_dim, len(classes)).to(device)
        classifier.load_state_dict(ckpt["classifier"])
        classifier.eval()

    prompts = build_prompts(classes)
    toks = model.bert.tokenize(prompts, max_length=cfg.max_text_tokens)
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.inference_mode():
        text = model.encode_text(
            toks["input_ids"],
            toks["attention_mask"],
            toks.get("token_type_ids"),
        )

    ds = _build_dataset(args.dataset, args.data_root, args.split, clip_frames, swin_frames)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_clip_batch,
        pin_memory=True,
    )
    if args.workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(ds, **loader_kwargs)

    correct, total = 0, 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval"):
            frames = batch["clip_frames"].to(device, non_blocking=True)
            swin_video = batch["swin_video"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                v, _ = model.encode_visual(frames, swin_video)
                if classifier is not None:
                    sims = classifier(v.mean(dim=1))
                else:
                    sims = _score_with_context(model, v, text, args.amp)
            preds = sims.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    acc = correct / max(total, 1)
    print(f"{args.dataset} {args.split} accuracy: {acc*100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
