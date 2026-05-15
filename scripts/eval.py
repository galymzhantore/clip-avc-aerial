"""Zero-shot-style classification evaluation (architecture.md §11).

    uv run python -m scripts.eval --dataset era --data-root /data/era \
        --checkpoint checkpoints/era/era_epoch050.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ERA_CLASSES, MOD20_CLASSES, collate_clip_batch, era_dataset, mod20_dataset
from models.clip_avc import CLIP_AVC, CLIP_AVC_Config
from utils.prompts import build_prompts


def _build_dataset(name: str, root: Path, split: str, n_frames: int):
    common = dict(root=root, split=split, n_frames=n_frames,
                  random_crop=False, horizontal_flip=False)
    if name == "era":
        return era_dataset(**common)
    if name == "mod20":
        return mod20_dataset(**common)
    raise ValueError(name)


def _classes(name: str):
    return list(ERA_CLASSES if name == "era" else MOD20_CLASSES)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["era", "mod20"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt.get("config", {}) or {}
    cfg = CLIP_AVC_Config(**{**saved, "n_frames": args.frames})
    model = CLIP_AVC(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    classes = _classes(args.dataset)
    prompts = build_prompts(classes)
    toks = model.bert.tokenize(prompts, max_length=cfg.max_text_tokens)
    toks = {k: v.to(device) for k, v in toks.items()}

    with torch.no_grad():
        class_protos = model.encode_text_only_refined(
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            token_type_ids=toks.get("token_type_ids"),
        )
        class_protos = F.normalize(class_protos, dim=-1)

    ds = _build_dataset(args.dataset, args.data_root, args.split, args.frames)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_clip_batch,
        pin_memory=True,
    )

    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            frames = batch["clip_frames"].to(device)
            swin_video = batch["swin_video"].to(device)
            labels = batch["label"].to(device)

            v, _ = model.encode_visual(frames, swin_video)
            zero_text = torch.zeros(frames.size(0), 1, cfg.embed_dim, device=device)
            zero_mask = torch.ones(frames.size(0), 1, dtype=torch.long, device=device)
            v_seq, _ = model.context(v, zero_text, text_attention_mask=zero_mask)
            v_hat = F.normalize(v_seq.mean(dim=1), dim=-1)

            sims = v_hat @ class_protos.t()
            preds = sims.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    acc = correct / max(total, 1)
    print(f"{args.dataset} {args.split} accuracy: {acc*100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
