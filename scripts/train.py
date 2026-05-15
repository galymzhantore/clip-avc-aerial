"""Training loop for CLIP-AVC on ERA / MOD20.

    uv run python -m scripts.train --dataset era --data-root /path/to/era
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ERA_CLASSES, MOD20_CLASSES, collate_clip_batch, era_dataset, mod20_dataset
from models.clip_avc import CLIP_AVC, CLIP_AVC_Config
from utils.prompts import build_prompts


def _build_dataset(name: str, root: Path, split: str, n_frames: int):
    common = dict(root=root, split=split, n_frames=n_frames)
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
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-c", type=float, default=1.0)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    cfg = CLIP_AVC_Config(n_frames=args.frames)
    model = CLIP_AVC(cfg).to(device)

    train_ds = _build_dataset(args.dataset, args.data_root, "train", args.frames)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_clip_batch,
        drop_last=True,
        pin_memory=True,
    )

    classes = _classes(args.dataset)
    prompts = build_prompts(classes)
    class_tokens = model.bert.tokenize(prompts, max_length=cfg.max_text_tokens)
    class_tokens = {k: v.to(device) for k, v in class_tokens.items()}

    optim = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * max(len(train_loader), 1))

    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    for epoch in range(args.epochs):
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        running, n_seen = 0.0, 0
        for batch in bar:
            labels = batch["label"].to(device)
            input_ids = class_tokens["input_ids"][labels]
            attn = class_tokens["attention_mask"][labels]
            tti = class_tokens.get("token_type_ids")
            if tti is not None:
                tti = tti[labels]

            out = model(
                frames=batch["clip_frames"].to(device, non_blocking=True),
                clip_video=batch["swin_video"].to(device, non_blocking=True),
                input_ids=input_ids,
                attention_mask=attn,
                token_type_ids=tti,
            )
            losses = model.compute_losses(out, lam=args.lambda_c)
            loss = losses["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
            optim.step()
            sched.step()

            running += loss.item() * labels.size(0)
            n_seen += labels.size(0)
            bar.set_postfix(
                loss=f"{running/n_seen:.4f}",
                Lr=f"{losses['L_r'].item():.4f}",
                Lc=f"{losses['L_c'].item():.4f}",
            )

        ckpt = out_dir / f"{args.dataset}_epoch{epoch+1:03d}.pt"
        torch.save({"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch + 1}, ckpt)


if __name__ == "__main__":
    main()
