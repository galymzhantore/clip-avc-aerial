"""Training loop for CLIP-AVC on ERA / MOD20.

    uv run python -m scripts.train --dataset era --data-root /path/to/era
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ERA_CLASSES, MOD20_CLASSES, collate_clip_batch, era_dataset, mod20_dataset
from models.clip_avc import CLIP_AVC, CLIP_AVC_Config
from utils.prompts import build_prompts


def _resolve_frame_args(args: argparse.Namespace) -> tuple[int, int]:
    if args.frames is not None:
        if args.clip_frames is None:
            args.clip_frames = args.frames
        if args.swin_frames is None:
            args.swin_frames = args.frames
    return args.clip_frames or 8, args.swin_frames or 16


def _build_dataset(
    name: str,
    root: Path,
    split: str,
    clip_frames: int,
    swin_frames: int,
    color_jitter: bool = True,
):
    common = dict(
        root=root,
        split=split,
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        color_jitter=color_jitter,
    )
    if name == "era":
        return era_dataset(**common)
    if name == "mod20":
        return mod20_dataset(**common)
    raise ValueError(name)


def _classes(name: str):
    return list(ERA_CLASSES if name == "era" else MOD20_CLASSES)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader_kwargs(args: argparse.Namespace) -> dict:
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_clip_batch,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )
    if args.workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return kwargs


def _build_scheduler(args: argparse.Namespace, optim, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs * max(steps_per_epoch, 1)
        )
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optim, step_size=args.lr_step_epochs, gamma=args.lr_gamma
        )
    raise ValueError(args.scheduler)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["era", "mod20"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-c", type=float, default=1.0)
    p.add_argument("--frames", type=int, default=None, help="Backward-compatible alias that sets both paths.")
    p.add_argument("--clip-frames", type=int, default=None, help="Paper default: 8.")
    p.add_argument("--swin-frames", type=int, default=None, help="Paper default: 16.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--scheduler", choices=["step", "cosine", "none"], default="step")
    p.add_argument("--lr-step-epochs", type=int, default=15)
    p.add_argument("--lr-gamma", type=float, default=0.1)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--checkpoint-video-swin", action="store_true")
    p.add_argument("--color-jitter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cross-transformer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--context-transformer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lr-loss", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    clip_frames, swin_frames = _resolve_frame_args(args)
    if args.lr_loss and not args.context_transformer:
        p.error("--lr-loss requires --context-transformer because L_r is defined on refined features.")
    _seed_everything(args.seed)
    device = torch.device("cuda")

    cfg = CLIP_AVC_Config(
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        use_cross_transformer=args.cross_transformer,
        use_context_transformer=args.context_transformer,
        checkpoint_video_swin=args.checkpoint_video_swin,
    )
    model = CLIP_AVC(cfg).to(device)

    train_ds = _build_dataset(
        args.dataset,
        args.data_root,
        "train",
        clip_frames,
        swin_frames,
        color_jitter=args.color_jitter,
    )
    train_loader = DataLoader(train_ds, **_loader_kwargs(args))

    classes = _classes(args.dataset)
    prompts = build_prompts(classes)
    class_tokens = model.bert.tokenize(prompts, max_length=cfg.max_text_tokens)
    class_tokens = {k: v.to(device) for k, v in class_tokens.items()}

    use_classifier = not args.lc and not args.lr_loss
    classifier = nn.Linear(cfg.embed_dim, len(classes)).to(device) if use_classifier else None
    params = list(model.trainable_parameters())
    if classifier is not None:
        params += list(classifier.parameters())

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = _build_scheduler(args, optim, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    out_dir = args.out / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    model.train()
    if classifier is not None:
        classifier.train()

    print(
        "config:",
        {
            "dataset": args.dataset,
            "data_root": str(args.data_root),
            "clip_frames": clip_frames,
            "swin_frames": swin_frames,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "amp": args.amp,
            "cross_transformer": args.cross_transformer,
            "context_transformer": args.context_transformer,
            "lc": args.lc,
            "lr_loss": args.lr_loss,
            "classifier": classifier is not None,
        },
    )

    for epoch in range(args.epochs):
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        running, n_seen = 0.0, 0
        for batch in bar:
            labels = batch["label"].to(device, non_blocking=True)
            input_ids = class_tokens["input_ids"][labels]
            attn = class_tokens["attention_mask"][labels]
            tti = class_tokens.get("token_type_ids")
            if tti is not None:
                tti = tti[labels]

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                out = model(
                    frames=batch["clip_frames"].to(device, non_blocking=True),
                    clip_video=batch["swin_video"].to(device, non_blocking=True),
                    input_ids=input_ids,
                    attention_mask=attn,
                    token_type_ids=tti,
                )
                if classifier is None:
                    losses = model.compute_losses(
                        out, lam=args.lambda_c, use_lc=args.lc, use_lr=args.lr_loss
                    )
                else:
                    ce = F.cross_entropy(classifier(out.v_bar), labels)
                    losses = {"loss": ce, "L_r": ce.detach().new_zeros(()), "L_c": ce.detach().new_zeros(())}
                loss = losses["loss"]

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optim)
            scaler.update()
            if sched is not None and args.scheduler == "cosine":
                sched.step()

            running += loss.item() * labels.size(0)
            n_seen += labels.size(0)
            bar.set_postfix(
                loss=f"{running/n_seen:.4f}",
                Lr=f"{losses['L_r'].item():.4f}",
                Lc=f"{losses['L_c'].item():.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}",
            )

        if sched is not None and args.scheduler == "step":
            sched.step()

        ckpt = out_dir / f"{args.dataset}_epoch{epoch+1:03d}.pt"
        payload = {"model": model.state_dict(), "config": cfg.__dict__, "epoch": epoch + 1}
        if classifier is not None:
            payload["classifier"] = classifier.state_dict()
        torch.save(payload, ckpt)


if __name__ == "__main__":
    main()
