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
from models.clip_avc import CLIP_AVC, CLIP_AVC_Config, CLIP_AVC_Outputs
from models.clip_encoder import SUPPORTED_IMAGENET_VIT_MODELS, SUPPORTED_VISUAL_PRETRAINING
from models.video_swin import SUPPORTED_VIDEO_MODELS
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
    resize_size: int,
    visual_pretraining: str,
    color_jitter: bool = True,
):
    common = dict(
        root=root,
        split=split,
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        resize_size=resize_size,
        clip_normalization="imagenet" if visual_pretraining == "imagenet" else "clip",
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


def _classifier_features(out: CLIP_AVC_Outputs, mode: str, use_context_transformer: bool) -> torch.Tensor:
    if mode == "auto":
        return out.v_hat if use_context_transformer else out.v_bar
    if mode == "refined":
        return out.v_hat
    if mode == "coarse":
        return out.v_bar
    raise ValueError(mode)


def _save_checkpoint(
    out_dir: Path,
    dataset: str,
    epoch: int,
    payload: dict,
    keep_checkpoints: int,
) -> None:
    ckpt = out_dir / f"{dataset}_epoch{epoch:03d}.pt"
    tmp = ckpt.with_suffix(ckpt.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(ckpt)

    if keep_checkpoints > 0:
        checkpoints = sorted(out_dir.glob(f"{dataset}_epoch*.pt"))
        for old in checkpoints[:-keep_checkpoints]:
            old.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["era", "mod20"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--micro-batch-size",
        type=int,
        default=0,
        help="Split model forward passes while keeping InfoNCE over --batch-size samples.",
    )
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-c", type=float, default=1.0)
    p.add_argument("--frames", type=int, default=None, help="Backward-compatible alias that sets both paths.")
    p.add_argument("--clip-frames", type=int, default=None, help="Paper default: 8.")
    p.add_argument("--swin-frames", type=int, default=None, help="Paper default: 16.")
    p.add_argument("--resize-size", type=int, default=256, help="Resize short side before 224 crop.")
    p.add_argument("--visual-pretraining", choices=SUPPORTED_VISUAL_PRETRAINING, default="wit")
    p.add_argument("--clip-model", type=str, default="ViT-B/32")
    p.add_argument("--imagenet-vit-model", choices=SUPPORTED_IMAGENET_VIT_MODELS, default="vit_b_32")
    p.add_argument("--text-encoder", choices=["clip", "bert"], default="clip")
    p.add_argument("--max-text-tokens", type=int, default=77)
    p.add_argument(
        "--refined-text-pooling",
        choices=["eos", "mean"],
        default="eos",
        help="Pool C-Trans text output with the EOS token or mean over real tokens.",
    )
    p.add_argument("--freeze-clip-vit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--freeze-video-swin-backbone", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--video-model", choices=SUPPORTED_VIDEO_MODELS, default="swin3d_b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save every N epochs plus the final epoch. Set 0 to save only the final epoch.",
    )
    p.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="Keep only the newest N saved checkpoints per dataset. Set 0 to keep all.",
    )
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
    p.add_argument(
        "--classifier-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the dense classification head used for paper accuracy; contrastive losses remain auxiliary.",
    )
    p.add_argument(
        "--classifier-mode",
        choices=["linear", "context"],
        default="linear",
        help="linear trains a dense head; context trains CE on target-conditioned C-Trans similarities.",
    )
    p.add_argument("--ce-weight", type=float, default=1.0)
    p.add_argument(
        "--context-logit-chunk-size",
        type=int,
        default=0,
        help="Chunk B*num_classes C-Trans pairs for --classifier-mode context. 0 = one chunk.",
    )
    p.add_argument(
        "--classifier-feature",
        choices=["auto", "coarse", "refined"],
        default="coarse",
        help="Visual representation used by the dense classifier.",
    )
    p.add_argument(
        "--contrastive-targets",
        choices=["instance", "class"],
        default="instance",
        help="instance matches CLIP-style one-positive InfoNCE; class treats same-label batch items as positives.",
    )
    args = p.parse_args()

    clip_frames, swin_frames = _resolve_frame_args(args)
    if args.lr_loss and not args.context_transformer:
        p.error("--lr-loss requires --context-transformer because L_r is defined on refined features.")
    if args.classifier_head and args.classifier_mode == "context" and not args.context_transformer:
        p.error("--classifier-mode context requires --context-transformer.")
    _seed_everything(args.seed)
    device = torch.device("cuda")

    cfg = CLIP_AVC_Config(
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        use_cross_transformer=args.cross_transformer,
        use_context_transformer=args.context_transformer,
        checkpoint_video_swin=args.checkpoint_video_swin,
        visual_pretraining=args.visual_pretraining,
        clip_model=args.clip_model,
        imagenet_vit_model=args.imagenet_vit_model,
        text_encoder=args.text_encoder,
        max_text_tokens=args.max_text_tokens,
        refined_text_pooling=args.refined_text_pooling,
        video_model=args.video_model,
        freeze_clip_vit=args.freeze_clip_vit,
        freeze_video_swin_backbone=args.freeze_video_swin_backbone,
    )
    model = CLIP_AVC(cfg).to(device)

    train_ds = _build_dataset(
        args.dataset,
        args.data_root,
        "train",
        clip_frames,
        swin_frames,
        args.resize_size,
        args.visual_pretraining,
        color_jitter=args.color_jitter,
    )
    train_loader = DataLoader(train_ds, **_loader_kwargs(args))

    classes = _classes(args.dataset)
    prompts = build_prompts(classes)
    class_tokens = model.bert.tokenize(prompts, max_length=cfg.max_text_tokens)
    class_tokens = {k: v.to(device) for k, v in class_tokens.items()}

    classifier = (
        nn.Linear(cfg.embed_dim, len(classes)).to(device)
        if args.classifier_head and args.classifier_mode == "linear"
        else None
    )
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
            "resize_size": args.resize_size,
            "visual_pretraining": args.visual_pretraining,
            "clip_model": args.clip_model,
            "imagenet_vit_model": args.imagenet_vit_model,
            "text_encoder": args.text_encoder,
            "max_text_tokens": args.max_text_tokens,
            "refined_text_pooling": args.refined_text_pooling,
            "freeze_clip_vit": args.freeze_clip_vit,
            "freeze_video_swin_backbone": args.freeze_video_swin_backbone,
            "video_model": args.video_model,
            "batch_size": args.batch_size,
            "micro_batch_size": args.micro_batch_size or args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "save_every": args.save_every,
            "keep_checkpoints": args.keep_checkpoints,
            "amp": args.amp,
            "cross_transformer": args.cross_transformer,
            "context_transformer": args.context_transformer,
            "lc": args.lc,
            "lr_loss": args.lr_loss,
            "classifier_head": args.classifier_head,
            "classifier_mode": args.classifier_mode,
            "ce_weight": args.ce_weight,
            "classifier_feature": args.classifier_feature,
            "contrastive_targets": args.contrastive_targets,
            "classifier": classifier is not None,
        },
    )

    for epoch in range(args.epochs):
        bar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        running, n_seen = 0.0, 0
        for batch in bar:
            labels = batch["label"].to(device, non_blocking=True)
            input_ids_all = class_tokens["input_ids"][labels]
            attn_all = class_tokens["attention_mask"][labels]
            tti_all = class_tokens.get("token_type_ids")
            if tti_all is not None:
                tti_all = tti_all[labels]
            frames_all = batch["clip_frames"].to(device, non_blocking=True)
            swin_all = batch["swin_video"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                micro = args.micro_batch_size or args.batch_size
                if micro >= labels.size(0):
                    out = model(
                        frames=frames_all,
                        clip_video=swin_all,
                        input_ids=input_ids_all,
                        attention_mask=attn_all,
                        token_type_ids=tti_all,
                    )
                else:
                    chunks: list[CLIP_AVC_Outputs] = []
                    for start in range(0, labels.size(0), micro):
                        end = start + micro
                        tti = None if tti_all is None else tti_all[start:end]
                        chunks.append(
                            model(
                                frames=frames_all[start:end],
                                clip_video=swin_all[start:end],
                                input_ids=input_ids_all[start:end],
                                attention_mask=attn_all[start:end],
                                token_type_ids=tti,
                            )
                        )
                    out = CLIP_AVC_Outputs(
                        v_bar=torch.cat([c.v_bar for c in chunks], dim=0),
                        s_eos=torch.cat([c.s_eos for c in chunks], dim=0),
                        v_hat=torch.cat([c.v_hat for c in chunks], dim=0),
                        s_hat=torch.cat([c.s_hat for c in chunks], dim=0),
                        v_seq=torch.cat([c.v_seq for c in chunks], dim=0),
                        logit_scale=chunks[0].logit_scale,
                    )
                losses = model.compute_losses(
                    out,
                    lam=args.lambda_c,
                    use_lc=args.lc,
                    use_lr=args.lr_loss,
                    labels=labels if args.contrastive_targets == "class" else None,
                )
                ce = out.v_bar.new_zeros(())
                if args.classifier_head:
                    if args.classifier_mode == "linear":
                        feats = _classifier_features(
                            out,
                            args.classifier_feature,
                            args.context_transformer,
                        )
                        ce = F.cross_entropy(classifier(feats), labels)
                    else:
                        class_text = model.encode_text(
                            class_tokens["input_ids"],
                            class_tokens["attention_mask"],
                            class_tokens.get("token_type_ids"),
                        )
                        logits = model.context_similarity_logits(
                            out.v_seq,
                            class_text.tokens,
                            class_text.attention_mask,
                            chunk_size=args.context_logit_chunk_size,
                        )
                        ce = F.cross_entropy(logits, labels)
                loss = args.ce_weight * ce + losses["loss"]

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
                CE=f"{ce.item():.4f}",
                Lr=f"{losses['L_r'].item():.4f}",
                Lc=f"{losses['L_c'].item():.4f}",
                lr=f"{optim.param_groups[0]['lr']:.2e}",
            )

        if sched is not None and args.scheduler == "step":
            sched.step()

        epoch_num = epoch + 1
        should_save = epoch_num == args.epochs or (
            args.save_every > 0 and epoch_num % args.save_every == 0
        )
        if should_save:
            payload = {
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "data_config": {
                    "resize_size": args.resize_size,
                    "clip_normalization": "imagenet" if args.visual_pretraining == "imagenet" else "clip",
                },
                "train_config": {
                    "classifier_head": args.classifier_head,
                    "classifier_mode": args.classifier_mode,
                    "ce_weight": args.ce_weight,
                    "classifier_feature": args.classifier_feature,
                    "context_logit_chunk_size": args.context_logit_chunk_size,
                },
                "epoch": epoch_num,
            }
            if classifier is not None:
                payload["classifier"] = classifier.state_dict()
            _save_checkpoint(
                out_dir,
                args.dataset,
                epoch_num,
                payload,
                keep_checkpoints=args.keep_checkpoints,
            )


if __name__ == "__main__":
    main()
