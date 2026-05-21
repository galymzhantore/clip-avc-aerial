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


def _build_dataset(
    name: str,
    root: Path,
    split: str,
    clip_frames: int,
    swin_frames: int,
    resize_size: int,
):
    common = dict(
        root=root,
        split=split,
        clip_frames=clip_frames,
        swin_frames=swin_frames,
        resize_size=resize_size,
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


def _config_from_checkpoint(
    saved: dict,
    clip_frames: int,
    swin_frames: int,
    state_dict: dict[str, torch.Tensor],
) -> CLIP_AVC_Config:
    cfg_dict = dict(saved)
    cfg_dict.pop("n_frames", None)
    cfg_dict["clip_frames"] = clip_frames
    cfg_dict["swin_frames"] = swin_frames
    if "text_encoder" not in cfg_dict and any(k.startswith("bert.bert.") for k in state_dict):
        cfg_dict["text_encoder"] = "bert"
    return CLIP_AVC_Config(**cfg_dict)


def _masked_text_mean(tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(tokens)
    return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


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


def _score_cached(
    model: CLIP_AVC,
    v: torch.Tensor,
    class_text: torch.Tensor,
    score_mode: str,
) -> torch.Tensor:
    if score_mode in {"coarse", "visual_coarse_text_refined"}:
        visual = v.mean(dim=1)
    else:
        visual = model.encode_visual_only_refined(v)

    visual = F.normalize(visual, dim=-1)
    class_text = F.normalize(class_text, dim=-1)
    return visual @ class_text.T


def _classifier_visual(model: CLIP_AVC, v: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "refined":
        return model.encode_visual_only_refined(v)
    # Existing classifier checkpoints, and the paper's dense visual head, use
    # visual features that do not require knowing the target class text.
    return v.mean(dim=1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["era", "mod20"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--frames", type=int, default=None, help="Backward-compatible alias that sets both paths.")
    p.add_argument("--clip-frames", type=int, default=None)
    p.add_argument("--swin-frames", type=int, default=None)
    p.add_argument("--resize-size", type=int, default=None, help="Resize short side before center crop.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--score-mode",
        choices=[
            "auto",
            "classifier",
            "refined_cached",
            "refined_pair",
            "coarse",
            "visual_refined_text_coarse",
            "visual_coarse_text_refined",
        ],
        default="auto",
        help="auto uses classifier checkpoints when present, otherwise cached refined CLIP-style scoring.",
    )
    args = p.parse_args()

    device = torch.device("cuda")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    saved = ckpt.get("config", {}) or {}
    clip_frames, swin_frames = _resolve_frame_args(args, saved)
    data_config = ckpt.get("data_config", {}) or {}
    resize_size = args.resize_size or data_config.get("resize_size") or 256
    cfg = _config_from_checkpoint(saved, clip_frames, swin_frames, ckpt["model"])
    model = CLIP_AVC(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    classes = _classes(args.dataset)
    classifier = None
    if "classifier" in ckpt:
        classifier = nn.Linear(cfg.embed_dim, len(classes)).to(device)
        classifier.load_state_dict(ckpt["classifier"])
        classifier.eval()
    train_config = ckpt.get("train_config", {}) or {}
    classifier_feature = train_config.get("classifier_feature", "coarse")

    prompts = build_prompts(classes)
    toks = model.bert.tokenize(prompts, max_length=cfg.max_text_tokens)
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.inference_mode():
        text = model.encode_text(
            toks["input_ids"],
            toks["attention_mask"],
            toks.get("token_type_ids"),
        )
        coarse_text = text.eos
        if cfg.use_context_transformer:
            refined_text = model.encode_text_only_refined(
                toks["input_ids"],
                toks["attention_mask"],
                toks.get("token_type_ids"),
            )
        else:
            refined_text = _masked_text_mean(text.tokens, text.attention_mask)

    score_mode = args.score_mode
    if score_mode == "auto":
        score_mode = "classifier" if classifier is not None else "refined_cached"
    if score_mode == "classifier" and classifier is None:
        raise ValueError("--score-mode classifier requested, but checkpoint has no classifier head.")
    print(f"score_mode: {score_mode}")

    ds = _build_dataset(args.dataset, args.data_root, args.split, clip_frames, swin_frames, resize_size)
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
                if score_mode == "classifier":
                    sims = classifier(_classifier_visual(model, v, classifier_feature))
                elif score_mode == "refined_pair":
                    sims = _score_with_context(model, v, text, args.amp)
                else:
                    class_text = refined_text
                    if score_mode in {"coarse", "visual_refined_text_coarse"}:
                        class_text = coarse_text
                    sims = _score_cached(model, v, class_text, score_mode)
            preds = sims.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()

    acc = correct / max(total, 1)
    print(f"{args.dataset} {args.split} accuracy: {acc*100:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
