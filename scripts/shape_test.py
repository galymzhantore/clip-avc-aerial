"""End-to-end shape check matching the walkthrough in architecture.md §13.

    uv run python -m scripts.shape_test
"""
from __future__ import annotations

import argparse

import torch

from models.clip_avc import CLIP_AVC, CLIP_AVC_Config
from models.video_swin import SUPPORTED_VIDEO_MODELS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-frames", type=int, default=8)
    parser.add_argument("--swin-frames", type=int, default=16)
    parser.add_argument("--clip-model", type=str, default="ViT-B/32")
    parser.add_argument("--text-encoder", choices=["clip", "bert"], default="clip")
    parser.add_argument("--max-text-tokens", type=int, default=77)
    parser.add_argument("--refined-text-pooling", choices=["eos", "mean"], default="eos")
    parser.add_argument("--freeze-clip-vit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-video-swin-backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-model", choices=SUPPORTED_VIDEO_MODELS, default="swin3d_b")
    parser.add_argument("--no-swin-pretrained", action="store_true",
                        help="Skip Kinetics-400 weight download (random init).")
    args = parser.parse_args()

    cfg = CLIP_AVC_Config(
        clip_frames=args.clip_frames,
        swin_frames=args.swin_frames,
        clip_model=args.clip_model,
        swin_weights=None if args.no_swin_pretrained else "KINETICS400_V1",
        video_model=args.video_model,
        text_encoder=args.text_encoder,
        max_text_tokens=args.max_text_tokens,
        refined_text_pooling=args.refined_text_pooling,
        freeze_clip_vit=args.freeze_clip_vit,
        freeze_video_swin_backbone=args.freeze_video_swin_backbone,
    )
    device = torch.device("cuda")
    model = CLIP_AVC(cfg).to(device)
    model.eval()

    b, s = args.batch, args.image_size
    frames = torch.randn(b, cfg.clip_frames, 3, s, s, device=device)
    swin_video = torch.randn(b, 3, cfg.swin_frames, s, s, device=device)
    prompts = (["a photo of harvesting", "a photo of swimming"] * b)[:b]
    toks = model.bert.tokenize(prompts, max_length=cfg.max_text_tokens)
    toks = {k: v.to(device) for k, v in toks.items()}

    print("--- inputs ---")
    print(f"clip frames : {tuple(frames.shape)}  (B, T_clip, 3, H, W)")
    print(f"swin video  : {tuple(swin_video.shape)}  (B, 3, T_swin, H, W)")
    print(f"input_ids   : {tuple(toks['input_ids'].shape)}")

    with torch.no_grad():
        u = model.clip_vit(frames)
        u_tilde = model.temporal(u)
        w_grid = model.video_swin(swin_video)
        v = model.cross(u_tilde, w_grid)
        text = model.encode_text(toks["input_ids"], toks["attention_mask"], toks.get("token_type_ids"))
        v_seq, s_seq = model.context(v, text.tokens, text_attention_mask=text.attention_mask)
        out = model(
            frames=frames,
            clip_video=swin_video,
            input_ids=toks["input_ids"],
            attention_mask=toks["attention_mask"],
            token_type_ids=toks.get("token_type_ids"),
        )
        losses = model.compute_losses(out, lam=1.0)

    print("--- intermediate ---")
    print(f"U (CLIP per-frame)        : {tuple(u.shape)}")
    print(f"U_tilde (Temporal x{cfg.temporal_layers})    : {tuple(u_tilde.shape)}")
    print(f"W ({cfg.video_model} tokens)      : {tuple(w_grid.shape)}")
    print(f"V (Cross x{cfg.cross_layers})            : {tuple(v.shape)}")
    print(f"S ({cfg.text_encoder} tokens)           : {tuple(text.tokens.shape)}")
    print(f"Context-Enriched visual   : {tuple(v_seq.shape)}")
    print(f"Context-Enriched text     : {tuple(s_seq.shape)}")
    print("--- outputs ---")
    print(f"v_bar : {tuple(out.v_bar.shape)}")
    print(f"s_eos : {tuple(out.s_eos.shape)}")
    print(f"v_hat : {tuple(out.v_hat.shape)}")
    print(f"s_hat : {tuple(out.s_hat.shape)}")
    print(f"logit_scale : {out.logit_scale.item():.4f}")
    print("--- losses ---")
    for k, val in losses.items():
        print(f"{k:>6s} : {val.item():.4f}")

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("--- params ---")
    print(f"total       : {n_total/1e6:7.2f} M")
    print(f"trainable   : {n_trainable/1e6:7.2f} M")
    print(f"frozen      : {(n_total-n_trainable)/1e6:7.2f} M")


if __name__ == "__main__":
    main()
