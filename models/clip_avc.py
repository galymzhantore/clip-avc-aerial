
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import math
import torch
import torch.nn.functional as F
from torch import nn

from losses.info_nce import bidirectional_info_nce
from models.clip_encoder import CLIPViTEncoder
from models.context_enriched import ContextEnrichedTransformer
from models.cross_transformer import CrossTransformer
from models.temporal_transformer import TemporalTransformer
from models.text_encoder import BERTTextEncoder, CLIPTextEncoder
from models.video_swin import SUPPORTED_VIDEO_MODELS, VideoSwinEncoder


@dataclass
class CLIP_AVC_Config:
    embed_dim: int = 512
    n_heads: int = 8
    temporal_layers: int = 3
    cross_layers: int = 2
    context_layers: int = 2
    dropout: float = 0.0
    clip_model: str = "ViT-B/32"
    bert_model: str = "bert-base-uncased"
    text_encoder: str = "clip"
    refined_text_pooling: str = "eos"
    freeze_clip_vit: bool = True
    freeze_video_swin_backbone: bool = False
    video_model: str = "swin3d_b"
    swin_weights: str | None = "KINETICS400_V1"  # set to None to skip pretrained download
    clip_frames: int = 8
    swin_frames: int = 16
    n_frames: int | None = field(default=None, repr=False)
    max_text_tokens: int = 77
    init_logit_scale: float = math.log(1.0 / 0.07)  # CLIP default
    use_cross_transformer: bool = True
    use_context_transformer: bool = True
    checkpoint_video_swin: bool = False

    def __post_init__(self):
        # Backward compatibility for old checkpoints/config dicts saved with n_frames.
        if self.n_frames is not None:
            self.clip_frames = self.n_frames
            self.swin_frames = self.n_frames
        if self.refined_text_pooling not in {"eos", "mean"}:
            raise ValueError(f"Unsupported refined_text_pooling={self.refined_text_pooling!r}")
        if self.video_model not in SUPPORTED_VIDEO_MODELS:
            raise ValueError(
                f"Unsupported video_model={self.video_model!r}; "
                f"choose one of {SUPPORTED_VIDEO_MODELS!r}."
            )


class CLIP_AVC_Outputs(NamedTuple):
    v_seq: torch.Tensor        # (B, T, D) Cross-Transformer output sequence
    v_bar: torch.Tensor        # (B, D) coarse visual (mean over Cross-Transformer output)
    s_eos: torch.Tensor        # (B, D) coarse text   (BERT [SEP]/EOS projection)
    v_hat: torch.Tensor        # (B, D) refined visual (mean over Context-Enriched output)
    s_hat: torch.Tensor        # (B, D) refined text  (mean over Context-Enriched output)
    logit_scale: torch.Tensor  # scalar


class CLIP_AVC(nn.Module):
    def __init__(self, config: CLIP_AVC_Config | None = None):
        super().__init__()
        self.config = config or CLIP_AVC_Config()
        cfg = self.config

        self.clip_vit = CLIPViTEncoder(model_name=cfg.clip_model, trainable=not cfg.freeze_clip_vit)
        if self.clip_vit.embed_dim != cfg.embed_dim:
            raise ValueError(
                f"{cfg.clip_model} visual output dim is {self.clip_vit.embed_dim}, "
                f"but CLIP_AVC_Config.embed_dim={cfg.embed_dim}. "
                "Use a CLIP checkpoint with the same joint dimension or extend the projection path."
            )
        if cfg.text_encoder == "clip":
            self.bert = CLIPTextEncoder(model_name=cfg.clip_model, embed_dim=cfg.embed_dim)
        elif cfg.text_encoder == "bert":
            self.bert = BERTTextEncoder(model_name=cfg.bert_model, embed_dim=cfg.embed_dim)
        else:
            raise ValueError(f"Unsupported text_encoder={cfg.text_encoder!r}")
        self.video_swin = VideoSwinEncoder(
            embed_dim=cfg.embed_dim,
            model_name=cfg.video_model,
            pretrained=cfg.swin_weights is not None,
            weights=cfg.swin_weights or "KINETICS400_V1",
            gradient_checkpointing=cfg.checkpoint_video_swin,
            freeze_backbone=cfg.freeze_video_swin_backbone,
        )

        self.temporal = TemporalTransformer(
            d_model=cfg.embed_dim,
            n_layers=cfg.temporal_layers,
            n_heads=cfg.n_heads,
            max_frames=max(cfg.clip_frames, 32),
            dropout=cfg.dropout,
        )
        self.cross = CrossTransformer(
            d_model=cfg.embed_dim,
            n_layers=cfg.cross_layers,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.context = ContextEnrichedTransformer(
            d_model=cfg.embed_dim,
            n_layers=cfg.context_layers,
            n_heads=cfg.n_heads,
            max_tokens=cfg.clip_frames + cfg.max_text_tokens,
            dropout=cfg.dropout,
        )
        self.logit_scale = nn.Parameter(torch.tensor(cfg.init_logit_scale))

    # ------------------------------------------------------------------ helpers
    def encode_visual(self, frames: torch.Tensor, clip_video: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run both visual paths and return:
          V    : (B, T, D) Cross-Transformer output  (used for coarse mean-pool)
          V    : same tensor; reused for the Context-Enriched stage
        """
        u = self.clip_vit(frames)        # (B, T, D)
        u_tilde = self.temporal(u)       # (B, T, D)
        if self.config.use_cross_transformer:
            w = self.video_swin(clip_video)  # (B, T'*H'*W', D)
            v = self.cross(u_tilde, w)       # (B, T, D)
        else:
            v = u_tilde
        return v, u_tilde

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        frames: torch.Tensor,        # (B, T, 3, H, W) CLIP-normalised
        clip_video: torch.Tensor,    # (B, 3, T_f, H, W) Swin-normalised
        input_ids: torch.Tensor,     # (B, K)
        attention_mask: torch.Tensor,  # (B, K)
        token_type_ids: torch.Tensor | None = None,
    ) -> CLIP_AVC_Outputs:
        v, _ = self.encode_visual(frames, clip_video)
        text = self.encode_text(input_ids, attention_mask, token_type_ids)

        v_bar = v.mean(dim=1)        # coarse visual
        s_eos = text.eos             # coarse text ([SEP]/EOS)

        if self.config.use_context_transformer:
            v_seq, s_seq = self.context(v, text.tokens, text_attention_mask=text.attention_mask)
        else:
            v_seq, s_seq = v, text.tokens
        v_hat = v_seq.mean(dim=1)    # refined visual
        s_hat = self._pool_text(s_seq, text.attention_mask, self.config.refined_text_pooling)

        return CLIP_AVC_Outputs(
            v_seq=v,
            v_bar=v_bar,
            s_eos=s_eos,
            v_hat=v_hat,
            s_hat=s_hat,
            logit_scale=self.logit_scale.exp(),
        )

    @staticmethod
    def _pool_text(tokens: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mean":
            mask = attention_mask.unsqueeze(-1).type_as(tokens)
            return (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if mode == "eos":
            last_idx = attention_mask.sum(dim=1).long().clamp_min(1) - 1
            batch_idx = torch.arange(tokens.size(0), device=tokens.device)
            return tokens[batch_idx, last_idx]
        raise ValueError(f"Unsupported refined_text_pooling={mode!r}")

    def context_similarity_logits(
        self,
        v: torch.Tensor,
        text_tokens: torch.Tensor,
        text_attention_mask: torch.Tensor,
        chunk_size: int = 0,
    ) -> torch.Tensor:
        """Score every video against every class prompt through C-Trans.

        This matches the target-conditioned nature of the Context-Enriched
        Transformer: each candidate class text is paired with the visual token
        sequence before producing the class logit.
        """
        b, t, d = v.shape
        n_classes, k = text_tokens.shape[:2]
        flat_total = b * n_classes
        chunk_size = chunk_size or flat_total
        logits: list[torch.Tensor] = []
        scale = self.logit_scale.exp().clamp(max=100.0)

        for start in range(0, flat_total, chunk_size):
            end = min(flat_total, start + chunk_size)
            pair_idx = torch.arange(start, end, device=v.device)
            video_idx = torch.div(pair_idx, n_classes, rounding_mode="floor")
            class_idx = pair_idx.remainder(n_classes)
            v_flat = v[video_idx]
            s_flat = text_tokens[class_idx]
            mask_flat = text_attention_mask[class_idx]

            if self.config.use_context_transformer:
                v_seq, s_seq = self.context(v_flat, s_flat, text_attention_mask=mask_flat)
            else:
                v_seq, s_seq = v_flat, s_flat
            v_hat = F.normalize(v_seq.mean(dim=1), dim=-1)
            s_hat = self._pool_text(
                s_seq,
                mask_flat,
                self.config.refined_text_pooling,
            )
            s_hat = F.normalize(s_hat, dim=-1)
            logits.append(scale * (v_hat * s_hat).sum(dim=-1))

        return torch.cat(logits, dim=0).reshape(b, n_classes)

    # ------------------------------------------------------------------ losses
    def compute_losses(
        self,
        out: CLIP_AVC_Outputs,
        lam: float = 1.0,
        use_lc: bool = True,
        use_lr: bool = True,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        scale = out.logit_scale.clamp(max=100.0)
        zero = out.v_bar.new_zeros(())
        l_c = bidirectional_info_nce(out.v_bar, out.s_eos, scale, labels=labels) if use_lc else zero
        l_r = bidirectional_info_nce(out.v_hat, out.s_hat, scale, labels=labels) if use_lr else zero
        return {"loss": l_r + lam * l_c, "L_r": l_r, "L_c": l_c}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def encode_text_only_refined(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build the refined text embedding for a prompt without a paired video.

        Uses zeros as the visual stream — see §11 note: the Context-Enriched
        Transformer is jointly visual+textual, but text positions dominate their
        own outputs, so class prototypes are cached this way in practice.
        """
        text = self.encode_text(input_ids, attention_mask, token_type_ids)
        b = input_ids.size(0)
        zero_v = torch.zeros(b, self.config.clip_frames, self.config.embed_dim, device=input_ids.device)
        _, s_seq = self.context(zero_v, text.tokens, text_attention_mask=text.attention_mask)
        return self._pool_text(s_seq, text.attention_mask, self.config.refined_text_pooling)

    @torch.no_grad()
    def encode_visual_only_refined(self, v: torch.Tensor) -> torch.Tensor:
        """Build a refined visual embedding without pairing it to a class prompt."""
        if not self.config.use_context_transformer:
            return v.mean(dim=1)
        b = v.size(0)
        zero_s = torch.zeros(
            b,
            self.config.max_text_tokens,
            self.config.embed_dim,
            dtype=v.dtype,
            device=v.device,
        )
        zero_mask = torch.zeros(
            b,
            self.config.max_text_tokens,
            dtype=torch.long,
            device=v.device,
        )
        v_seq, _ = self.context(v, zero_s, text_attention_mask=zero_mask)
        return v_seq.mean(dim=1)

    def trainable_parameters(self):
        """Returns parameters that require gradients (skipping frozen CLIP/text towers)."""
        return [p for p in self.parameters() if p.requires_grad]
