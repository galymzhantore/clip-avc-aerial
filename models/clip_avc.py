
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import math
import torch
from torch import nn

from losses.info_nce import bidirectional_info_nce
from models.clip_encoder import CLIPViTEncoder
from models.context_enriched import ContextEnrichedTransformer
from models.cross_transformer import CrossTransformer
from models.temporal_transformer import TemporalTransformer
from models.text_encoder import BERTTextEncoder
from models.video_swin import VideoSwinEncoder


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
    swin_weights: str | None = "KINETICS400_V1"  # set to None to skip pretrained download
    clip_frames: int = 8
    swin_frames: int = 16
    n_frames: int | None = field(default=None, repr=False)
    max_text_tokens: int = 32
    init_logit_scale: float = math.log(1.0 / 0.07)  # CLIP default
    use_cross_transformer: bool = True
    use_context_transformer: bool = True
    checkpoint_video_swin: bool = False

    def __post_init__(self):
        # Backward compatibility for old checkpoints/config dicts saved with n_frames.
        if self.n_frames is not None:
            self.clip_frames = self.n_frames
            self.swin_frames = self.n_frames


class CLIP_AVC_Outputs(NamedTuple):
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

        self.clip_vit = CLIPViTEncoder(model_name=cfg.clip_model)
        self.bert = BERTTextEncoder(model_name=cfg.bert_model, embed_dim=cfg.embed_dim)
        self.video_swin = VideoSwinEncoder(
            embed_dim=cfg.embed_dim,
            pretrained=cfg.swin_weights is not None,
            weights=cfg.swin_weights or "KINETICS400_V1",
            gradient_checkpointing=cfg.checkpoint_video_swin,
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

        # Refined text: mean over real (non-pad) text tokens.
        mask = text.attention_mask.unsqueeze(-1).type_as(s_seq)  # (B, K, 1)
        s_hat = (s_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return CLIP_AVC_Outputs(
            v_bar=v_bar,
            s_eos=s_eos,
            v_hat=v_hat,
            s_hat=s_hat,
            logit_scale=self.logit_scale.exp(),
        )

    # ------------------------------------------------------------------ losses
    def compute_losses(
        self,
        out: CLIP_AVC_Outputs,
        lam: float = 1.0,
        use_lc: bool = True,
        use_lr: bool = True,
    ) -> dict[str, torch.Tensor]:
        scale = out.logit_scale.clamp(max=100.0)
        zero = out.v_bar.new_zeros(())
        l_c = bidirectional_info_nce(out.v_bar, out.s_eos, scale) if use_lc else zero
        l_r = bidirectional_info_nce(out.v_hat, out.s_hat, scale) if use_lr else zero
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
        mask = text.attention_mask.unsqueeze(-1).type_as(s_seq)
        return (s_seq * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def trainable_parameters(self):
        """Returns parameters that require gradients (skipping frozen CLIP/BERT)."""
        return [p for p in self.parameters() if p.requires_grad]
