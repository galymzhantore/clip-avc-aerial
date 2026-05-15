"""Prompt construction and batching."""
from __future__ import annotations

import torch
from transformers import BertTokenizer


def build_prompts(class_names: list[str], template: str = "a photo of {}") -> list[str]:
    return [template.format(c.replace("_", " ").lower()) for c in class_names]


def tokenize_prompts(
    tokenizer: BertTokenizer,
    prompts: list[str],
    max_length: int = 32,
) -> dict[str, torch.Tensor]:
    return tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
