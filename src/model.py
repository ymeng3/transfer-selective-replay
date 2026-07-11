"""Base model loading."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_base(model_path: str, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else None)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok
