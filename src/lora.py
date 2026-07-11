"""Minimal LoRA layer: a frozen base linear plus one trainable low-rank
pair (A, B).  forward(x) = base(x) + scaling * x @ A^T @ B^T."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        dtype = base_linear.weight.dtype
        device = base_linear.weight.device
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(
            torch.empty(r, base_linear.in_features, dtype=dtype, device=device))
        self.B = nn.Parameter(
            torch.empty(base_linear.out_features, r, dtype=dtype, device=device))
        self.reset()

    def reset(self) -> None:
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling


def attach_lora(model: nn.Module, targets=("q_proj", "k_proj", "v_proj", "o_proj"),
                r: int = 16, alpha: int = 32) -> int:
    """Replace every matching nn.Linear with a LoRALinear wrapper."""
    n = 0
    for name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(parent, child_name, LoRALinear(child, r=r, alpha=alpha))
                n += 1
    return n


def lora_layers(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            yield m


def reset_all(model: nn.Module) -> None:
    for l in lora_layers(model):
        l.reset()


def trainable_params(model: nn.Module):
    for l in lora_layers(model):
        yield l.A
        yield l.B


def snapshot(layers) -> list:
    """Detached copy of every layer's (A, B)."""
    return [(l.A.detach().clone(), l.B.detach().clone()) for l in layers]


@torch.no_grad()
def restore(layers, snap) -> None:
    for l, (a, b) in zip(layers, snap):
        l.A.data.copy_(a)
        l.B.data.copy_(b)
