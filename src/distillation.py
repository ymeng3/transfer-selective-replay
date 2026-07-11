"""Token-level KL distillation toward a frozen reference model."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_distill_loss(student_logits, teacher_logits, *, temperature: float = 2.0,
                    attention_mask=None):
    """Mask-weighted token-wise KL(teacher || student) at temperature T,
    scaled by T^2 (standard knowledge distillation)."""
    T = temperature
    student_log = F.log_softmax(student_logits / T, dim=-1)
    teacher_p = F.softmax(teacher_logits / T, dim=-1)
    kl = (teacher_p * (torch.log(teacher_p.clamp(min=1e-12)) - student_log)).sum(-1)
    if attention_mask is not None:
        m = attention_mask.to(kl.dtype)
        kl = (kl * m).sum() / m.sum().clamp(min=1.0)
    else:
        kl = kl.mean()
    return kl * (T * T)
