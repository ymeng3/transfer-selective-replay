"""Task signature and routing: probe the incoming task at the fixed shared
initialisation, compare against stored signatures, and produce routing
weights (Algorithm 2 of the paper)."""
from __future__ import annotations

import math

import torch

from lora import restore, snapshot


def probe_signature(model, layers, init_snap, loader, n_batches: int, device):
    """Mean probe-batch gradient of the LoRA B factors at the shared
    initialisation, flattened across layers.  The adapter state is
    swapped out and restored, so probing leaves training untouched."""
    cur = snapshot(layers)
    restore(layers, init_snap)
    model.train()
    grad_acc, n = None, 0
    for _ in range(n_batches):
        b = {k: v.to(device) for k, v in next(loader).items()}
        model(**b).loss.backward()
        gs = torch.cat([l.B.grad.detach().float().flatten().cpu()
                        for l in layers])
        grad_acc = gs if grad_acc is None else grad_acc + gs
        for l in layers:
            l.B.grad = None
            if l.A.grad is not None:
                l.A.grad = None
        n += 1
    restore(layers, cur)
    return grad_acc / n


def cosine_scores(signatures: dict, g: torch.Tensor) -> dict:
    return {k: torch.nn.functional.cosine_similarity(
        s.unsqueeze(0), g.unsqueeze(0)).item() for k, s in signatures.items()}


def routing_weights(scores: dict, tau: float):
    """softmax(cos / tau) over past tasks, keys in sorted order."""
    keys = sorted(scores)
    zs = [math.exp(scores[k] / tau) for k in keys]
    Z = sum(zs)
    return keys, [z / Z for z in zs]
