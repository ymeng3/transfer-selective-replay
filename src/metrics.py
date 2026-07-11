"""Evaluation and the three stream metrics (Overall / Plas / BWT)."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from data import make_collate_fn


@torch.no_grad()
def eval_exact_match(model, ds, tok, batch_size: int, max_length: int, device):
    """Teacher-forced exact match over the supervised (answer) tokens,
    plus mean evaluation loss.  Returns (acc, loss)."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(tok, max_length=max_length))
    n_correct, n_total, losses = 0, 0, []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
        logits = out.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        correct = (logits.argmax(-1) == labels) & mask
        n_tok_ok = correct.sum(dim=-1)
        n_tok = mask.sum(dim=-1)
        ex_ok = (n_tok_ok == n_tok) & (n_tok > 0)
        n_correct += ex_ok.sum().item()
        n_total += (n_tok > 0).sum().item()
    model.train()
    return n_correct / max(n_total, 1), sum(losses) / max(len(losses), 1)


def stream_metrics(eval_matrix: dict, just_after: dict, cls_ids: list, K: int):
    """Overall / Plas / BWT over the classification tasks, following the
    paper's Eqs. (2)-(4)."""
    final = {s: eval_matrix[f"{K - 1}_{s}"]["acc"] for s in range(K)}
    overall = sum(final[s] for s in cls_ids) / len(cls_ids)
    plas = sum(just_after[s] for s in cls_ids) / len(cls_ids)
    bwt_terms = [final[s] - just_after[s] for s in cls_ids if s < K - 1]
    bwt = sum(bwt_terms) / max(len(bwt_terms), 1)
    return {"overall": overall, "plas": plas, "bwt": bwt}
