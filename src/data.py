"""Task container, label-masked collation, and sampling helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class Task:
    id: int
    name: str
    train_ds: object
    eval_ds: object
    train_small: object = None


def make_collate_fn(tokenizer, max_length: int = 384):
    """Collate (prompt, answer) pairs with loss on answer tokens only.
    If prompt + answer exceeds max_length, the prompt is truncated from
    the left so the answer always survives intact."""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    eos = tokenizer.eos_token or ""

    def _collate(batch):
        ids_per, lbl_per, attn_per = [], [], []
        for ex in batch:
            p_ids = tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(ex["answer"] + eos, add_special_tokens=False)["input_ids"]
            total = len(p_ids) + len(a_ids)
            if total > max_length:
                drop = total - max_length
                if drop >= len(p_ids):
                    a_ids = a_ids[:max_length]
                    p_ids = []
                else:
                    p_ids = p_ids[drop:]
            ids = p_ids + a_ids
            ids_per.append(ids)
            lbl_per.append([-100] * len(p_ids) + list(a_ids))
            attn_per.append([1] * len(ids))
        m = max(len(x) for x in ids_per)
        return {
            "input_ids": torch.tensor(
                [x + [pad_id] * (m - len(x)) for x in ids_per], dtype=torch.long),
            "attention_mask": torch.tensor(
                [x + [0] * (m - len(x)) for x in attn_per], dtype=torch.long),
            "labels": torch.tensor(
                [x + [-100] * (m - len(x)) for x in lbl_per], dtype=torch.long),
        }

    return _collate


def subsample(ds, n: int, seed: int):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return torch.utils.data.Subset(ds, idx[: min(n, len(ds))])


def infinite_loader(ds, tok, batch_size: int, max_length: int, seed: int):
    collate = make_collate_fn(tok, max_length=max_length)

    def gen():
        g = torch.Generator()
        g.manual_seed(seed)
        while True:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                collate_fn=collate, generator=g)
            for b in loader:
                yield b

    return gen()
