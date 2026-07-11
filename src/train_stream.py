"""Sequential continual learning on TRACE-8 (one accumulating LoRA
adapter, low-budget protocol).  Methods:

  seqft : naive sequential fine-tuning
  er    : uniform experience replay from the buffer (50/50)
  tsr   : Transfer-Selective Replay (Algorithm 2 of the paper) --
          probe the task at the fixed shared initialisation, route
          replay by softmax over signature cosines, and distil each
          batch against its routed record's era snapshot.

Usage:
    python src/train_stream.py --config configs/trace_0.5b.yaml \
        --method tsr --seed 0 --out_json results/tsr_s0.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import infinite_loader, subsample
from distillation import kl_distill_loss
from lora import (attach_lora, lora_layers, reset_all, restore, snapshot,
                  trainable_params)
from metrics import eval_exact_match, stream_metrics
from model import load_base
from routing import cosine_scores, probe_signature, routing_weights
from tasks_trace import make_trace_tasks

CLS_NAMES = {"cstance", "fomc", "scienceqa", "numglue_cm", "numglue_ds"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["seqft", "er", "tsr"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_base(cfg["model_path"])
    attach_lora(model, r=cfg["rank"], alpha=cfg["alpha"])
    model = model.to(device)
    layers = list(lora_layers(model))
    reset_all(model)
    init_snap = snapshot(layers)  # fixed shared initialisation for probing

    tasks = make_trace_tasks(num_per_task=500, num_eval_per_task=500,
                             seed=args.seed, n_tasks=8, task_order=1)
    for t in tasks:
        t.train_small = subsample(t.train_ds, cfg["n_target"], args.seed)
    cls_ids = [i for i, t in enumerate(tasks) if t.name in CLS_NAMES]

    def loader_for(ds, seed):
        return infinite_loader(ds, tok, cfg["batch_size"],
                               cfg["max_length"], seed)

    def teacher_logits(batch, snap):
        cur = snapshot(layers)
        with torch.no_grad():
            restore(layers, snap)
            tl = model(**batch).logits.detach()
            restore(layers, cur)
        return tl

    buffer, sigs, snaps = {}, {}, {}
    eval_matrix, just_after = {}, {}

    for t_idx, task in enumerate(tasks):
        # ---- TSR: probe and route before training ----
        sel_j, keys, probs = None, [], []
        if args.method == "tsr":
            g = probe_signature(model, layers, init_snap,
                                loader_for(task.train_small, args.seed + 977),
                                cfg["probe_batches"], device)
            if sigs:
                scores = cosine_scores(sigs, g)
                sel_j = max(scores, key=scores.get)
                keys, probs = routing_weights(scores, cfg["sig_tau"])
                print(f"[route] task {t_idx} ({task.name}) "
                      f"top source = {sel_j} ({tasks[sel_j].name})", flush=True)
            sigs[t_idx] = g

        opt = torch.optim.AdamW(list(trainable_params(model)), lr=cfg["lr"])
        ld_target = loader_for(task.train_small, args.seed + t_idx)
        replay_loaders = []
        if args.method == "er" and buffer:
            replay_loaders = [loader_for(ds, args.seed + 31 + j)
                              for j, ds in buffer.items()]
        elif args.method == "tsr" and keys:
            replay_loaders = [loader_for(buffer[k], args.seed + 31 + k)
                              for k in keys]
        rng = random.Random(args.seed * 7919 + t_idx)
        use_kd = args.method == "tsr" and t_idx > 0

        model.train()
        for _ in range(cfg["steps"]):
            is_replay = bool(replay_loaders) and rng.random() < cfg["replay_frac"]
            replay_task = None
            if is_replay and args.method == "tsr":
                ri = rng.choices(range(len(replay_loaders)), weights=probs)[0]
                replay_task = keys[ri]
                ld = replay_loaders[ri]
            elif is_replay:
                ld = replay_loaders[rng.randrange(len(replay_loaders))]
            else:
                ld = ld_target
            b = {k: v.to(device) for k, v in next(ld).items()}
            out = model(**b)
            loss = out.loss
            if use_kd:
                # Route the triple: a replay batch from task j is distilled
                # against task j's era snapshot; target batches anchor to
                # the top-routed snapshot.
                snap = snaps.get(replay_task if replay_task is not None
                                 else sel_j)
                if snap is not None:
                    tl = teacher_logits(b, snap)
                    loss = loss + cfg["kd_lambda"] * kl_distill_loss(
                        out.logits, tl, temperature=cfg["kd_T"],
                        attention_mask=b.get("attention_mask"))
            loss.backward()
            opt.step()
            opt.zero_grad()

        # ---- end of task: write the record, evaluate seen tasks ----
        buffer[t_idx] = subsample(task.train_ds, cfg["buffer_per_task"],
                                  args.seed + 53)
        snaps[t_idx] = snapshot(layers)
        for s_idx in range(t_idx + 1):
            acc, lo = eval_exact_match(model, tasks[s_idx].eval_ds, tok,
                                       cfg["batch_size"], cfg["max_length"],
                                       device)
            eval_matrix[f"{t_idx}_{s_idx}"] = {"acc": acc, "loss": lo}
            if s_idx == t_idx:
                just_after[t_idx] = acc
            print(f"[eval] after={t_idx} on={s_idx} ({tasks[s_idx].name}) "
                  f"acc={acc:.4f} loss={lo:.3f}", flush=True)

    m = stream_metrics(eval_matrix, just_after, cls_ids, len(tasks))
    result = {"method": args.method, "seed": args.seed,
              "eval_matrix": eval_matrix, **m}
    print(f"[RESULT] method={args.method} seed={args.seed} "
          f"overall={m['overall']:.4f} plas={m['plas']:.4f} "
          f"bwt={m['bwt']:+.4f}", flush=True)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
