# Transfer-Selective Replay (TSR)

Official implementation of
*Rethinking Transfer in Continual Learning: A Replay-Based Realisation*.

One LoRA adapter trains sequentially through the eight TRACE tasks under
the paper's low-budget protocol (50 examples per task, 500 steps). Three
methods form the mechanism chain of the paper:

| method   | replay                              | stability                        |
|----------|-------------------------------------|----------------------------------|
| `seqft`  | none                                | none                             |
| `er`     | uniform over all past tasks         | none                             |
| `tsr`    | routed by task-signature similarity | distillation to era snapshots    |

TSR (Algorithm 2 of the paper): when a task arrives, a few probe batches
at the fixed shared initialisation produce its gradient signature; a
softmax over signature cosines routes every replay batch to the past
tasks most aligned with the new one; and each batch is distilled against
its routed record's end-of-task ("era") snapshot — replay batches from
task *j* against snapshot *j*, current-task batches against the
top-routed snapshot.

## Setup

```bash
pip install -r requirements.txt
```

A single GPU with ~16 GB memory suffices for the 0.5B configuration.

## Data

All eight TRACE task sources load directly from the Hugging Face Hub on
first run (see `src/tasks_trace.py` for the exact sources and prompt
formats); no manual download is required.

## Run

```bash
bash scripts/run_baseline.sh 0        # SeqFT, seed 0
bash scripts/run_er.sh 0              # ER, seed 0
bash scripts/run_tsr.sh 0             # TSR, seed 0
```

Each script writes `results/<method>_s<seed>.json`. The paper reports
5 seeds (0-4). For the 7B configuration pass the config explicitly:

```bash
bash scripts/run_tsr.sh 0 configs/trace_7b.yaml
```

## Aggregate

```bash
python eval/aggregate_results.py results/*.json
```

prints Overall / Plasticity / BWT (mean ± std over seeds) per method,
computed over the five classification tasks as in the paper.

## Expected output

TRACE-8 at Qwen2.5-0.5B-Instruct, 5 seeds (paper Table 3):

| method | Overall | Plas | BWT |
|--------|---------|------|-----|
| SeqFT  | ~0.26   | ~0.40 | negative |
| ER     | ~0.45   | ~0.43 | ~+0.03 |
| TSR    | ~0.48   | ~0.44 | ~+0.04 |

Results vary slightly across hardware, library versions, and seeds.

## Repository scope

This repository provides the reference implementation of TSR and the
scripts for the TRACE-8 experiments (Table 3 of the paper). Auxiliary
benchmarks and diagnostic analyses reported in the paper are not part
of this release.
