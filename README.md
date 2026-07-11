# Transfer-Selective Replay (TSR)

Official implementation of
**Rethinking Transfer in Continual Learning: A Replay-Based Realisation**.

This repository reproduces the TRACE-8 continual-learning experiments
reported in the paper: one LoRA adapter trained sequentially through
the eight TRACE tasks under the standard protocol at a low target
budget. TSR routes replay by task-signature similarity and distils
each batch against an era snapshot; details are given in Algorithm 2
of the paper.

## Installation

```bash
pip install -r requirements.txt
```

A single GPU with ~16 GB memory suffices for the 0.5B configuration.

## Data

TRACE-8 is downloaded automatically from the Hugging Face Hub on first
run. No manual preprocessing is required.

## Running experiments

```bash
bash scripts/run_seqft.sh 0
bash scripts/run_er.sh 0
bash scripts/run_tsr.sh 0
```

Each command runs one seed (the paper reports seeds 0-4) and writes
`results/<method>_s<seed>.json`. For the 7B configuration pass the
config explicitly:

```bash
bash scripts/run_tsr.sh 0 configs/trace_7b.yaml
```

## Results

```bash
python eval/aggregate_results.py results/*.json
```

reports Overall / Plasticity / BWT (mean ± std over seeds) following
the evaluation protocol of the paper. Aggregated values should closely
match Table 3 of the paper.

## Citation

```bibtex
@article{meng2026rethinking,
  title   = {Rethinking Transfer in Continual Learning:
             A Replay-Based Realisation},
  author  = {Meng, Yang and Liu, Zhenya and Zhao, Zhuokai
             and Chen, Yuxin},
  journal = {arXiv preprint},
  year    = {2026}
}
```
