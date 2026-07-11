#!/usr/bin/env bash
# Usage: bash scripts/run_tsr_v2.sh [seed] [config]
set -e
SEED="${1:-0}"
CONFIG="${2:-configs/trace_0.5b.yaml}"
python src/train_stream.py --config "$CONFIG" --method tsr_v2 \
    --seed "$SEED" --out_json "results/tsr_v2_s${SEED}.json"
