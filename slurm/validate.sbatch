#!/usr/bin/env bash
# Pre-release validation: SeqFT / ER / TSR, seed 0, TRACE-8 at 0.5B.
# Target: TSR Overall within .475 +- .030 (paper Table 3).
#SBATCH --job-name=tsr_val
#SBATCH --partition=yuxinchen-contrib,general
#SBATCH --constraint=a40|h100|L40S
#SBATCH --exclude=q001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm/val_a%a_%j.out
#SBATCH --error=slurm/val_a%a_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-/net/scratch/ymeng3/hf_cache}"
export TMPDIR="/local/ymeng3/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
source /opt/conda/etc/profile.d/conda.sh; conda activate clrd
export PATH="$CONDA_PREFIX/bin:$PATH"; export PYTHONNOUSERSITE=1
METHODS=(seqft er tsr)
M=${METHODS[$SLURM_ARRAY_TASK_ID]}
python -u src/train_stream.py --config configs/trace_0.5b.yaml \
    --method "$M" --seed 0 --out_json "results/${M}_s0.json"
