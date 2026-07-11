#!/usr/bin/env bash
#SBATCH --job-name=tsr_v1
#SBATCH --partition=yuxinchen-contrib,general
#SBATCH --constraint=a40|h100|L40S
#SBATCH --exclude=q001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=slurm/v1_a%a_%j.out
#SBATCH --error=slurm/v1_a%a_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="${HF_HOME:-/net/scratch/ymeng3/hf_cache}"
export TMPDIR="/local/ymeng3/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
source /opt/conda/etc/profile.d/conda.sh; conda activate clrd
export PATH="$CONDA_PREFIX/bin:$PATH"; export PYTHONNOUSERSITE=1
python -u src/train_stream.py --config configs/trace_0.5b.yaml \
    --method tsr_v1 --seed "$SLURM_ARRAY_TASK_ID" \
    --out_json "results/tsr_v1_s${SLURM_ARRAY_TASK_ID}.json"
