#!/bin/bash
# Evaluate the theme selector on the challenge set.
#SBATCH --job-name=theme-ood-eval
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=logs/theme-ood-eval-%j.out
#SBATCH --error=logs/theme-ood-eval-%j.err

set -euo pipefail
export WORK=/orcd/pool/005/avb
module purge
module load miniforge/25.11.0-0 cuda/12.9.1
cd "$WORK/paraclient-v2"
mkdir -p logs

export HF_HOME="/tmp/avb-theme-eval-$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

srun python eval_slideshow_theme.py \
  --data data/slideshow_theme_ood.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct \
  --adapter adapters/slideshow_theme \
  --min-accuracy 0.92
