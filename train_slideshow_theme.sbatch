#!/bin/bash
# Train the slideshow theme selector on one preemptible H200.
#SBATCH --job-name=slideshow-theme
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/slideshow-theme-%j.out
#SBATCH --error=logs/slideshow-theme-%j.err

set -euo pipefail
export WORK=/orcd/pool/005/avb
module purge
module load miniforge/25.11.0-0 cuda/12.9.1
cd "$WORK/paraclient-v2"
mkdir -p logs
nvidia-smi

export HF_HOME="/tmp/avb-slideshow-theme-$SLURM_JOB_ID"
mkdir -p "$HF_HOME"
trap 'rm -rf "$HF_HOME"' EXIT

srun python train_adapter.py \
  --subject slideshow_theme \
  --data data/slideshow_theme.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct \
  --out-dir adapters \
  --epochs 3 --rank 16 --alpha 32 \
  --max-seq-len 512 --batch-size 8 --grad-accum 4 \
  --no-flash-attn --save-steps 100

srun python eval_slideshow_theme.py \
  --data data/slideshow_theme_holdout.jsonl \
  --base Qwen/Qwen2.5-7B-Instruct \
  --adapter adapters/slideshow_theme \
  --min-accuracy 0.95
