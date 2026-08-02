#!/bin/bash
# QLoRA one subject adapter on Qwen2.5-14B-Instruct on a single H200.
# Usage:  sbatch -J pc14b-<subj> train_14b.sbatch <subject> <data.jsonl>
# 14B adapters are written under adapters14b/ (kept separate from the 7B ones).
#SBATCH --job-name=pc14b
#SBATCH --account=mit_general
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
SUBJ=${1:?usage: train_14b.sbatch <subject> <data.jsonl>}
DATA=${2:?usage: train_14b.sbatch <subject> <data.jsonl>}

export WORK=/orcd/pool/005/avb
module purge
module load miniforge/25.11.0-0 cuda/12.9.1
source "$WORK/envs/paraclient/bin/activate"
export HF_HOME="$WORK/hf"

cd "$WORK/paraclient-v2"
nvidia-smi
srun python train_adapter.py \
  --subject "$SUBJ" \
  --data "$DATA" \
  --base Qwen/Qwen2.5-14B-Instruct \
  --out-dir adapters14b \
  --no-flash-attn
