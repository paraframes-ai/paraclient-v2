#!/bin/bash
#SBATCH --job-name=guard-distill
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/guard-distill-%j.out
# Distill the fast K-12 guard on one GPU (ORCD). Data + base model are staged to
# the HF cache on the login node first, so this runs offline. ~minutes on any card.
set -euo pipefail
cd /orcd/pool/005/avb/paraclient-v2
# envs/paraclient is a plain venv (no conda-meta) -> use its python directly.
PY=/orcd/pool/005/avb/envs/paraclient/bin/python
export HF_HOME=/orcd/pool/005/avb/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # use the pre-staged cache
mkdir -p logs models

echo "[job] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
$PY -m guard_distill.train_student \
  --data data/guard_labeled.jsonl \
  --base roberta-base \
  --out models/guard-student --epochs 6 --batch-size 32

echo "[job] === eval on the human-labeled holdout ==="
$PY -m guard_distill.eval_student \
  --data data/guard_holdout.jsonl \
  --model models/guard-student --target-recall 0.95
echo "[job] done."
