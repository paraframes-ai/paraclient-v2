#!/bin/bash
# One-time: pull Qwen2.5-14B-Instruct into the HF cache on scratch. CPU/network
# only (no GPU), on the general CPU partition — NOT the login node.
#SBATCH --job-name=dl14b
#SBATCH --account=mit_general
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export WORK=/orcd/pool/005/avb
module purge
module load miniforge/25.11.0-0
source "$WORK/envs/paraclient/bin/activate"
export HF_HOME="$WORK/hf"

python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-14B-Instruct')"
echo "[✓] Qwen2.5-14B-Instruct cached under $HF_HOME"
