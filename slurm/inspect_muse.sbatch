#!/usr/bin/env bash
#SBATCH --job-name=muse-inspect
#SBATCH --account=quanta
#SBATCH --partition=quanta-gh200
#SBATCH --qos=quanta-main
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200000M
#SBATCH --time=00:30:00
#SBATCH --chdir=/data/scratch/ashwin/paraclient-muse
#SBATCH --output=/data/scratch/ashwin/logs/%x_%j.out
set -euo pipefail
source slurm/env.sh
srun python -u inspect_muse.py "$@"
CUDA_VISIBLE_DEVICES= python -m pytest -q tests --junitxml=artifacts/unit-tests.xml
