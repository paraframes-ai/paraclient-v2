#!/usr/bin/env bash
# Source from compute jobs; all persistent state belongs on NFS, never AFS.
export NFS_ROOT=${NFS_ROOT:-/data/scratch/ashwin}
export HF_HOME="$NFS_ROOT/hf"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
export XDG_CACHE_HOME="$NFS_ROOT/cache"
export TORCH_HOME="$NFS_ROOT/cache/torch"
export TRITON_CACHE_DIR="$NFS_ROOT/cache/triton"
export TMPDIR="$NFS_ROOT/tmp/${SLURM_JOB_ID:-manual}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$TMPDIR" "$HF_HOME" "$TRITON_CACHE_DIR"
source "$NFS_ROOT/envs/muse-arm/bin/activate"
cd "$NFS_ROOT/paraclient-muse"
