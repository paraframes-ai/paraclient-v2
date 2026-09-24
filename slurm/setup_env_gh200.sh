#!/usr/bin/env bash
# Run only in a GH200 Slurm allocation. No installs on the login node.
set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script through Slurm}"
[[ $(uname -m) == aarch64 ]] || { echo 'Expected aarch64 GH200'; exit 1; }
export NFS_ROOT=${NFS_ROOT:-/data/scratch/ashwin}
export HF_HOME="$NFS_ROOT/hf"
export PIP_CACHE_DIR="$NFS_ROOT/cache/pip"
export XDG_CACHE_HOME="$NFS_ROOT/cache"
export TORCH_HOME="$NFS_ROOT/cache/torch"
export TMPDIR="$NFS_ROOT/tmp/setup-$SLURM_JOB_ID"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_DISABLE_TELEMETRY=1
PROJECT="$NFS_ROOT/paraclient-muse"
VENV="$NFS_ROOT/envs/muse-arm"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$PROJECT/artifacts" "$NFS_ROOT/envs"
cd "$PROJECT"
trap 'echo "Environment setup failed at line $LINENO. Inspect this job log for the failing package. If ARM wheels/builds are unavailable, use an aarch64 NVIDIA NGC PyTorch container via Apptainer --nv; do not install on the login node." >&2' ERR
if [[ -e "$VENV" && ! -f "$VENV/.paraclient-muse-env" ]]; then
    echo "Refusing to modify an unrecognized existing environment: $VENV" >&2
    exit 1
fi
if [[ ! -d "$VENV" ]]; then
    mkdir -p "$VENV"
    touch "$VENV/.paraclient-muse-env"
    python3 -m venv --without-pip "$VENV"
fi
source "$VENV/bin/activate"
if ! python -m pip --version >/dev/null 2>&1; then
    python - <<'PYBOOT'
import os
from pathlib import Path
import urllib.request
path = Path(os.environ['TMPDIR']) / 'get-pip.py'
with urllib.request.urlopen('https://bootstrap.pypa.io/get-pip.py', timeout=60) as response:
    path.write_bytes(response.read())
PYBOOT
    python "$TMPDIR/get-pip.py"
fi
python -m pip install --upgrade pip setuptools wheel
python -m pip install --only-binary=:all: torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'transformers==5.17.0' 'peft==0.21.0' 'trl==1.13.0' 'accelerate==1.15.0' 'datasets==5.0.1' pillow sentencepiece openai fastapi uvicorn httpx jsonschema pytest
python -m pip check
python - <<'PY'
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, MuseGlimmerConfig
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
assert torch.version.cuda == '12.8', f'Expected cu128, got {torch.version.cuda}'
assert torch.cuda.is_bf16_supported(), 'bf16 is required'
assert MuseGlimmerConfig in AutoModelForImageTextToText._model_mapping
assert (torch.ones(2, device='cuda') + 1).sum().item() == 4
packages = {p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'transformers', 'peft', 'trl', 'accelerate', 'datasets']}
report = dict(job_id=os.environ['SLURM_JOB_ID'], host=platform.node(), machine=platform.machine(), cuda_available=torch.cuda.is_available(), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory, packages=packages)
Path('artifacts/environment.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
PY
python -m pip freeze > slurm/requirements-gh200.lock
printf 'Environment ready: %s\n' "$VENV"
