#!/usr/bin/env bash
# Run later on tardy, from a separate x86_64 vLLM environment. Do not use muse-arm.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2,3
PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export HF_HOME=${HF_HOME:-/data/scratch/ashwin/hf}
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export MUSE_BASE=meta-models/Muse-Glimmer-30B
export ADAPTER_ROOT=${ADAPTER_ROOT:-"$PROJECT_DIR/adapters/muse"}
export INSPECTION_REPORT=${INSPECTION_REPORT:-"$PROJECT_DIR/artifacts/muse_inspection.json"}
[[ -f "$INSPECTION_REPORT" ]] || { echo 'Copy the verified inspection report alongside the adapters.' >&2; exit 1; }
MUSE_REVISION=$(python -c 'import json,os; print(json.load(open(os.environ["INSPECTION_REPORT"]))["revision"])')
export MUSE_REVISION
python - <<'PY'
import inspect
import json
import os
from pathlib import Path
import vllm
from transformers import AutoConfig
from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.interfaces import supports_lora
from vllm.reasoning import ReasoningParserManager
from vllm.tool_parsers import ToolParserManager
base = os.environ['MUSE_BASE']
revision = os.environ['MUSE_REVISION']
config = AutoConfig.from_pretrained(base, revision=revision, token=False)
if config.architectures != ['MuseGlimmerForConditionalGeneration']:
    raise SystemExit(f'Unexpected architecture: {config.architectures}')
try:
    kwargs = {}
    if 'model_config' in inspect.signature(ModelRegistry.resolve_model_cls).parameters:
        from vllm.config import ModelConfig
        kwargs['model_config'] = ModelConfig(model=base, revision=revision, tokenizer_revision=revision, dtype='bfloat16', max_model_len=4096, model_impl='vllm')
    cls, architecture = ModelRegistry.resolve_model_cls(config.architectures, **kwargs)
    if not supports_lora(cls):
        raise RuntimeError(f'{cls.__name__} does not implement SupportsLoRA')
    if not getattr(cls, 'hf_to_vllm_mapper', None):
        raise RuntimeError('Muse HF-to-vLLM language/gate weight mapping is missing')
    ToolParserManager.get_tool_parser('muse_glimmer')
    ReasoningParserManager.get_reasoning_parser('muse_glimmer')
except Exception as exc:
    raise SystemExit(f'Installed vLLM {vllm.__version__} cannot serve this Muse LoRA/parser combination: {exc}. Install a compatible x86_64 vLLM release before serving.') from exc
for subject in ('math', 'language_arts', 'civics', 'general'):
    folder = Path(os.environ['ADAPTER_ROOT']) / subject
    adapter = json.loads((folder / 'adapter_config.json').read_text())
    if adapter.get('base_model_name_or_path') != base or adapter.get('r', 999) > 16:
        raise SystemExit(f'Wrong base/rank in {folder}')
    targets = adapter.get('target_modules', [])
    if not isinstance(targets, list) or not targets or any(not item.startswith('model.language_model.layers.') for item in targets):
        raise SystemExit(f'Expected language-only full-path LoRA targets in {folder}')
    if not (folder / 'adapter_model.safetensors').is_file():
        raise SystemExit(f'Missing adapter weights in {folder}')
print(f'vLLM {vllm.__version__}: {architecture} supports LoRA; both Muse parsers are registered.')
PY
exec vllm serve "$MUSE_BASE" \
    --revision "$MUSE_REVISION" --tokenizer-revision "$MUSE_REVISION" \
    --dtype bfloat16 --tensor-parallel-size 2 \
    --enable-lora --max-loras 4 --max-cpu-loras 4 --max-lora-rank 16 \
    --lora-modules "math=$ADAPTER_ROOT/math" "language_arts=$ADAPTER_ROOT/language_arts" \
                   "civics=$ADAPTER_ROOT/civics" "general=$ADAPTER_ROOT/general" \
    --reasoning-parser muse_glimmer --enable-auto-tool-choice --tool-call-parser muse_glimmer \
    --max-model-len 4096 --max-num-seqs 2 --gpu-memory-utilization 0.90 --enforce-eager \
    --host 127.0.0.1 --port 8000
