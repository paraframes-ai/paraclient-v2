#!/usr/bin/env bash
# ParaClient-v4: serve Gemma-4 12B (multimodal) with the 6 QLoRA adaptersG4 via
# vLLM 0.26 in the ISOLATED venv-g4. In-flight bitsandbytes 4-bit quant so the
# 12B + adapters + KV cache fit the L4's 24 GB.
#
# Keeps the SAME --served-model-name and --lora-module names as the Qwen/7B
# configs so the gateway (auth_gateway.py) needs NO changes. The base
# (ParaClient-v2.2) is multimodal -> powers /v1/notes + image upload; the text
# adapters power the tutoring/subject routes. Loads VLLM_API_KEY from .env.dev.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env.dev ]; then
  echo "ERROR: .env.dev not found (holds VLLM_API_KEY)." >&2
  exit 1
fi
# shellcheck disable=SC1091
set -a; source .env.dev; set +a
: "${VLLM_API_KEY:?VLLM_API_KEY not set in .env.dev}"

source venv-g4/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0
# bitsandbytes in-flight dequant needs transient scratch that vLLM's profiler
# doesn't count, so leave GPU headroom (0.82) and skip CUDA-graph capture
# (--enforce-eager) which OOM'd on the 24GB L4. expandable_segments cuts frag.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve google/gemma-4-12B-it \
  --host 127.0.0.1 \
  --served-model-name ParaFrames/ParaClient-v2.2 \
  --quantization bitsandbytes \
  --enforce-eager \
  --enable-lora \
  --max-loras 6 \
  --max-lora-rank 16 \
  --lora-modules ParaFrames/ParaClient-math-v2.2=adaptersG4/math \
                 ParaFrames/ParaClient-language-v2.2=adaptersG4/language_arts \
                 ParaFrames/ParaClient-cad-v2.2=adaptersG4/cad \
                 ParaFrames/ParaClient-civics-v2.2=adaptersG4/civics \
                 ParaFrames/ParaClient-spreadsheet-v2.2=adaptersG4/spreadsheet \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.82
