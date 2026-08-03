#!/usr/bin/env bash
# Launch the DEV vLLM on the 14B AWQ base with the 14B QLoRA adapters.
# Keeps the SAME --served-model-name and --lora-module names as the 7B config so
# the gateway (auth_gateway.py) needs no changes. Loads VLLM_API_KEY from .env.dev.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env.dev ]; then
  echo "ERROR: .env.dev not found (holds VLLM_API_KEY)." >&2
  exit 1
fi
# shellcheck disable=SC1091
set -a; source .env.dev; set +a
: "${VLLM_API_KEY:?VLLM_API_KEY not set in .env.dev}"

source venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0

exec vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ \
  --host 127.0.0.1 \
  --served-model-name ParaFrames/ParaClient-v2.2 \
  --quantization awq_marlin \
  --enable-lora \
  --max-loras 3 \
  --max-lora-rank 16 \
  --lora-modules ParaFrames/ParaClient-math-v2.2=adapters14b/math \
                 ParaFrames/ParaClient-language-v2.2=adapters14b/language_arts \
                 ParaFrames/ParaClient-cad-v2.2=adapters14b/cad \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92
