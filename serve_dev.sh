#!/usr/bin/env bash
# Launch the DEV vLLM environment (internal, key-protected).
# Loads VLLM_API_KEY from .env.dev so the key stays out of the process argv (ps).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f .env.dev ]; then
  echo "ERROR: .env.dev not found (holds VLLM_API_KEY). Create it first." >&2
  exit 1
fi

# shellcheck disable=SC1091
set -a; source .env.dev; set +a
: "${VLLM_API_KEY:?VLLM_API_KEY not set in .env.dev}"

source venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=0

exec vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 \
  --served-model-name ParaFrames/ParaClient-v2.2 \
  --enable-lora \
  --max-loras 3 \
  --lora-modules ParaFrames/ParaClient-math-v2.2=adapters/math \
                 ParaFrames/ParaClient-language-v2.2=adapters/language_arts \
                 ParaFrames/ParaClient-cad-v2.2=adapters/cad \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9
