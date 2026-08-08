#!/usr/bin/env bash
# ParaClient-v4 (FP8): serve Gemma-4 12B + the 6 QLoRA adaptersG4 via vLLM 0.26
# in venv-g4, quantized to FP8 (W8A8 dynamic) ONLINE at load.
#
# Why FP8 instead of the old bitsandbytes path (serve_g4.sh):
#   - bnb does in-flight dequant -> slow, and forced --enforce-eager (no CUDA
#     graphs). Baseline measured ~6 tok/s on the L4.
#   - FP8 is native on the L4 (Ada/sm_89). Weights land at ~12 GB, leaving room
#     for CUDA graphs + a real KV cache -> much higher throughput, same quality
#     class (near-lossless vs bf16).
#   - No llmcompressor needed (it pins transformers<=5.10.1, but gemma4_unified
#     needs the installed 5.14.1). vLLM quantizes shard-by-shard at load; peak
#     stays under the 23 GB card.
#
# Keeps the SAME --served-model-name and --lora-module names as every prior
# config so auth_gateway.py needs NO changes. Loads VLLM_API_KEY from .env.dev.
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
# expandable_segments cuts fragmentation during the online quantize pass.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve google/gemma-4-12B-it \
  --host 127.0.0.1 \
  --served-model-name ParaFrames/ParaClient-v2.2 \
  --quantization fp8 \
  --enable-lora \
  --max-loras 6 \
  --max-lora-rank 16 \
  --lora-modules ParaFrames/ParaClient-math-v2.2=adaptersG4/math \
                 ParaFrames/ParaClient-language-v2.2=adaptersG4/language_arts \
                 ParaFrames/ParaClient-cad-v2.2=adaptersG4/cad \
                 ParaFrames/ParaClient-civics-v2.2=adaptersG4/civics \
                 ParaFrames/ParaClient-spreadsheet-v2.2=adaptersG4/spreadsheet \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.80
