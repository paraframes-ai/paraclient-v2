#!/usr/bin/env bash
# Free-tier ParaClient v2 (Qwen2.5-7B-Instruct, GGUF Q4_K_M) on CPU via
# llama.cpp on :8002. CPU-ONLY — runs alongside the GPU v4 tutor without
# touching the L4 GPU. The gateway routes version=v2 here (V2_URL) and applies
# the ParaClient tutoring system prompt on top (free tier = base + prompt, not
# the fine-tuned adapters, which are the paid v4 differentiator).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
exec llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 --port 8002 \
  -c 8192 -t 6 \
  --alias paraclient-v2
