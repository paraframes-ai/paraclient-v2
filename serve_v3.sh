#!/usr/bin/env bash
# Free-tier ParaClient v3 (Qwen2.5-14B-Instruct, GGUF Q4_K_M) on CPU via
# llama.cpp on :8003. CPU-ONLY. NOTE: a 14B on 8 CPU cores is SLOW (~2-4 tok/s),
# so v3 is opt-in, not the free default (v2/7B is the responsive free default).
# The gateway routes version=v3 here (V3_URL) with the ParaClient tutoring prompt.
set -euo pipefail
cd "$(dirname "$0")"
MODEL=models/qwen2.5-14b-gguf/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf
exec llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  --host 127.0.0.1 --port 8003 \
  -c 8192 -t 6 \
  --alias paraclient-v3
