#!/usr/bin/env bash
# Serve the CPU circuit model (Qwen2.5-3B, GGUF Q4_K_M) via llama.cpp on :8001.
# CPU-ONLY — runs alongside the GPU tutor without touching the L4 GPU. The
# gateway's /v1/circuit proxies here (CIRCUIT_URL=http://127.0.0.1:8001/v1) and
# does ERC-gated rejection sampling on top.
set -euo pipefail
cd "$(dirname "$0")"
exec llama.cpp/build/bin/llama-server \
  -m models/circuit-q4_k_m.gguf \
  --host 127.0.0.1 --port 8001 \
  -c 4096 -t 6
