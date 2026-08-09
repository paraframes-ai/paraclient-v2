#!/usr/bin/env bash
# On-prem safety classifier: ShieldGemma-2B (GGUF Q4_K_M) via llama.cpp on :8004.
# CPU-ONLY (GPU is fully committed to the v4 tutor; a 2nd CUDA process there would
# risk OOM-ing the live endpoint). --parallel 4 so the gateway can run the 4 K-12
# policy checks concurrently instead of serially.
set -euo pipefail
cd "$(dirname "$0")"
exec llama.cpp/build/bin/llama-server \
  -m models/shieldgemma-2b-gguf/shieldgemma-2b.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8004 \
  -c 4096 -t 4 --parallel 4 --alias shieldgemma-2b
