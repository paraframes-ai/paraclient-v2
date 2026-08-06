#!/usr/bin/env bash
# On-prem safety classifier: ShieldGemma-2B (GGUF Q4_K_M) via llama.cpp on :8004.
# CPU-ONLY. The gateway's content-safety layer (moderation.ShieldGemmaBackend)
# screens student input + tutor output here — nothing leaves the box.
set -euo pipefail
cd "$(dirname "$0")"
exec llama.cpp/build/bin/llama-server \
  -m models/shieldgemma-2b-gguf/shieldgemma-2b.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8004 \
  -c 2048 -t 8 --alias shieldgemma-2b
