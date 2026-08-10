#!/usr/bin/env bash
# Launch the ParaFrames API gateway (DEV SCAFFOLD): auth + safety filter +
# mode/subject routing + docs/slides, in front of the vLLM tutor.
#
# CPU-only — it forwards to vLLM on localhost:8000, so it runs alongside the
# tutor (serve_dev.sh) without using the GPU. Loads VLLM_API_KEY (the key the
# gateway uses to reach vLLM) from .env.dev.
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

# Bind to the TAILNET interface only (reachable by e2 over the tailnet, NOT the
# public internet). Fall back to the known tailnet IP if the CLI is unavailable.
HOST="$(tailscale ip -4 2>/dev/null | head -1 || true)"
HOST="${HOST:-100.122.196.7}"

exec python auth_gateway.py \
  --tutor-url http://127.0.0.1:8002/v1 \
  --host "$HOST" \
  --port 8080
