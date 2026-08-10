#!/usr/bin/env bash
# Free-tier ParaClient v2 (Qwen2.5-7B-Instruct, GGUF Q4_K_M) on CPU via
# llama.cpp on :8002. CPU-ONLY — runs alongside the GPU v4 tutor without
# touching the L4 GPU. The gateway routes version=v2 here (V2_URL) and applies
# the ParaClient tutoring system prompt on top (free tier = base + prompt, not
# the fine-tuned adapters, which are the paid v4 differentiator).
set -euo pipefail
cd "$(dirname "$0")"
MODEL=models/qwen2.5-7b-gguf/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf
# NOTE: speculative decoding (Qwen2.5-0.5B draft via --model-draft) was tried and
# measured here — it gave NO speedup on this 4-vCPU box (6.9 tok/s with draft vs
# ~7.0 without). On CPU the draft's own forward passes cost real cycles, so unless
# draft acceptance is very high the overhead cancels the memory-bandwidth win it
# gets "for free" on a GPU. Reverted to single-model; the draft gguf is kept in
# models/ for the eventual GPU/bigger-box path where it would pay off.
exec llama.cpp/build/bin/llama-server \
  -m "$MODEL" \
  -c 8192 -t 4 \
  --host 127.0.0.1 --port 8002 \
  --alias paraclient-v2
