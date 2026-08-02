#!/usr/bin/env bash
# Stage 4 — build the CPU circuit model: install build tools, build llama.cpp,
# merge the circuit LoRA, convert HF->GGUF, quantize to Q4_K_M. All CPU; the
# GPU-served tutor keeps running. Progress -> this job's stdout; detail -> out/*.log
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p out models
export DEBIAN_FRONTEND=noninteractive
log(){ echo "[$(date '+%T')] $*"; }

log "installing build tools (cmake, build-essential)..."
sudo apt-get update -qq >/dev/null 2>&1
sudo apt-get install -y -qq cmake build-essential >/dev/null 2>&1 \
  && log "build tools OK" || { log "APT FAILED"; exit 1; }

if [ ! -x llama.cpp/build/bin/llama-quantize ]; then
  [ -d llama.cpp ] || { log "cloning llama.cpp..."; git clone --depth 1 https://github.com/ggml-org/llama.cpp >/dev/null 2>&1; }
  log "building llama.cpp with SYSTEM toolchain, OpenMP off (avoids linuxbrew/glibc mismatch)..."
  export PATH=/usr/bin:/usr/sbin:/bin:/sbin:$PATH   # system ld/gcc ahead of linuxbrew
  rm -rf llama.cpp/build
  cmake -S llama.cpp -B llama.cpp/build \
    -DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON > out/llama_build.log 2>&1
  cmake --build llama.cpp/build --target llama-quantize llama-server -j4 >> out/llama_build.log 2>&1
fi
if [ -x llama.cpp/build/bin/llama-quantize ] && [ -x llama.cpp/build/bin/llama-server ]; then
  log "llama.cpp binaries ready"
else
  log "BUILD FAILED -- see out/llama_build.log"; tail -20 out/llama_build.log; exit 2
fi

source venv/bin/activate
log "merging circuit LoRA into base (CPU, ~5-10 min)..."
python merge_lora.py --base Qwen/Qwen2.5-3B-Instruct --adapter adapters/circuits \
  --out models/circuit-merged > out/merge_circuit.log 2>&1
[ -f models/circuit-merged/config.json ] && log "merge OK" \
  || { log "MERGE FAILED"; tail -20 out/merge_circuit.log; exit 3; }

log "converting HF -> GGUF (f16)..."
CONV=$(ls llama.cpp/convert_hf_to_gguf.py 2>/dev/null || ls llama.cpp/convert-hf-to-gguf.py 2>/dev/null)
python "$CONV" models/circuit-merged --outfile models/circuit-f16.gguf --outtype f16 \
  > out/convert_circuit.log 2>&1
[ -f models/circuit-f16.gguf ] && log "convert OK ($(du -h models/circuit-f16.gguf | cut -f1))" \
  || { log "CONVERT FAILED"; tail -25 out/convert_circuit.log; exit 4; }

log "quantizing -> Q4_K_M..."
llama.cpp/build/bin/llama-quantize models/circuit-f16.gguf models/circuit-q4_k_m.gguf Q4_K_M \
  > out/quant_circuit.log 2>&1
[ -f models/circuit-q4_k_m.gguf ] && log "quant OK ($(du -h models/circuit-q4_k_m.gguf | cut -f1))" \
  || { log "QUANT FAILED"; tail -20 out/quant_circuit.log; exit 5; }

log "STAGE 4 BUILD COMPLETE -> models/circuit-q4_k_m.gguf"
