#!/usr/bin/env bash
# Unattended: wait for the CAD adapter training to finish, then train the circuit
# 3B LoRA on the freed L4. Does NOT restart serving — that + smoke-testing is done
# in-session once this completes (so results can be inspected). Detailed training
# output -> out/train_circuit.log; progress markers -> this job's stdout.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p out
log(){ echo "[$(date '+%F %T')] $*"; }

log "waiting for CAD training (train_adapter.py --subject cad) to finish..."
while pgrep -f "train_adapter.py --subject cad" >/dev/null 2>&1; do sleep 30; done
log "CAD training process ended."
if [ -f adapters/cad/adapter_config.json ]; then
  log "adapters/cad present (CAD OK)."
else
  log "WARNING: adapters/cad MISSING — CAD training may have failed."
fi

log "waiting for the L4 to release memory..."
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do sleep 5; done
log "GPU free. Training circuit LoRA on Qwen2.5-3B-Instruct (may download the base first)..."

source venv/bin/activate
python train_adapter.py --subject circuits --data data/circuit.jsonl \
  --base Qwen/Qwen2.5-3B-Instruct --no-flash-attn > out/train_circuit.log 2>&1
rc=$?
if [ -f adapters/circuits/adapter_config.json ]; then
  log "circuit adapter trained OK (rc=$rc) -> adapters/circuits"
else
  log "WARNING: circuit training FAILED (rc=$rc); see out/train_circuit.log"
  tail -15 out/train_circuit.log
fi
log "orchestrate_circuit: done."
