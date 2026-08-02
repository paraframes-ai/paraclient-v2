#!/usr/bin/env python3
"""Merge a LoRA adapter into its base and save a standalone fp16 HF model
(CPU) -- the input for GGUF conversion + CPU serving via llama.cpp.

    python merge_lora.py --base Qwen/Qwen2.5-3B-Instruct \
        --adapter adapters/circuits --out models/circuit-merged
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

print(f"[*] loading base {a.base} (CPU, fp16)...", flush=True)
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(
    a.base, torch_dtype=torch.float16, device_map="cpu")
print(f"[*] applying + merging adapter {a.adapter}...", flush=True)
model = PeftModel.from_pretrained(model, a.adapter)
model = model.merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True)
tok.save_pretrained(a.out)
print(f"[✓] merged -> {a.out}", flush=True)
