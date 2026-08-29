#!/usr/bin/env python3
"""Evaluate exact theme-selection accuracy for a trained LoRA adapter."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/slideshow_theme_holdout.jsonl")
    ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter", default="adapters/slideshow_theme")
    ap.add_argument("--min-accuracy", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    correct, confusion = 0, Counter()
    for row in rows:
        prompt = tok.apply_chat_template(row["messages"][:-1], tokenize=False,
                                         add_generation_prompt=True)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=20, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        match = re.search(r'"theme"\s*:\s*"([a-z_]+)"', raw)
        predicted = match.group(1) if match else "invalid"
        expected = row["theme"]
        correct += predicted == expected
        confusion[(expected, predicted)] += 1
    accuracy = correct / len(rows) if rows else 0.0
    print(json.dumps({"rows": len(rows), "correct": correct,
                      "accuracy": round(accuracy, 4),
                      "min_accuracy": args.min_accuracy,
                      "gate": "PASS" if accuracy >= args.min_accuracy else "FAIL",
                      "errors": {f"{a}->{b}": n for (a, b), n in confusion.items()
                                 if a != b}}, indent=2, sort_keys=True))
    return 0 if accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    raise SystemExit(main())
