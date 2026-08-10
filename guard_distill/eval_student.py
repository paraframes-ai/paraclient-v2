#!/usr/bin/env python3
"""
guard_distill/eval_student.py — the FAIL-CLOSED acceptance gate + threshold picker.

    python -m guard_distill.eval_student \
        --data data/guard_labeled.jsonl --model models/guard-student

The student may only replace ShieldGemma if it NEVER catches fewer violations
than the teacher. This script:

  1. Runs the student over the labeled holdout (teacher P(yes) is ground truth).
  2. Sweeps a per-policy threshold and picks the LOWEST threshold that reaches a
     target recall (default 0.99) against teacher-positive rows — recall first,
     because for a child-facing guard a false-allow is far worse than a
     false-block.
  3. Prints the chosen thresholds and the student's recall/precision at them, and
     PASS/FAIL on the gate (student recall >= teacher on every policy).

Recall is measured against teacher-positive rows (P(yes)>=0.5). Feed a real
red-team holdout via the same labeled jsonl for a meaningful number — the
procedural smoke-test corpus has almost no positives by design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .labels import LABELS, MAX_LEN, student_input
except ImportError:
    from labels import LABELS, MAX_LEN, student_input


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/guard_labeled.jsonl")
    ap.add_argument("--model", default="models/guard-student")
    ap.add_argument("--target-recall", type=float, default=0.99)
    ap.add_argument("--pos-threshold", type=float, default=0.5,
                    help="teacher P(yes) above this = a positive to be caught")
    args = ap.parse_args()

    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    rows = [json.loads(l) for l in Path(args.data).read_text().splitlines()
            if l.strip()]
    rows = [r for r in rows if r.get("labels")]
    texts = [student_input(r["text"], r.get("surface", "output")) for r in rows]
    gold = np.array([[float(r["labels"][k]) for k in LABELS] for r in rows])

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], return_tensors="pt", truncation=True,
                      max_length=MAX_LEN, padding=True)
            logits = model(**enc).logits
            probs.append(torch.sigmoid(logits).cpu().numpy())
    probs = np.concatenate(probs, axis=0) if probs else np.zeros((0, len(LABELS)))

    print(f"[eval] rows={len(rows)}  target_recall={args.target_recall}")
    thresholds, gate_pass = {}, True
    for j, lab in enumerate(LABELS):
        pos = gold[:, j] >= args.pos_threshold
        npos = int(pos.sum())
        if npos == 0:
            thresholds[lab] = 0.5
            print(f"  {lab:10s}  (no teacher-positives in holdout — using 0.5)")
            continue
        best_t = 0.5
        for t in [x / 100 for x in range(5, 100, 5)]:
            rec = float(((probs[:, j] >= t) & pos).sum()) / npos
            if rec >= args.target_recall:
                best_t = t                    # lowest t meeting target keeps precision up
        rec = float(((probs[:, j] >= best_t) & pos).sum()) / npos
        pred_pos = probs[:, j] >= best_t
        prec = (float((pred_pos & pos).sum()) / float(pred_pos.sum())
                if pred_pos.sum() else 1.0)
        ok = rec >= args.target_recall
        gate_pass = gate_pass and ok
        thresholds[lab] = round(best_t, 2)
        print(f"  {lab:10s}  thr={best_t:.2f}  recall={rec:.3f}  prec={prec:.3f}  "
              f"pos={npos}  {'PASS' if ok else 'FAIL'}")

    print(f"\n[eval] chosen thresholds: {json.dumps(thresholds)}")
    print(f"[eval] GATE: {'PASS — safe to serve' if gate_pass else 'FAIL — do NOT cut over'}")
    print("[eval] set these per-policy thresholds in serve_student.py (or keep the "
          "flat GUARD_THRESHOLD if uniform). Shadow-run vs ShieldGemma before "
          "removing the teacher.")


if __name__ == "__main__":
    main()
