#!/usr/bin/env python3
"""
guard_distill/train_student.py — distill ShieldGemma-2B into a tiny single-pass
multi-label classifier. RUN ON ORCD (GPU); this box has no CUDA.

    python -m guard_distill.train_student \
        --data data/guard_labeled.jsonl \
        --base distilbert-base-uncased \
        --out models/guard-student

The student is an encoder with a 4-way multi-label head (one sigmoid per policy
in labels.LABELS order). It sees the whole content in ONE forward pass and emits
all four P(yes) at once — vs ShieldGemma's four full-content prefills. That is the
entire speed win: ~140M/66M params, one pass, <1s on CPU.

DISTILLATION, not hard labels: we train BCEWithLogitsLoss against the teacher's
SOFT P(yes) in [0,1] (HF's multi_label_classification loss is exactly BCE over
float targets), so the student inherits ShieldGemma's calibration, not just its
0/1 decisions. Calibrate serving thresholds afterwards with eval_student.py —
never trade child-safety recall for speed.

Accuracy upgrade: pass --base microsoft/deberta-v3-small (needs sentencepiece);
distilbert is the zero-friction default.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .labels import LABELS, MAX_LEN, student_input
except ImportError:
    from labels import LABELS, MAX_LEN, student_input


def _load(path: str):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if not d.get("labels"):
            continue                          # skip teacher-error rows
        rows.append({
            "text": student_input(d["text"], d.get("surface", "output")),
            "labels": [float(d["labels"][k]) for k in LABELS],
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/guard_labeled.jsonl")
    ap.add_argument("--base", default="distilbert-base-uncased")
    ap.add_argument("--out", default="models/guard-student")
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--val-frac", type=float, default=0.1)
    args = ap.parse_args()

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification,
                              TrainingArguments, Trainer,
                              DataCollatorWithPadding)

    if not torch.cuda.is_available():
        print("[train_student] WARNING: no CUDA — this is meant for ORCD GPU.")

    rows = _load(args.data)
    print(f"[train_student] {len(rows)} labeled rows | labels={LABELS}")
    split = max(1, int(len(rows) * args.val_frac))
    val, train = rows[:split], rows[split:]

    tok = AutoTokenizer.from_pretrained(args.base)

    def _tok(batch):
        enc = tok(batch["text"], truncation=True, max_length=MAX_LEN)
        enc["labels"] = batch["labels"]
        return enc

    ds_train = Dataset.from_list(train).map(_tok, batched=True,
                                            remove_columns=["text"])
    ds_val = Dataset.from_list(val).map(_tok, batched=True,
                                        remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={i: l for i, l in enumerate(LABELS)},
        label2id={l: i for i, l in enumerate(LABELS)})

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        pred = (probs >= 0.5).astype(int)
        gold = (np.array(labels) >= 0.5).astype(int)
        # macro recall — the safety-critical metric (missing a violation is worst)
        rec = []
        for j in range(len(LABELS)):
            tp = int(((pred[:, j] == 1) & (gold[:, j] == 1)).sum())
            fn = int(((pred[:, j] == 0) & (gold[:, j] == 1)).sum())
            rec.append(tp / (tp + fn) if (tp + fn) else 1.0)
        return {"macro_recall": float(np.mean(rec)),
                **{f"recall_{LABELS[j]}": rec[j] for j in range(len(LABELS))}}

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="macro_recall",
        greater_is_better=True, logging_steps=25, report_to=[])

    trainer = Trainer(model=model, args=targs,
                      train_dataset=ds_train, eval_dataset=ds_val,
                      tokenizer=tok,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    (Path(args.out) / "labels.json").write_text(json.dumps(LABELS))
    print(f"[train_student] saved student -> {args.out}")
    print("[train_student] NEXT: eval_student.py to set fail-closed thresholds, "
          "then serve_student.py on :8005 and flip GUARD_BACKEND=distilled.")


if __name__ == "__main__":
    main()
