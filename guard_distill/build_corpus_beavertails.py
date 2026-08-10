#!/usr/bin/env python3
"""
guard_distill/build_corpus_beavertails.py — real training data from BeaverTails.

    HF_HOME=/orcd/pool/005/avb/hf_cache \
    python -m guard_distill.build_corpus_beavertails \
        --out data/guard_labeled.jsonl --holdout-out data/guard_holdout.jsonl

Maps PKU-Alignment/BeaverTails' 14 HUMAN harm categories onto our 4 block
policies (sexual / violence / hate / dangerous) and emits labeled rows in the
{text, surface, labels} shape train_student / eval_student consume. Human labels
mean NO ShieldGemma teacher pass is needed (and no gated model on ORCD): the
student learns directly from human red-team annotations, and ShieldGemma stays
as the eval baseline it must match on the box.

Run on the ORCD LOGIN node (it has internet; compute nodes may not). Downloads
into HF_HOME, so the sbatch training job can run offline from the cache.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .labels import LABELS
except ImportError:
    from labels import LABELS

# BeaverTails category -> our policy. Only categories that map to one of our four
# BLOCK harms are used; the rest are left as 0 across all four (i.e. "not one of
# the harms this guard blocks"), which is correct — e.g. self_harm is an
# ESCALATION signal handled by the heuristic, not a block label here.
CAT_MAP = {
    "sexually_explicit,adult_content":            "sexual",
    "violence,aiding_and_abetting,incitement":    "violence",
    "terrorism,organized_crime":                  "violence",
    "child_abuse":                                "violence",
    "animal_abuse":                               "violence",
    "hate_speech,offensive_language":             "hate",
    "discrimination,stereotype,injustice":        "hate",
    "drug_abuse,weapons,banned_substance":        "dangerous",
    "financial_crime,property_crime,theft":       "dangerous",
    "privacy_violation":                          "dangerous",
    "non_violent_unethical_behavior":             "dangerous",
    # unmapped (not one of our 4 block-harms):
    #   self_harm (escalation -> heuristic), controversial_topics,politics,
    #   misinformation_regarding_ethics,laws_and_safety
}


def _row(ex: dict):
    cats = ex.get("category") or {}
    lab = {k: 0.0 for k in LABELS}
    for bt_cat, on in cats.items():
        if on and bt_cat in CAT_MAP:
            lab[CAT_MAP[bt_cat]] = 1.0
    text = (ex.get("response") or "").strip()
    if not text:
        return None
    # BeaverTails labels the RESPONSE's harmfulness -> output-surface training.
    return {"text": text, "surface": "output", "source": "beavertails",
            "labels": lab}


def _dump(split: str, out: str, limit: int):
    from datasets import load_dataset
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    n, pos = 0, {k: 0 for k in LABELS}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for ex in ds:
            r = _row(ex)
            if not r:
                continue
            f.write(json.dumps(r) + "\n")
            n += 1
            for k in LABELS:
                if r["labels"][k] >= 0.5:
                    pos[k] += 1
    print(f"[beavertails] {split}: wrote {n} rows -> {out} | positives {pos}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="30k_train")
    ap.add_argument("--holdout-split", default="30k_test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/guard_labeled.jsonl")
    ap.add_argument("--holdout-out", default="data/guard_holdout.jsonl")
    args = ap.parse_args()
    _dump(args.split, args.out, args.limit)
    _dump(args.holdout_split, args.holdout_out, args.limit)


if __name__ == "__main__":
    main()
