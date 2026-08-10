#!/usr/bin/env python3
"""
guard_distill/label_teacher.py — teacher (ShieldGemma-2B) soft-labeling.

    python -m guard_distill.label_teacher \
        --in data/guard_corpus.jsonl --out data/guard_labeled.jsonl

Runs the SAME ShieldGemma classifier the gateway serves (moderation._policy_prompt
+ the guard server at GUARD_URL) over every corpus row and records the per-policy
P(yes) as SOFT labels — distillation targets, not just 0/1. Adds:
    {"labels": {"sexual": p, "violence": p, "hate": p, "dangerous": p}}
to each row, in LABELS order.

Runs anywhere the guard server is reachable. On the CPU box it is SLOW
(~2.4s/policy) — fine for a few hundred smoke-test rows, but run the full corpus
on ORCD where ShieldGemma-2B on a GPU labels the whole set in minutes. The soft
scores are the teacher's knowledge; BCE-distilling against them (not hard 0/1)
is what lets a much smaller student match its calibration.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Reuse the EXACT teacher: same prompt, same P(yes) extraction, same server.
from moderation import POLICIES, ShieldGemmaBackend  # type: ignore

try:
    from .labels import LABELS
except ImportError:
    from labels import LABELS


def _score_row(row: dict, teacher: ShieldGemmaBackend) -> dict:
    """Per-policy P(yes) from the teacher, via its own _score() (same request,
    prompt and probability extraction the gateway uses)."""
    text, surface = row["text"], row.get("surface", "output")
    try:
        labels = {label: round(teacher._score(text, surface, policy), 6)
                  for label, policy in POLICIES}
    except Exception as e:  # noqa: BLE001
        return {**row, "labels": None, "error": str(e)[:120]}
    return {**row, "labels": {k: labels.get(k, 0.0) for k in LABELS}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="data/guard_corpus.jsonl")
    ap.add_argument("--out", default="data/guard_labeled.jsonl")
    ap.add_argument("--url", default=None, help="guard server (default GUARD_URL)")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent rows; keep low on CPU, raise on GPU")
    ap.add_argument("--limit", type=int, default=0, help="label only first N (smoke test)")
    args = ap.parse_args()

    teacher = ShieldGemmaBackend(url=args.url, timeout=args.timeout)

    rows = [json.loads(l) for l in Path(args.inp).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = err = 0
    with out.open("w") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(lambda r: _score_row(r, teacher), rows):
            if res.get("labels") is None:
                err += 1
            else:
                ok += 1
            f.write(json.dumps(res) + "\n")
    print(f"[label_teacher] labeled {ok} rows ({err} errors) -> {out}")
    if ok:
        # quick sanity: mean P(yes) per label
        import statistics
        good = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        good = [g for g in good if g.get("labels")]
        for lab in LABELS:
            vals = [g["labels"][lab] for g in good]
            hi = sum(1 for v in vals if v >= 0.5)
            print(f"  {lab:10s} mean={statistics.mean(vals):.3f}  >=0.5: {hi}/{len(vals)}")


if __name__ == "__main__":
    main()
