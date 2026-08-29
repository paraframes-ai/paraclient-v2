#!/usr/bin/env python3
"""Compare the distilled guard with ShieldGemma without storing traffic text.

Exit zero only when both backends stayed available and the student was never
more permissive than the teacher. Reports contain aggregate counts and latency;
raw inputs and outputs are deliberately never printed or written.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from moderation import (DistilledGuardBackend, POLICIES,
                        ShieldGemmaBackend)

UNAVAILABLE = "_moderation_unavailable"
LABELS = {label for label, _ in POLICIES}


def _timed(backend, text: str, surface: str):
    started = time.monotonic()
    verdict = backend.classify(text, surface)
    return verdict, time.monotonic() - started


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def compare(rows: list[dict], teacher, student) -> tuple[dict, bool]:
    counts = Counter()
    categories = Counter()
    teacher_times, student_times = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for row in rows:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                counts["invalid_rows"] += 1
                continue
            surface = row.get("surface", "output")
            ft = pool.submit(_timed, teacher, text, surface)
            fs = pool.submit(_timed, student, text, surface)
            teacher_v, teacher_s = ft.result()
            student_v, student_s = fs.result()
            teacher_times.append(teacher_s)
            student_times.append(student_s)
            counts["compared"] += 1
            if UNAVAILABLE in teacher_v:
                counts["teacher_unavailable"] += 1
                continue
            if UNAVAILABLE in student_v:
                counts["student_unavailable"] += 1
                continue
            teacher_v &= LABELS
            student_v &= LABELS
            missing = teacher_v - student_v
            extra = student_v - teacher_v
            if missing:
                counts["student_more_permissive"] += 1
                categories.update(f"missed_{label}" for label in missing)
            elif extra:
                counts["student_more_restrictive"] += 1
                categories.update(f"extra_{label}" for label in extra)
            else:
                counts["exact_match"] += 1

    def latency(values):
        return {"p50_ms": round(statistics.median(values) * 1000, 1) if values else 0,
                "p95_ms": round(_percentile(values, 0.95) * 1000, 1)}

    report = {"counts": dict(sorted(counts.items())),
              "disagreement_categories": dict(sorted(categories.items())),
              "latency": {"teacher": latency(teacher_times),
                          "student": latency(student_times)}}
    passed = (counts["compared"] > 0 and not counts["invalid_rows"]
              and not counts["teacher_unavailable"]
              and not counts["student_unavailable"]
              and not counts["student_more_permissive"])
    return report, passed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="JSONL rows with text and optional input/output surface")
    ap.add_argument("--teacher-url", default=None)
    ap.add_argument("--student-url", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(line) for line in Path(args.data).read_text().splitlines()
            if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    report, passed = compare(
        rows, ShieldGemmaBackend(url=args.teacher_url),
        DistilledGuardBackend(url=args.student_url))
    report["gate"] = "PASS" if passed else "FAIL"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
