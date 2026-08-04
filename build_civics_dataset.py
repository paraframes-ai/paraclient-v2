#!/usr/bin/env python3
"""
build_civics_dataset.py — assemble K-12 civics *candidates* for human review.

Reads license-clean source items from civics_sources/*.jsonl and renders each
into tutoring-style candidate rows (one per requested grade band), tagged and
flagged per civics_schema. NOTHING here is training-ready: output goes to
data/civics_candidates.jsonl for the mandatory human-review gate. Approved rows
are promoted to data/civics.jsonl via --promote (which refuses anything not
marked review_status="approved").

Source item schema (civics_sources/*.jsonl), one JSON object per line:
  {
    "q": "What is the supreme law of the land?",
    "a": "the Constitution",
    "strand": "constitution",            # civics_schema.NAEP_STRANDS key
    "grade_bands": ["elementary","middle","high"],
    "time_varying": false,
    "concept": "",                       # stable teachable content (for time_varying items)
    "source": "USCIS-2008-Q1",
    "license": "public_domain",          # civics_schema.LICENSES key
    "attribution": ""                    # required when license == "cc_by_sa_4"
  }
"""
import argparse
import glob
import json
import os

import civics_schema as cs


def _time_varying_answer(item: dict) -> str:
    """Teach the stable concept + that the current fact is looked up live from an
    official source — never assert the volatile fact from memory. At serving time
    civics_agent.answer_time_varying runs the ReAct search/fetch loop against a
    .gov/.mil source and fills in the current, cited answer."""
    concept = (item.get("concept") or "").strip()
    tail = ("That can change over time, so instead of answering from memory I "
            "look it up from an official U.S. government (.gov) source and cite "
            "it, so the answer is current.")
    return f"{concept} {tail}".strip() if concept else tail


def to_candidate(item: dict, band: str) -> dict:
    time_varying = bool(item.get("time_varying", False))
    assistant = _time_varying_answer(item) if time_varying else item["a"]
    row = {
        "subject": "civics",
        "grade_band": band,
        "strand": item["strand"],
        "time_varying": time_varying,
        # How the serving layer answers this row: "static" = the trained answer
        # is used directly; "live_agent" = route to civics_agent for a fresh,
        # officially-sourced answer (time-varying facts).
        "route": "live_agent" if time_varying else "static",
        "source": item["source"],
        "license": item["license"],
        "attribution": item.get("attribution", ""),
        "review_status": "candidate",
        "reviewer": "",
        "review_notes": "",
        "messages": [
            {"role": "system", "content": cs.sys_for()},
            {"role": "user", "content": item["q"]},
            {"role": "assistant", "content": assistant},
        ],
    }
    return cs.validate(row)  # structural check only; humans judge content


def build(src_dir: str, out: str) -> tuple[int, int]:
    items = []
    for f in sorted(glob.glob(os.path.join(src_dir, "*.jsonl"))):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
    n = 0
    with open(out, "w") as w:
        for it in items:
            for band in it.get("grade_bands", list(cs.GRADE_BANDS)):
                w.write(json.dumps(to_candidate(it, band)) + "\n")
                n += 1
    return n, len(items)


def promote(candidates: str, out: str, approve_all: bool = False,
            reviewer: str = "") -> tuple[int, int]:
    kept = total = 0
    with open(out, "w") as w:
        with open(candidates) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                total += 1
                row = json.loads(line)
                if approve_all and row.get("review_status") != "rejected":
                    # Explicit, auditable bypass of the human-review gate: stamp
                    # every non-rejected candidate as approved by the given tag
                    # so the promoted file records that review was auto-approved.
                    row["review_status"] = "approved"
                    row["reviewer"] = reviewer or "auto-approved"
                    row["review_notes"] = ("auto-approved (human review bypassed "
                                           "by operator request)")
                if row.get("review_status") == "approved":
                    cs.validate(row, require_approved=True)
                    w.write(json.dumps(row) + "\n")
                    kept += 1
    return kept, total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="civics_sources")
    ap.add_argument("--candidates", default="data/civics_candidates.jsonl")
    ap.add_argument("--out", default="data/civics.jsonl")
    ap.add_argument("--promote", action="store_true",
                    help="promote human-approved candidates into the training file")
    ap.add_argument("--approve-all", action="store_true",
                    help="BYPASS human review: mark every non-rejected candidate "
                         "approved before promoting (records reviewer tag)")
    ap.add_argument("--reviewer", default="auto-approved",
                    help="reviewer tag stamped on rows when --approve-all is set")
    a = ap.parse_args()
    os.makedirs("data", exist_ok=True)

    if a.promote:
        kept, total = promote(a.candidates, a.out,
                              approve_all=a.approve_all, reviewer=a.reviewer)
        print(f"[✓] promoted {kept}/{total} approved rows -> {a.out}")
        if kept == 0:
            print("    (nothing approved yet — set review_status=\"approved\" on reviewed rows)")
    else:
        n, items = build(a.src, a.candidates)
        print(f"[✓] {n} candidates from {items} source items -> {a.candidates}")
        print("    NEXT: human review each candidate (edit + set review_status), then --promote")


if __name__ == "__main__":
    main()
