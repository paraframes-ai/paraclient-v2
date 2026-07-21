#!/usr/bin/env python3
"""
review_dialogues.py — Human-in-the-loop review for synthesized dialogues.

Synthesized data is NOT training-ready until a human has checked it. This is the
propose-then-review gate (same idea as ftcli): the generator proposes, a person
disposes. For civics this is mandatory — automated generation cannot be trusted
to stay neutral or factually faithful to the source on its own.

It runs a terminal review loop over a .synth.jsonl file, showing each dialogue
with automatic FLAGS to focus attention, and records your decision:
  [a]ccept  [r]eject  [e]dit-note  [s]kip  [q]uit-and-save

Accepted rows are written to <input>.reviewed.jsonl (only accepted rows), which
is what you then feed to build_dataset.py.

Automatic flags (heuristics to guide the reviewer, NOT a substitute for reading):
  - LEAK:     socratic-mode tutor turn appears to hand over a direct answer
  - STANCE:   civics dialogue contains opinion/advocacy language (neutrality risk)
  - LENGTH:   dialogue very short/long (often a malformed generation)
  - ENDROLE:  doesn't end on the tutor (assistant) turn

USAGE:
  python scripts/review_dialogues.py --in data/raw/civics.synth.jsonl
  # produces data/raw/civics.synth.reviewed.jsonl (accepted only)
"""
import argparse
import json
import re
from pathlib import Path

# Cheap neutrality/opinion signals for the civics stance flag. These are
# deliberately broad — better to over-flag and let the human judge.
_STANCE_PATTERNS = [
    r"\bI (think|believe|feel)\b", r"\bin my opinion\b", r"\byou should support\b",
    r"\bthe (right|correct) side\b", r"\bobviously\b", r"\bclearly the best\b",
    r"\beveryone (knows|agrees)\b", r"\bthe good guys\b", r"\bthe bad guys\b",
]
_STANCE_RE = re.compile("|".join(_STANCE_PATTERNS), re.IGNORECASE)

# Very rough answer-leak signal for socratic mode: the tutor states a
# conclusion outright ("the answer is", "it is because", a bare definition).
_LEAK_RE = re.compile(
    r"\b(the answer is|the correct answer|it is because|this means that|"
    r"the main idea is|the theme is|the reason is)\b", re.IGNORECASE)


def flags_for(row: dict):
    flags = []
    msgs = row.get("messages", [])
    mode = row.get("mode")
    subject = row.get("subject")

    if not msgs or msgs[0]["role"] != "system":
        flags.append("NOSYS")
        return flags
    body = msgs[1:]
    if not body or body[-1]["role"] != "assistant":
        flags.append("ENDROLE")
    if not (6 <= len(body) <= 24):
        flags.append("LENGTH")

    tutor_turns = " ".join(m["content"] for m in body if m["role"] == "assistant")
    if mode == "socratic" and _LEAK_RE.search(tutor_turns):
        flags.append("LEAK")
    if subject == "civics" and _STANCE_RE.search(tutor_turns):
        flags.append("STANCE")
    return flags


def render(row: dict, idx: int, total: int):
    print("\n" + "=" * 72)
    fl = flags_for(row)
    flag_str = ("  ⚠ FLAGS: " + ", ".join(fl)) if fl else "  (no auto-flags)"
    print(f"[{idx+1}/{total}]  subject={row.get('subject')}  "
          f"mode={row.get('mode')}{flag_str}")
    print("-" * 72)
    for m in row.get("messages", []):
        role = m["role"]
        if role == "system":
            print(f"  «system» {m['content'][:100]}...")
            continue
        who = "STUDENT" if role == "user" else "TUTOR  "
        print(f"  {who}: {m['content']}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default=None,
                    help="default: <input>.reviewed.jsonl")
    ap.add_argument("--flagged-only", action="store_true",
                    help="only review rows that tripped an auto-flag")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out) if args.out else inp.with_suffix(".reviewed.jsonl")

    rows = []
    with open(inp) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if args.flagged_only:
        rows = [r for r in rows if flags_for(r)]
        print(f"[*] Reviewing {len(rows)} FLAGGED rows only.")
    else:
        print(f"[*] Reviewing {len(rows)} rows. Flagged rows are marked ⚠.")

    accepted = []
    quit_early = False
    for i, row in enumerate(rows):
        render(row, i, len(rows))
        while True:
            choice = input("[a]ccept  [r]eject  [e]dit-note  [s]kip  [q]uit > ").strip().lower()
            if choice in ("a", "r", "e", "s", "q"):
                break
            print("  (please enter a, r, e, s, or q)")

        if choice == "a":
            accepted.append(row)
        elif choice == "e":
            note = input("  edit note (stored on the row, review later): ").strip()
            row["_review_note"] = note
            keep = input("  accept with note? [y/N] > ").strip().lower()
            if keep == "y":
                accepted.append(row)
        elif choice == "q":
            quit_early = True
            break
        # 'r' and 's' both drop the row from the accepted set

    with open(out, "w") as fh:
        for r in accepted:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[✓] Accepted {len(accepted)}/{len(rows)} -> {out}")
    if quit_early:
        print("    (quit early; remaining rows not reviewed)")
    print(f"    Feed accepted rows into training via build_dataset.py.")


if __name__ == "__main__":
    main()
