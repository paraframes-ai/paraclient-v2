#!/usr/bin/env python3
"""
build_dataset.py — Build mode-tagged `messages` JSONL for ParaFrames tutor adapters.

Design goals:
- Subject-agnostic: same code path for math / civics / language_arts / general.
- Two behavior MODES baked into the data so ONE adapter can do both:
    * "socratic"      -> tutor withholds the answer, asks guiding questions.
    * "graduated_hint"-> tutor guides first, then escalates to hints / worked steps.
- Only clean, commercially-usable sources are enabled by default.

Currently implemented source:
  - GSM8K "socratic" config  (license: MIT)  -> subject: math

Sources that need a license check before enabling for a COMMERCIAL product:
  - MathDial (eth-nlped/mathdial)  -> confirm license, then add a builder.
  - Eedi Question-Anchored...      -> NON-COMMERCIAL, do NOT use here.

For civics / language_arts / general you will SYNTHESIZE dialogues (separate
step) and drop them into data/raw/<subject>.synth.jsonl in the same schema this
script emits; then this script just validates + merges them.

Output schema (one JSON object per line):
{
  "subject": "math",
  "mode": "socratic" | "graduated_hint",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
    ...
  ]
}
"""
import argparse
import json
import os
import re
from pathlib import Path

# --- System prompts per (subject, mode). Kept short; the app injects the real
#     production system prompt at inference. These exist so the adapter learns
#     to CONDITION its behavior on the mode signal. ---------------------------

SUBJECT_LABEL = {
    "math": "math",
    "civics": "civics",
    "language_arts": "language arts",
    "general": "general study",
}

def system_prompt(subject: str, mode: str) -> str:
    subj = SUBJECT_LABEL.get(subject, subject)
    if mode == "socratic":
        return (
            f"You are a Socratic {subj} tutor for a K-12 student. "
            f"Never state the final answer. Guide the student to it with one "
            f"clear question at a time. If they are stuck, make your question "
            f"smaller and more concrete — but do not solve it for them."
        )
    elif mode == "graduated_hint":
        return (
            f"You are a {subj} homework tutor for a K-12 student. "
            f"Guide with questions first. If the student stays stuck, escalate "
            f"your support step by step: a broader hint, then a partial worked "
            f"step, so they can make progress and finish. Never just dump the "
            f"full answer up front."
        )
    raise ValueError(f"unknown mode: {mode}")


# --- GSM8K socratic builder -------------------------------------------------
# The GSM8K "socratic" config interleaves the worked solution with Socratic
# sub-questions. Each answer line looks like:
#     "How many clips did she sell in May? ** She sold 48/2 = <<48/2=24>>24 ...
# i.e.  "<sub-question> ** <sub-answer with <<calc>> annotations>"
# We turn that into a multi-turn guiding dialogue.

_CALC_RE = re.compile(r"<<[^>]*>>")          # strip <<48/2=24>> calculator tags
_FINAL_RE = re.compile(r"####\s*(.+)\s*$")   # final answer marker


def _clean(text: str) -> str:
    return _CALC_RE.sub("", text).strip()


def _parse_gsm8k_socratic(question: str, answer: str):
    """Yield (sub_question, sub_answer) steps and the final answer string."""
    final = None
    m = _FINAL_RE.search(answer)
    if m:
        final = m.group(1).strip()
        answer = _FINAL_RE.sub("", answer).strip()

    steps = []
    for line in answer.splitlines():
        line = line.strip()
        if not line:
            continue
        if "**" in line:
            q, a = line.split("**", 1)
            steps.append((_clean(q), _clean(a)))
        else:
            # continuation of previous sub-answer
            if steps:
                sq, sa = steps[-1]
                steps[-1] = (sq, (sa + " " + _clean(line)).strip())
    return steps, final


def build_gsm8k_socratic(limit: int | None, grade_floor: bool = True):
    """
    Emit BOTH modes from each GSM8K-socratic problem:
      - socratic:       tutor asks the sub-questions, never gives the final answer.
      - graduated_hint: same guiding path, but the last turn escalates to a
                        worked final step so a stuck student can finish.
    """
    from datasets import load_dataset  # imported lazily so --help works offline

    ds = load_dataset("openai/gsm8k", "socratic", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    rows = []
    for ex in ds:
        q = ex["question"].strip()
        steps, final = _parse_gsm8k_socratic(q, ex["answer"])
        if not steps:
            continue

        # ---- socratic mode: guiding questions only, no final answer ----
        soc = [{"role": "system", "content": system_prompt("math", "socratic")},
               {"role": "user", "content": q}]
        # Tutor opens with the first guiding sub-question.
        soc.append({"role": "assistant", "content": steps[0][0]})
        # Simulate the student answering each sub-step; tutor asks the next.
        for i in range(1, len(steps)):
            soc.append({"role": "user", "content": steps[i - 1][1]})
            soc.append({"role": "assistant", "content": steps[i][0]})
        # Final turn: student gives last sub-answer; tutor affirms WITHOUT
        # stating the numeric answer (stays Socratic).
        soc.append({"role": "user", "content": steps[-1][1]})
        soc.append({"role": "assistant",
                    "content": "Good — now put those together. What total do you get, and how can you check it?"})
        rows.append({"subject": "math", "mode": "socratic", "messages": soc})

        # ---- graduated_hint mode: guide, then escalate to the worked answer ----
        grad = [{"role": "system", "content": system_prompt("math", "graduated_hint")},
                {"role": "user", "content": q}]
        grad.append({"role": "assistant", "content": steps[0][0]})
        for i in range(1, len(steps)):
            grad.append({"role": "user", "content": steps[i - 1][1]})
            grad.append({"role": "assistant", "content": steps[i][0]})
        # Escalation turn: student is stuck -> tutor gives the worked final step.
        grad.append({"role": "user", "content": "I'm still stuck, can you help me finish?"})
        if final is not None:
            grad.append({"role": "assistant",
                         "content": (f"No problem — let's finish it together. Combining the steps above, "
                                     f"the result is {final}. Walk back through why each step led there so it sticks.")})
        else:
            grad.append({"role": "assistant",
                         "content": "No problem — let's finish it together by combining the steps above one at a time."})
        rows.append({"subject": "math", "mode": "graduated_hint", "messages": grad})

    return rows


# --- Synthetic / pre-built subject files ------------------------------------

def load_synth(subject: str, raw_dir: Path):
    """
    Load synthesized dialogues for non-math subjects if present.
    Expected file: data/raw/<subject>.synth.jsonl  in the SAME output schema.
    (You produce these via the LLM-propose / human-review loop, then drop here.)
    """
    f = raw_dir / f"{subject}.synth.jsonl"
    if not f.exists():
        return []
    rows = []
    with open(f) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- Validation -------------------------------------------------------------

def validate(rows):
    ok = []
    for i, r in enumerate(rows):
        if r.get("subject") not in SUBJECT_LABEL:
            raise ValueError(f"row {i}: bad subject {r.get('subject')!r}")
        if r.get("mode") not in ("socratic", "graduated_hint"):
            raise ValueError(f"row {i}: bad mode {r.get('mode')!r}")
        msgs = r.get("messages")
        if not msgs or msgs[0]["role"] != "system":
            raise ValueError(f"row {i}: messages must start with a system turn")
        # must alternate user/assistant after system, end on assistant
        roles = [m["role"] for m in msgs[1:]]
        if roles[-1] != "assistant":
            raise ValueError(f"row {i}: dialogue must end on an assistant turn")
        ok.append(r)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True,
                    choices=list(SUBJECT_LABEL.keys()))
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of source problems (for quick test runs)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if args.subject == "math":
        print("[*] Building math from GSM8K-socratic (MIT license)...")
        rows += build_gsm8k_socratic(limit=args.limit)
    else:
        print(f"[*] Loading synthesized {args.subject} dialogues (if any)...")
        rows += load_synth(args.subject, raw_dir)
        if not rows:
            print(f"[!] No data for '{args.subject}' yet.")
            print(f"    Create {raw_dir}/{args.subject}.synth.jsonl via the")
            print(f"    synthesis + human-review step, then re-run.")
            return

    rows = validate(rows)

    out_path = out_dir / f"{args.subject}.jsonl"
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_soc = sum(1 for r in rows if r["mode"] == "socratic")
    n_grad = sum(1 for r in rows if r["mode"] == "graduated_hint")
    print(f"[✓] Wrote {len(rows)} examples to {out_path}")
    print(f"    socratic: {n_soc}   graduated_hint: {n_grad}")


if __name__ == "__main__":
    main()
