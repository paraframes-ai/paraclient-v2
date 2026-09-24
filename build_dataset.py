#!/usr/bin/env python3
"""
build_dataset.py — Build mode-tagged `messages` JSONL for ParaFrames tutor adapters.

Design goals:
- Subject-agnostic: same code path for math / civics / language_arts / general.
- Two behavior MODES baked into the data so ONE adapter can do both:
    * "socratic"      -> tutor withholds the answer, asks guiding questions.
    * "graduated_hint"-> tutor guides first, then escalates to hints / worked steps.
- Retain source provenance and review data for this CSAIL research project.

Currently implemented source:
  - GSM8K "socratic" config  (license: MIT)  -> subject: math

Check source license and attribution requirements before adding datasets.

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

def validate(rows, base="qwen"):
    ok = []
    for i, r in enumerate(rows):
        if r.get("subject") not in SUBJECT_LABEL:
            raise ValueError(f"row {i}: bad subject {r.get('subject')!r}")
        if r.get("mode") not in ("socratic", "graduated_hint"):
            raise ValueError(f"row {i}: bad mode {r.get('mode')!r}")
        msgs = r.get("messages")
        if not msgs or msgs[0]["role"] != "system":
            raise ValueError(f"row {i}: messages must start with a system turn")
        if not isinstance(msgs[0].get("content"), str):
            raise ValueError(f"row {i}: system content must be text")
        has_tools = bool(r.get("tools")) or any(m.get("tool_calls") or m.get("role") == "tool" for m in msgs)
        if has_tools:
            if base != "muse":
                raise ValueError(f"row {i}: native tool dialogues require --base muse")
            from tool_schema import validate_tool_dialogue
            validate_tool_dialogue(r)
            ok.append(r)
            continue
        for m in msgs:
            if not isinstance(m.get("content"), str):
                raise ValueError(f"row {i}: message content must be text")
            if m.get("recipient", "user") != "user":
                raise ValueError(f"row {i}: non-tool dialogues must be student-facing")
        # must alternate user/assistant after system, end on assistant
        roles = [m["role"] for m in msgs[1:]]
        if not roles or roles[0] != "user":
            raise ValueError(f"row {i}: dialogue must start on a user turn")
        if roles[-1] != "assistant":
            raise ValueError(f"row {i}: dialogue must end on an assistant turn")
        # Second gate mirroring synthesize_dialogues.parse_turns: reject any
        # dialogue whose turns don't strictly alternate user/assistant (e.g.
        # several tutor turns in a row from a malformed generation).
        expected = ["user", "assistant"]
        for j, role in enumerate(roles):
            if role != expected[j % 2]:
                raise ValueError(
                    f"row {i}: turns must strictly alternate user/assistant")
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
    ap.add_argument("--base", choices=("qwen", "muse"), default="qwen")
    ap.add_argument("--revision", default=None, help="HF revision; Muse defaults to the inspected commit")
    ap.add_argument("--tool-data", action="append", default=[],
                    help="merge reviewed Muse-native tool JSONL; repeat for multiple files")
    args = ap.parse_args()
    if args.limit is not None and args.limit <= 0:
        ap.error("--limit must be positive")

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
        if args.limit:
            rows = rows[:args.limit]

    tool_paths = [Path(path) for path in args.tool_data]
    default_tools = raw_dir / f"{args.subject}.tools.jsonl"
    if args.base == "muse" and default_tools.exists() and default_tools not in tool_paths:
        tool_paths.append(default_tools)
    for path in tool_paths:
        with path.open() as fh:
            extra = [json.loads(line) for line in fh if line.strip()]
        if any(not row.get("tools") for row in extra):
            raise ValueError(f"{path}: tool examples must include tools schemas")
        rows.extend(extra[:args.limit] if args.limit else extra)
    if not rows:
        raise SystemExit(f"No reviewed data for {args.subject}; create {raw_dir}/{args.subject}.synth.jsonl")
    if any(row.get("subject") != args.subject for row in rows):
        raise ValueError("Merged data contains a different subject")
    rows = validate(rows, base=args.base)
    from transformers import AutoTokenizer
    from model_support import MODELS, render_dialogue
    revision = args.revision
    inspection = Path(__file__).resolve().parent / "artifacts/muse_inspection.json"
    if args.base == "muse" and revision is None and inspection.exists():
        revision = json.loads(inspection.read_text())["revision"]
    tok = AutoTokenizer.from_pretrained(MODELS[args.base], revision=revision, token=False)
    for row in rows:
        # Preserve subject/mode/messages and native tool fields; text is additive.
        row["text"] = render_dialogue(tok, row, args.base)
        row["base_model"] = MODELS[args.base]
        row["base_revision"] = tok.init_kwargs.get("_commit_hash") or revision

    out_path = out_dir / f"{args.subject}.jsonl"
    # Refuse to overwrite a previous build; choose a new --out-dir instead.
    with open(out_path, "x") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_soc = sum(1 for r in rows if r["mode"] == "socratic")
    n_grad = sum(1 for r in rows if r["mode"] == "graduated_hint")
    print(f"[✓] Wrote {len(rows)} examples to {out_path}")
    print(f"    socratic: {n_soc}   graduated_hint: {n_grad}")


if __name__ == "__main__":
    main()
