#!/usr/bin/env python3
"""
synthesize_dialogues.py — Turn licensed SEED content into mode-tagged tutoring
dialogues for the Civics and Language Arts adapters.

PROVENANCE MODEL (read this):
  The dialogues this script produces are only as clean as TWO inputs:
    1. the SEED content (FairytaleQA for LA; NAEP/CivEd items for Civics), and
    2. the GENERATOR model you plug in below.
  This script is generator-AGNOSTIC on purpose: it calls whatever model you
  point --generator-url at. Use a model whose output license permits training a
  commercial model. The script does not (and cannot) clear that for you.

PIPELINE:
  seed loader  ->  synthesis core (adds scaffolding, two modes)  ->  your schema
  Output rows match build_dataset.py exactly, so you can run them through its
  validator and train with train_adapter.py unchanged.

  socratic        -> tutor withholds the answer, guides with questions
  graduated_hint  -> tutor guides, then escalates to hints/worked steps

USAGE:
  # 1. put licensed seed files here:
  #    data/seed/fairytaleqa.jsonl     (LA)
  #    data/seed/naep_civics.jsonl     (Civics)
  # 2. run against a generator you've cleared for commercial output use:
  python scripts/synthesize_dialogues.py \
      --subject language_arts \
      --seed data/seed/fairytaleqa.jsonl \
      --generator-url http://localhost:8001/v1 \
      --generator-model my-cleared-open-model \
      --out data/raw/language_arts.synth.jsonl

  # then validate + merge into training file:
  python scripts/build_dataset.py --subject language_arts
"""
import argparse
import json
import re
import time
from pathlib import Path


# --------------------------------------------------------------------------
# SEED LOADERS — normalize each source into a common "seed item":
#   {"subject", "grade", "topic", "context", "question", "reference_answer"}
# `context` is the passage/stimulus (may be empty for pure Q&A civics items).
# --------------------------------------------------------------------------

def load_fairytaleqa(path: Path):
    """
    FairytaleQA (CC BY 4.0): reading-comprehension Q&A over public-domain
    fairy tales. Expected fields per row (adapt to the copy you download;
    FairytaleQA ships as CSVs — pre-convert to JSONL with these keys):
      story_text, question, answer, attribute (skill), (optional) grade
    """
    for row in _read_jsonl(path):
        yield {
            "subject": "language_arts",
            "grade": row.get("grade", "elementary"),
            "topic": row.get("attribute", "reading comprehension"),
            "context": (row.get("story_text") or row.get("context") or "").strip(),
            "question": (row.get("question") or "").strip(),
            "reference_answer": (row.get("answer") or "").strip(),
        }


def load_naep_civics(path: Path):
    """
    NAEP / CivEd released civics items. Expected fields per row (convert the
    released items into JSONL with these keys):
      question, answer (or correct_choice), grade, (optional) content_area,
      (optional) stimulus/passage
    NOTE: civics is the highest-review subject; keep the reference_answer so the
    reviewer can verify the tutor never distorts the sourced fact.
    """
    for row in _read_jsonl(path):
        yield {
            "subject": "civics",
            "grade": row.get("grade", "8"),
            "topic": row.get("content_area", "civics"),
            "context": (row.get("stimulus") or row.get("passage") or "").strip(),
            "question": (row.get("question") or "").strip(),
            "reference_answer": (row.get("answer")
                                 or row.get("correct_choice") or "").strip(),
        }


SEED_LOADERS = {
    "language_arts": load_fairytaleqa,
    "civics": load_naep_civics,
}


def _read_jsonl(path: Path):
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# --------------------------------------------------------------------------
# SYNTHESIS CORE — build the generation prompt. The scaffolding structure is
# grounded in the CIMA tutor-action taxonomy (Question / Hint / Correction /
# Confirmation) + MathDial teacher-moves, so the dialogue escalates support
# the way a real tutor does rather than dumping the answer.
# --------------------------------------------------------------------------

SUBJECT_LABEL = {"language_arts": "language arts", "civics": "civics"}

_MODE_INSTRUCTION = {
    "socratic": (
        "Produce a SOCRATIC tutoring dialogue. The tutor NEVER states the final "
        "answer. The tutor guides the student with one focused question at a "
        "time, using the reference answer only to steer — never to reveal. If "
        "the student is stuck, the tutor makes the question smaller, not easier "
        "to cheat. End with the student reasoning toward the idea themselves."
    ),
    "graduated_hint": (
        "Produce a GRADUATED-HINT homework dialogue. The tutor guides with "
        "questions first, but when the student stays stuck it escalates support "
        "step by step: a broader hint, then a partial worked step, so the "
        "student can finish. The tutor never opens with the full answer."
    ),
}


def build_generation_prompt(seed: dict, mode: str) -> list:
    subj = SUBJECT_LABEL[seed["subject"]]
    ctx = f"\n\nSOURCE PASSAGE:\n{seed['context']}" if seed["context"] else ""
    ref = f"\nReference answer (for the tutor's eyes only, do NOT reveal verbatim in socratic mode): {seed['reference_answer']}" \
        if seed["reference_answer"] else ""

    sys = (
        f"You generate training data: realistic K-12 {subj} tutoring dialogues "
        f"for grade level {seed['grade']}. {_MODE_INSTRUCTION[mode]}\n\n"
        "Output STRICT JSON only — no prose, no markdown fences — as a list of "
        "turns: [{\"role\":\"user\"|\"assistant\",\"content\":\"...\"}, ...]. "
        "The FIRST turn is role \"user\" (the student). Alternate strictly. The "
        "LAST turn must be role \"assistant\" (the tutor). 6–12 turns. Keep it "
        "age-appropriate, warm, and free of anything unsuitable for a child."
    )
    if seed["subject"] == "civics":
        sys += (" For civics: stay strictly neutral. Present the sourced civic "
                "fact or the multiple documented perspectives; never advocate a "
                "partisan position or inject the tutor's opinion.")

    user = (f"Topic: {seed['topic']}\nStudent's starting question or task: "
            f"{seed['question']}{ctx}{ref}\n\nGenerate the dialogue now.")

    return [{"role": "system", "content": sys},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# GENERATOR ADAPTER — OpenAI-compatible chat endpoint (works with vLLM, or any
# OpenAI-compatible server). Swap the body of `generate` to target something
# else; nothing above here depends on the provider.
# --------------------------------------------------------------------------

def generate(client, model, messages, max_tokens=1200, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.8)
            return r.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 — surface + retry any API error
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"generation failed after {retries} tries: {last}")


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_turns(raw: str):
    """Parse the model's JSON turn list; tolerate accidental code fences."""
    txt = _FENCE.sub("", raw.strip())
    turns = json.loads(txt)  # will raise if the model didn't obey — caught below
    if not isinstance(turns, list) or not turns:
        raise ValueError("not a non-empty list")
    for t in turns:
        if t.get("role") not in ("user", "assistant") or "content" not in t:
            raise ValueError("bad turn shape")
    if turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        raise ValueError("must start on user, end on assistant")
    return turns


def system_prompt_for(subject: str, mode: str) -> str:
    # Mirror build_dataset.py's system prompts so the training rows are
    # consistent across math (real data) and synthesized subjects.
    subj = SUBJECT_LABEL[subject]
    if mode == "socratic":
        return (f"You are a Socratic {subj} tutor for a K-12 student. Never "
                f"state the final answer. Guide the student to it with one clear "
                f"question at a time.")
    return (f"You are a {subj} homework tutor for a K-12 student. Guide with "
            f"questions first; if the student stays stuck, escalate support step "
            f"by step so they can finish. Never just dump the full answer.")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, choices=list(SEED_LOADERS))
    ap.add_argument("--seed", required=True, help="path to seed JSONL")
    ap.add_argument("--out", required=True, help="output .synth.jsonl path")
    ap.add_argument("--generator-url", required=True,
                    help="OpenAI-compatible base URL for the CLEARED generator")
    ap.add_argument("--generator-model", required=True)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap seed items (for a small trial run)")
    ap.add_argument("--modes", nargs="+", default=["socratic", "graduated_hint"],
                    choices=["socratic", "graduated_hint"])
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.generator_url, api_key=args.api_key)

    seeds = list(SEED_LOADERS[args.subject](Path(args.seed)))
    if args.limit:
        seeds = seeds[:args.limit]
    if not seeds:
        print(f"[!] No seed items loaded from {args.seed}. Check the file format.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    with open(out_path, "w") as fh:
        for i, seed in enumerate(seeds):
            if not seed["question"]:
                skipped += 1
                continue
            for mode in args.modes:
                prompt = build_generation_prompt(seed, mode)
                try:
                    raw = generate(client, args.generator_model, prompt)
                    turns = parse_turns(raw)
                except Exception as e:  # noqa: BLE001
                    print(f"  [skip] item {i} mode {mode}: {e}")
                    skipped += 1
                    continue

                messages = [{"role": "system",
                             "content": system_prompt_for(args.subject, mode)}]
                messages.extend(turns)
                row = {"subject": args.subject, "mode": mode,
                       "messages": messages}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1

            if (i + 1) % 25 == 0:
                print(f"  ...processed {i + 1}/{len(seeds)} seeds "
                      f"({written} dialogues written)")

    print(f"[✓] Wrote {written} dialogues to {out_path}  (skipped {skipped})")
    print(f"    Next: human-review this file (REQUIRED for civics), then")
    print(f"    `python scripts/build_dataset.py --subject {args.subject}` to "
          f"validate + merge for training.")
    if args.subject == "civics":
        print("    ⚠ Civics: review every dialogue for neutrality before training.")


if __name__ == "__main__":
    main()
