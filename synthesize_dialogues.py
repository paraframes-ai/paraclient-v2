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
import collections
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            # provenance only (not used in the prompt) — lets us report which
            # story each synthesized dialogue came from.
            "story": (row.get("story_name") or "unknown").strip(),
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
        "Produce a SOCRATIC tutoring dialogue. The tutor NEVER states, confirms, "
        "or restates the final answer — not at any point, and especially not in "
        "the final turn. The tutor guides with one focused question at a time, "
        "using the reference answer only to steer, never to reveal. If the "
        "student is stuck, the tutor makes the question smaller and more "
        "concrete, not easier to cheat. The LAST tutor turn MUST be a guiding "
        "question or a prompt for the student to put the answer in their own "
        "words; it must NOT state, confirm, or paraphrase the conclusion. The "
        "dialogue ends WITHOUT the tutor ever stating the answer — the student "
        "is the only one who articulates it."
    ),
    "graduated_hint": (
        "Produce a GRADUATED-HINT homework dialogue. The tutor's VERY FIRST "
        "turn MUST be a direct question addressed to the student — for example, "
        "open with \"What do you already know about ...?\" or \"Where in the "
        "passage would you look for ...?\". Begin the dialogue by engaging the "
        "student directly; the tutor never narrates its own process. Ask at "
        "least one or two genuine guiding questions BEFORE offering any hint, "
        "and do NOT state the answer within the first two turns. When the "
        "student stays stuck, escalate support gradually, in this order: "
        "guiding question -> a small hint -> a bigger hint -> a partial worked "
        "step -> and only then help the student reach the full answer. The "
        "tutor never opens with the answer and never jumps straight to it; the "
        "step-by-step escalation must be visible in the dialogue."
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
        "CONTRAST BETWEEN THE TWO MODES (respect this exactly): in SOCRATIC "
        "mode the tutor NEVER resolves the problem — the answer only ever comes "
        "out of the student's mouth. In GRADUATED-HINT mode the tutor DOES "
        "eventually help the student reach the answer, but ONLY after real, "
        "visible, step-by-step escalation — never immediately.\n"
        "The tutor always speaks DIRECTLY to the student and NEVER narrates its "
        "own thought process: do not write phrases like \"let me think\", "
        "\"let me read the passage again\", or any first-person meta-commentary "
        "about reading or reasoning — just tutor the student.\n\n"
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
                max_tokens=max_tokens, temperature=0.8,
                # Qwen3 defaults to "thinking mode" and emits <think>...</think>
                # before its reply, which breaks strict-JSON parsing below.
                # Disable it so the model returns only the JSON turn list.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            return r.choices[0].message.content.strip()
        except Exception as e:  # noqa: BLE001 — surface + retry any API error
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"generation failed after {retries} tries: {last}")


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")

# Backup safety net for meta-narration that slips past the prompt fix: strip a
# leading self-narration clause ("let me think", "let me ask you", "let's see",
# "I'll take a look", ...) from the START of a tutor turn. Applied to EVERY
# tutor turn (the tic shows up mid-dialogue too, not just turn 1). Conservative
# by design: it only fires on a curated set of meta openers and only consumes up
# to the first clause boundary ( , . ; : ? ! ), so it strips the opener phrase
# without eating the legitimate question/content that follows. The prompt
# instruction is still primary; this just cleans leftovers.
_META_LEAD = re.compile(
    r"^\s*(?:"
    r"let me (?:think|read|reread|re-read|see|look|check|"
    r"help(?: a little| you out)?|ask(?: you)?)"
    r"|let'?s (?:see|think|look|start)"
    r"|i'?ll (?:take a look|read|check|see|help)"
    r")\b[^.?!,;:]*[.?!,;:]+\s*",
    re.IGNORECASE)


def strip_leading_meta(text: str) -> str:
    """Strip a single leading meta-narration clause if present.

    Only removes a curated meta opener up to the first clause boundary, so
    legitimate content is preserved. Returns the original text if stripping
    would empty it (e.g. the whole turn was just the meta phrase)."""
    cleaned = _META_LEAD.sub("", text, count=1).lstrip()
    return cleaned if cleaned else text


def parse_turns(raw: str):
    """Parse the model's JSON turn list; tolerate accidental code fences.

    Enforces: non-empty list, valid turn shapes, starts on user, ends on
    assistant, and STRICT user/assistant alternation. Malformed generations
    (e.g. several tutor turns in a row) are rejected here so they never reach
    the dataset.
    """
    txt = _FENCE.sub("", raw.strip())
    turns = json.loads(txt)  # will raise if the model didn't obey — caught below
    if not isinstance(turns, list) or not turns:
        raise ValueError("not a non-empty list")
    for t in turns:
        if t.get("role") not in ("user", "assistant") or "content" not in t:
            raise ValueError("bad turn shape")
    if turns[0]["role"] != "user" or turns[-1]["role"] != "assistant":
        raise ValueError("must start on user, end on assistant")
    expected = ["user", "assistant"]
    for idx, t in enumerate(turns):
        if t["role"] != expected[idx % 2]:
            raise ValueError("turns do not strictly alternate user/assistant")
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
    ap.add_argument("--concurrency", type=int, default=12,
                    help="parallel generation requests (vLLM batches these; "
                         "8-16 is sane on one L4). 1 = sequential.")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.generator_url, api_key=args.api_key)

    seeds = list(SEED_LOADERS[args.subject](Path(args.seed)))
    # Shuffle deterministically BEFORE applying --limit so a trial samples
    # across many stories instead of taking the first N Q&A from one tale
    # (the seed file is ordered by story). Fixed seed => reproducible trials.
    random.Random(42).shuffle(seeds)
    if args.limit:
        seeds = seeds[:args.limit]
    if not seeds:
        print(f"[!] No seed items loaded from {args.seed}. Check the file format.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    skips = collections.Counter()        # reason -> count (reject rate/bias)
    story_drops = collections.Counter()  # story -> drop count (per-story bias)

    # Build the task list first (one per (seed, mode)); skip empty questions.
    tasks = []  # (i, seed, mode)
    for i, seed in enumerate(seeds):
        if not seed["question"]:
            print(f"  [skip] item {i}: empty question")
            skips["empty question"] += 1
            continue
        for mode in args.modes:
            tasks.append((i, seed, mode))

    def work(seed, mode):
        raw = generate(client, args.generator_model,
                       build_generation_prompt(seed, mode))
        turns = parse_turns(raw)
        # Backup meta-narration strip on EVERY tutor turn (the tic can appear
        # mid-dialogue). Prompt fix is primary; this cleans leftovers.
        for t in turns:
            if t["role"] == "assistant":
                t["content"] = strip_leading_meta(t["content"])
        messages = [{"role": "system",
                     "content": system_prompt_for(args.subject, mode)}]
        messages.extend(turns)
        return {"subject": args.subject, "mode": mode, "messages": messages}

    # Run generations concurrently (vLLM batches them); the OpenAI client is
    # thread-safe. Results are keyed by (i, mode) so we can write them back in
    # deterministic seed order regardless of completion order.
    results = {}
    done = 0
    print(f"[*] Generating {len(tasks)} dialogues at concurrency "
          f"{args.concurrency}...")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        fut_meta = {ex.submit(work, seed, mode): (i, seed, mode)
                    for (i, seed, mode) in tasks}
        for fut in as_completed(fut_meta):
            i, seed, mode = fut_meta[fut]
            done += 1
            try:
                results[(i, mode)] = fut.result()
            except Exception as e:  # noqa: BLE001 — surface + tally any failure
                reason = ("malformed JSON"
                          if isinstance(e, json.JSONDecodeError) else str(e))
                print(f"  [skip] item {i} mode {mode}: {reason}")
                skips[reason] += 1
                story_drops[seed["story"]] += 1
            if done % 200 == 0:
                print(f"  ...{done}/{len(tasks)} done "
                      f"({len(results)} ok, {sum(skips.values())} skipped)")

    # Write in deterministic (seed, mode) order — reproducible, not race-order.
    written = 0
    with open(out_path, "w") as fh:
        for i, seed in enumerate(seeds):
            for mode in args.modes:
                row = results.get((i, mode))
                if row is not None:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

    total_skipped = sum(skips.values())
    print(f"[✓] Wrote {written} dialogues to {out_path}  (skipped {total_skipped})")
    if skips:
        print("    Drop reasons (watch for content bias at scale):")
        for reason, n in skips.most_common():
            print(f"      {n:4d}  {reason}")
    if story_drops:
        n_stories = len({s["story"] for _, s, _ in tasks})
        print(f"    Drops span {len(story_drops)}/{n_stories} stories; "
              f"top offenders (watch for per-story bias):")
        for story, n in story_drops.most_common(10):
            print(f"      {n:4d}  {story}")
    print(f"    Next: human-review this file (REQUIRED for civics), then")
    print(f"    `python build_dataset.py --subject {args.subject}` to "
          f"validate + merge for training.")
    if args.subject == "civics":
        print("    ⚠ Civics: review every dialogue for neutrality before training.")


if __name__ == "__main__":
    main()
